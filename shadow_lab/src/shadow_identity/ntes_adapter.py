"""
Read-Only ARDS NTES Adapter (§1 - §5)

Connects DelayedNtesVerificationWorker to the existing approved ARDS NTES
cache and scraper without modifying ARDS-main or writing anything outside shadow_lab.

Key Invariants:
1. Site Isolation: Station A (NDLS) never queries Station B (TESTX).
2. Zero Production Writes: audit_dir is strictly None; zero files written to ARDS.
3. Snapshots saved ONLY under shadow_lab/data/ntes_snapshots/<site_id>/.
4. Safe Fallbacks: Network / parsing failure returns an unavailable snapshot
   leaving the passage valid while marking Stage T2 unavailable.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shadow_identity.config_loader import ConfigError, SiteRegistry
from shadow_identity.models import NtesSnapshot, NtesTrainEntry

logger = logging.getLogger("shadow_identity.ntes_adapter")


class ArdsNtesAdapter:
    """
    Read-only adapter wrapping the approved ARDS NTES cache/scraper.
    Accepts site_id and uses that site's station code and platform configuration.
    """

    def __init__(
        self,
        shadow_lab_root: Optional[Path] = None,
        ards_root: Optional[Path] = None,
        site_registry: Optional[SiteRegistry] = None,
        force_network: bool = False,
    ):
        if shadow_lab_root is None:
            self.shadow_lab_root = Path(__file__).resolve().parents[2]
        else:
            self.shadow_lab_root = Path(shadow_lab_root).resolve()

        if ards_root is None:
            self.ards_root = self.shadow_lab_root.parent / "ARDS-main"
        else:
            self.ards_root = Path(ards_root).resolve()

        self.site_registry = site_registry or SiteRegistry(shadow_lab_root=self.shadow_lab_root)
        self.force_network = force_network
        self.snapshots_dir = self.shadow_lab_root / "data" / "ntes_snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._caches: Dict[str, Any] = {}
        self._ensure_ards_importable()

    def _ensure_ards_importable(self) -> None:
        """Add ARDS-main to sys.path in a read-only fashion if not already present."""
        ards_str = str(self.ards_root)
        if ards_str not in sys.path and self.ards_root.is_dir():
            sys.path.insert(0, ards_str)

    def _get_or_create_cache(self, site_id: str, mock_trains: Optional[List[Any]] = None) -> Any:
        """
        Instantiate or retrieve site-specific NtesCache instance.
        Enforces audit_dir=None to prevent any writes to ARDS.
        """
        if site_id in self._caches and mock_trains is None:
            return self._caches[site_id]

        site_cfg = self.site_registry.get_site(site_id)
        station_cfg = site_cfg.get("station", {})
        station_code = station_cfg.get("code", "")
        station_query = station_cfg.get("ntes_query") or f"{station_code} - {station_cfg.get('name', station_code)}"
        platforms = [str(p) for p in site_cfg.get("platforms", {}).get("configured", ["12", "13"])]

        ntes_cfg = site_cfg.get("ntes", {})
        ttl_s = int(ntes_cfg.get("ttl_s", 180))
        timeout_s = int(ntes_cfg.get("fetch_timeout_s", 8))
        replay_dir = ntes_cfg.get("replay_dir")

        try:
            from orchestrator.schedule.ntes_cache import NtesCache
        except ImportError as e:
            logger.error(f"Cannot import NtesCache from {self.ards_root}: {e}")
            raise RuntimeError(f"Approved ARDS NTES scraper not found at {self.ards_root}: {e}")

        # Strictly audit_dir=None: GUARANTEES zero writes to ARDS production tree!
        cache = NtesCache(
            ttl_s=ttl_s,
            timeout_s=timeout_s,
            mock=mock_trains is not None,
            mock_trains=mock_trains,
            station=station_query,
            platforms=platforms,
            audit_dir=None,
            replay_dir=replay_dir,
        )
        self._caches[site_id] = cache
        return cache

    def save_snapshot_locally(self, site_id: str, snap: NtesSnapshot) -> Path:
        """
        Save fetched snapshot strictly within shadow_lab/data/ntes_snapshots/<site_id>/.
        Guarantees ZERO mutation to production ARDS.
        """
        target_dir = self.snapshots_dir / site_id
        target_dir.mkdir(parents=True, exist_ok=True)
        ts_str = snap.fetched_at.replace(":", "-").replace(".", "-") if snap.fetched_at else str(int(time.time()))
        out_path = target_dir / f"snap_{ts_str}.json"

        # Integrity verification: must NEVER write outside shadow_lab
        resolved_str = str(out_path.resolve())
        shadow_root_str = str(self.shadow_lab_root.resolve())
        ards_root_str = str(self.ards_root.resolve())

        if not resolved_str.startswith(shadow_root_str):
            raise PermissionError(f"Security violation: attempt to write outside shadow_lab: {out_path}")
        if resolved_str.startswith(ards_root_str):
            raise PermissionError(f"Security violation: attempt to write into ARDS production: {out_path}")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(snap), f, indent=2)

        return out_path

    def get_snapshot(
        self,
        site_id: str,
        passage_end_time: Optional[str] = None,
        mock_trains: Optional[List[Any]] = None,
    ) -> NtesSnapshot:
        """
        Fetch or retrieve an NTES snapshot for the given site_id.
        Uses that site's station code and platforms.
        Guarantees Station A never fetches Station B.
        If fetch fails, returns an unavailable snapshot without raising,
        leaving the passage valid and Stage T2 marked unavailable.
        """
        req_iso = datetime.now().isoformat()

        # 1. Site Lookup & Station Verification
        site_cfg = self.site_registry.get_site(site_id)
        station_cfg = site_cfg.get("station", {})
        station_code = station_cfg.get("code", "")
        platforms = [str(p) for p in site_cfg.get("platforms", {}).get("configured", [])]

        ntes_cfg = site_cfg.get("ntes", {})
        allow_network = self.force_network or bool(ntes_cfg.get("allow_network", False))
        replay_dir = ntes_cfg.get("replay_dir")

        try:
            cache = self._get_or_create_cache(site_id, mock_trains=mock_trains)

            # Ensure cache is using this site's station and platforms (Site Isolation)
            if hasattr(cache, "_station"):
                expected_query = station_cfg.get("ntes_query") or station_code
                if station_code not in cache._station:
                    raise ConfigError(
                        f"Site isolation violation: cache station '{cache._station}' "
                        f"does not match site {site_id} ({station_code})!"
                    )

            # If mock or replay or network allowed, fetch from NtesCache
            if mock_trains is not None or replay_dir or allow_network:
                ards_snap = cache.get_snapshot()
            else:
                # In offline testing without network/replay, look for local snapshots under shadow_lab
                local_snaps = sorted((self.snapshots_dir / site_id).glob("*.json"))
                if local_snaps:
                    try:
                        with open(local_snaps[-1], "r", encoding="utf-8") as lf:
                            d = json.load(lf)
                        return NtesSnapshot(
                            site_id=site_id,
                            station_code=station_code,
                            requested_at=req_iso,
                            fetched_at=d.get("fetched_at", req_iso),
                            window_hr=d.get("window_hr", 2),
                            available=d.get("available", True),
                            from_cache=True,
                            stale_fallback=d.get("stale_fallback", False),
                            source="shadow_local_cache",
                            trains=[NtesTrainEntry(**t) for t in d.get("trains", [])],
                            raw_hash=d.get("raw_hash", ""),
                        )
                    except Exception:
                        pass
                ards_snap = None

            if ards_snap is not None:
                train_entries: List[NtesTrainEntry] = []
                for t in ards_snap.trains:
                    raw_no = str(getattr(t, "train_number", "")).strip().lstrip("0")
                    norm_no = f"{int(raw_no):05d}" if raw_no.isdigit() else raw_no
                    train_entries.append(
                        NtesTrainEntry(
                            train_number=norm_no,
                            train_name=str(getattr(t, "train_name", "")),
                            platform=str(getattr(t, "platform", "")),
                            time_min=getattr(t, "time_min", None),
                            time_raw=str(getattr(t, "time_raw", "")),
                            is_originating=bool(getattr(t, "is_originating", False)),
                            is_terminating=bool(getattr(t, "is_terminating", False)),
                            dep_min=getattr(t, "dep_min", None),
                        )
                    )

                fetched_dt = getattr(ards_snap, "fetched_at", None)
                fetched_str = (
                    fetched_dt.isoformat()
                    if isinstance(fetched_dt, datetime)
                    else (str(fetched_dt) if fetched_dt else req_iso)
                )

                snap = NtesSnapshot(
                    site_id=site_id,
                    station_code=station_code,
                    requested_at=req_iso,
                    fetched_at=fetched_str,
                    window_hr=getattr(ards_snap, "window_hr", 2),
                    available=bool(getattr(ards_snap, "available", True)),
                    from_cache=getattr(ards_snap, "stale_fallback", False),
                    stale_fallback=getattr(ards_snap, "stale_fallback", False),
                    source="ards_adapter",
                    trains=train_entries,
                    raw_hash="",
                )
                self.save_snapshot_locally(site_id, snap)
                return snap

        except Exception as exc:
            logger.warning(f"NTES fetch exception for site {site_id} ({station_code}): {exc}")

        # Failed NTES request: return unavailable snapshot
        # Crucial: passage remains valid, Stage T2 marked unavailable
        unavail_snap = NtesSnapshot(
            site_id=site_id,
            station_code=station_code,
            requested_at=req_iso,
            fetched_at=req_iso,
            window_hr=2,
            available=False,
            from_cache=False,
            stale_fallback=False,
            source="ards_adapter",
            trains=[],
            raw_hash="",
        )
        try:
            self.save_snapshot_locally(site_id, unavail_snap)
        except Exception as e:
            logger.warning(f"Could not save unavailable snapshot locally: {e}")
        return unavail_snap
