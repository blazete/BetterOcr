"""
NTES Snapshot Archive and Offline Replay Engine
Parses recorded NTES JSONL audit logs, extracts station boards for specific
timestamps without network queries, and models T1 and T2 (+180s) verification.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from shadow_identity.models import NtesSnapshot, NtesTrainEntry


def parse_ntes_train_entry(t_dict: Dict[str, Any]) -> NtesTrainEntry:
    raw_num = str(t_dict.get("train_number", "")).strip().lstrip("0")
    if raw_num.isdigit():
        train_no = f"{int(raw_num):05d}"
    else:
        train_no = raw_num

    return NtesTrainEntry(
        train_number=train_no,
        train_name=str(t_dict.get("train_name", "")),
        platform=str(t_dict.get("platform", "")),
        time_min=t_dict.get("time_min"),
        time_raw=str(t_dict.get("time_raw", "")),
        is_originating=bool(t_dict.get("is_originating", False)),
        is_terminating=bool(t_dict.get("is_terminating", False)),
        dep_min=t_dict.get("dep_min"),
    )


class NtesArchiveReplayer:
    def __init__(self, audit_dir: Optional[Path] = None):
        self.snapshots: List[NtesSnapshot] = []
        if audit_dir and Path(audit_dir).exists():
            self.load_audit_directory(Path(audit_dir))

    def load_audit_directory(self, audit_dir: Path) -> int:
        """Load all ntes_*.jsonl files from audit directory."""
        count = 0
        for p in sorted(audit_dir.glob("ntes_*.jsonl")):
            count += self.load_jsonl_file(p)
        return count

    def load_jsonl_file(self, jsonl_path: Path) -> int:
        """Load single JSONL audit file."""
        count = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    snap = self._parse_json_record(data)
                    self.snapshots.append(snap)
                    count += 1
                except Exception:
                    continue
        # Sort snapshots chronologically by fetched_at
        self.snapshots.sort(key=lambda s: s.fetched_at)
        return count

    def _parse_json_record(self, data: Dict[str, Any]) -> NtesSnapshot:
        fetched_at_str = data.get("fetched_at", "")
        station_str = data.get("station", "NDLS")
        stn_code = station_str.split()[0].replace("-", "").strip() if station_str else "NDLS"
        window = int(data.get("window_hr", data.get("nHr", 2)))

        train_objs: List[NtesTrainEntry] = []
        raw_trains = data.get("trains", [])
        for rt in raw_trains:
            train_objs.append(parse_ntes_train_entry(rt))

        return NtesSnapshot(
            site_id="replayed_site",
            station_code=stn_code,
            requested_at=fetched_at_str,
            fetched_at=fetched_at_str,
            window_hr=window,
            available=data.get("available", True),
            from_cache=data.get("from_cache", False),
            stale_fallback=data.get("stale_fallback", False),
            source="archive_replay",
            trains=train_objs,
            raw_hash="",
        )

    def find_nearest_snapshot(
        self, station_code: str, target_dt: datetime, max_delta_s: float = 3600.0
    ) -> Optional[NtesSnapshot]:
        """
        Find closest snapshot fetched at or immediately before target_dt.
        Ensures strict causality: NEVER leaks a future snapshot into T0/T1 decisions!
        """
        best: Optional[NtesSnapshot] = None
        min_diff = max_delta_s

        for s in self.snapshots:
            if s.station_code != station_code:
                continue
            try:
                s_dt = datetime.fromisoformat(s.fetched_at)
                diff = (target_dt - s_dt).total_seconds()
                # Causal: snapshot must be fetched AT or BEFORE target_dt
                if 0 <= diff <= min_diff:
                    min_diff = diff
                    best = s
            except Exception:
                continue

        if best:
            # Set snapshot age at decision time
            best.snapshot_age_seconds_at_decision = min_diff
        return best

    def get_snapshot_before(
        self, station_code: str, timestamp: datetime, max_delta_s: float = 3600.0
    ) -> Optional[NtesSnapshot]:
        """Convenience alias for find_nearest_snapshot enforcing causal ordering."""
        return self.find_nearest_snapshot(station_code, timestamp, max_delta_s)

    def find_post_passage_snapshot(
        self, station_code: str, passage_end_dt: datetime, offset_s: float = 180.0
    ) -> Optional[NtesSnapshot]:
        """
        Locate snapshot available around passage_end + 180s.
        """
        target_post_dt = passage_end_dt + timedelta(seconds=offset_s)
        # Find snapshot closest to target_post_dt within ±120s
        best: Optional[NtesSnapshot] = None
        min_diff = 120.0

        for s in self.snapshots:
            if s.station_code != station_code:
                continue
            try:
                s_dt = datetime.fromisoformat(s.fetched_at)
                diff = abs((s_dt - target_post_dt).total_seconds())
                if diff <= min_diff:
                    min_diff = diff
                    best = s
            except Exception:
                continue

        return best
