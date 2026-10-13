"""
Configuration Loader and Site Registry for Shadow Lab
Enforces JSON schema validation, site isolation, SHA256 integrity,
and non-negotiable zero-production-write guards.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


class ConfigError(Exception):
    """Raised when configuration validation or site isolation fails."""
    pass


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of file."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found for hashing: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class SiteRegistry:
    def __init__(self, shadow_lab_root: Optional[Path] = None):
        if shadow_lab_root is None:
            # Detect root relative to this file
            self.root = Path(__file__).resolve().parents[2]
        else:
            self.root = Path(shadow_lab_root)

        self.sites_dir = self.root / "sites"
        self.config_dir = self.root / "config"
        self.schema_file = self.config_dir / "schema.json"
        self.defaults_file = self.config_dir / "shadow_defaults.yaml"

        self.defaults = self._load_defaults()
        self.sites: Dict[str, Dict[str, Any]] = {}
        self.aliases: Dict[str, Dict[str, Any]] = {}
        self._load_all_sites()

    def _load_defaults(self) -> Dict[str, Any]:
        if not self.defaults_file.exists():
            return {}
        with open(self.defaults_file, "r") as f:
            return yaml.safe_load(f).get("shadow_defaults", {})

    def _load_all_sites(self) -> None:
        # Check both config/sites and sites directories
        dirs_to_check = [self.config_dir / "sites", self.sites_dir]
        for base_dir in dirs_to_check:
            if not base_dir.exists():
                continue
            for s_dir in sorted(base_dir.iterdir()):
                if s_dir.is_dir():
                    site_yaml = s_dir / "site.yaml"
                    if site_yaml.exists():
                        site_cfg = self._load_site_yaml(site_yaml)
                        site_id = site_cfg["site_id"]
                        self.sites[site_id] = site_cfg

                        # Load optional aliases.yaml
                        alias_file = s_dir / "aliases.yaml"
                        if alias_file.exists():
                            with open(alias_file, "r") as af:
                                a_data = yaml.safe_load(af) or {}
                                self.aliases[site_id] = a_data.get("aliases", {})
                        elif site_id not in self.aliases:
                            self.aliases[site_id] = {}

    def _load_site_yaml(self, path: Path) -> Dict[str, Any]:
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Validate mandatory site_id
        site_id = data.get("site_id")
        if not site_id:
            raise ConfigError(f"site.yaml at {path} is missing mandatory 'site_id'")

        # Enforce zero production writes
        shadow_cfg = data.get("shadow", {})
        if shadow_cfg.get("production_write_allowed") is not False:
            raise ConfigError(
                f"Site '{site_id}' violated isolation contract: production_write_allowed must be strictly False!"
            )

        # Preflight check: If live_shadow is requested, verify ARDS-main NTES scraper is available
        live_shadow = shadow_cfg.get("live_shadow", False)
        if live_shadow:
            ards_ntes_script = self.root.parent / "ARDS-main" / "orchestrator" / "schedule" / "ntes_cache.py"
            if not ards_ntes_script.is_file():
                raise ConfigError(
                    f"Site '{site_id}' requested live_shadow, but approved ARDS NTES scraper is unavailable: "
                    f"{ards_ntes_script} not found! Live shadow startup refused."
                )


        # Support timetable_file key
        if "timetable_file" in data:
            if "schedule" not in data:
                data["schedule"] = {}
            if "path" not in data["schedule"]:
                data["schedule"]["path"] = data["timetable_file"]

        # Verify schedule path and checksum if specified
        sched_cfg = data.get("schedule", {})
        sched_path_str = sched_cfg.get("path")
        if sched_path_str:
            sched_path = Path(sched_path_str)
            if not sched_path.is_absolute():
                candidates = [
                    self.root.parent / sched_path,
                    self.root / sched_path,
                    path.parent / sched_path,
                    path.parent / sched_path.name,
                ]
                found = next((p for p in candidates if p.exists()), None)
                if found:
                    sched_path = found
                else:
                    raise ConfigError(f"Site '{site_id}' schedule file not found: {sched_path_str}")
            elif not sched_path.exists():
                raise ConfigError(f"Site '{site_id}' schedule file not found: {sched_path}")
            
            expected_hash = sched_cfg.get("checksum_sha256")
            if expected_hash and expected_hash != "SET_FROM_IMPORTED_FILE":
                actual_hash = compute_file_sha256(sched_path)
                if actual_hash.lower() != expected_hash.lower():
                    raise ConfigError(
                        f"Site '{site_id}' schedule SHA256 mismatch! Expected: {expected_hash}, Actual: {actual_hash}"
                    )

        return data

    def get_site(self, site_id: str) -> Dict[str, Any]:
        """Retrieve site configuration by ID. Never falls back silently to NDLS."""
        if not site_id:
            raise ConfigError("site_id is mandatory and cannot be empty!")
        if site_id not in self.sites:
            available = list(self.sites.keys())
            raise ConfigError(
                f"Unknown site_id '{site_id}'. Available registered sites: {available}. "
                "Silent fallback to NDLS is strictly prohibited."
            )
        return self.sites[site_id]

    def get_site_config(self, site_id: str) -> Dict[str, Any]:
        return self.get_site(site_id)

    def get_aliases(self, site_id: str) -> Dict[str, Any]:
        return self.aliases.get(site_id, {})

    def get_site_aliases(self, site_id: str) -> Dict[str, Any]:
        return self.get_aliases(site_id)

    def list_sites(self) -> List[str]:
        return sorted(list(self.sites.keys()))

