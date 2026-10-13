#!/usr/bin/env python3
"""Preflight check for Multi-Site Train Identity Shadow Lab.
Validates site isolation, configuration schema, read-only permissions, and schedule integrity.
"""
import sys
import os
from pathlib import Path

# Add src to pythonpath
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "shadow_lab" / "src"))

from shadow_identity.config_loader import SiteRegistry
from shadow_identity.timetable_importer import TimetableImporter


def run_preflight(site_id: str = None) -> bool:
    print("=" * 60)
    print(" SHADOW LAB PREFLIGHT HEALTH CHECK")
    print("=" * 60)

    # 1. Writable workspace check
    workspace_root = repo_root / "shadow_lab" / "workspace"
    for sub in ["evidence", "reports", "logs", "fixtures"]:
        (workspace_root / sub).mkdir(parents=True, exist_ok=True)
    print(f"[PASS] Shadow workspace root verified at: {workspace_root}")

    # 2. Registry loading
    shadow_lab_root = repo_root / "shadow_lab"
    registry = SiteRegistry(shadow_lab_root=shadow_lab_root)

    available_sites = registry.list_sites()
    print(f"[INFO] Discovered sites: {available_sites}")
    if not available_sites:
        print("[FAIL] No sites found in site registry!")
        return False

    targets = [site_id] if site_id else available_sites
    for sid in targets:
        print(f"\n--- Validating site: {sid} ---")
        try:
            cfg = registry.get_site(sid)
        except Exception as e:
            print(f"[FAIL] Failed to load site configuration for {sid}: {e}")
            return False

        # Verify read-only guarantees
        shadow_cfg = cfg.get("shadow", {})
        if shadow_cfg.get("production_write_allowed") is not False:
            print(f"[FAIL] CRITICAL: production_write_allowed is True for {sid}!")
            return False

        ntes_cfg = cfg.get("ntes", {})
        if ntes_cfg.get("allow_network") is not False:
            print(f"[FAIL] CRITICAL: ntes.allow_network is True for {sid}! Must be replay_only.")
            return False
        print(f"[PASS] Zero-write & network-disabled constraints verified for {sid}")

        # Check schedule
        sched_cfg = cfg.get("schedule", {})
        sched_path_str = sched_cfg.get("path")
        if not sched_path_str:
            print(f"[FAIL] No schedule path configured for {sid}")
            return False

        sched_path = repo_root / sched_path_str
        if not sched_path.exists():
            print(f"[FAIL] Schedule file missing: {sched_path}")
            return False

        # Load timetable
        importer = TimetableImporter()
        station_cfg = cfg.get("station", {})
        snapshot = importer.import_schedule(
            site_id=sid,
            station_code=station_cfg.get("code", ""),
            csv_path=str(sched_path),
            version=sched_cfg.get("version", "unknown"),
            effective_from=sched_cfg.get("effective_from"),
        )
        print(f"[PASS] Timetable loaded: {len(snapshot.rows)} rows (checksum: {snapshot.checksum_sha256[:12]}...)")

        # Verify checksum if configured
        expected_hash = sched_cfg.get("checksum_sha256")
        if expected_hash and expected_hash != "SET_FROM_IMPORTED_FILE":
            if snapshot.checksum_sha256.lower() != expected_hash.lower():
                print(f"[FAIL] Schedule checksum mismatch for {sid}: expected {expected_hash}, got {snapshot.checksum_sha256}")
                return False
            print(f"[PASS] Schedule checksum matches verified reference.")

    print("\n" + "=" * 60)
    print(" PREFLIGHT CHECK COMPLETED SUCCESSFULLY: ALL SITES OPERATIONAL")
    print("=" * 60)
    return True


if __name__ == "__main__":
    sid_arg = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "--site-id" and len(sys.argv) > 2:
            sid_arg = sys.argv[2]
        else:
            sid_arg = sys.argv[1]
    ok = run_preflight(sid_arg)
    sys.exit(0 if ok else 1)
