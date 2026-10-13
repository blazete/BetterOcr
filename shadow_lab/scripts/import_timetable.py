#!/usr/bin/env python3
"""CLI utility to import and validate station timetables into shadow lab.
"""
import argparse
import sys
from pathlib import Path

# Setup sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "shadow_lab" / "src"))

from shadow_identity.config_loader import SiteRegistry
from shadow_identity.timetable_importer import TimetableImporter


def main():
    parser = argparse.ArgumentParser(description="Import station timetable CSV")
    parser.add_argument("--site-id", required=True, help="Site identifier")
    parser.add_argument("--csv", required=False, help="Optional override CSV path")
    args = parser.parse_args()

    registry = SiteRegistry(shadow_lab_root=repo_root / "shadow_lab")
    cfg = registry.get_site(args.site_id)

    sched_path = Path(args.csv) if args.csv else (repo_root / cfg["schedule"]["path"])
    if not sched_path.exists():
        print(f"[FAIL] Timetable file not found: {sched_path}")
        sys.exit(1)

    snapshot = TimetableImporter.import_schedule(
        site_id=args.site_id,
        station_code=cfg["station"]["code"],
        csv_path=str(sched_path),
        version=cfg["schedule"]["version"],
        effective_from=cfg["schedule"]["effective_from"],
    )

    print(f"[PASS] Successfully imported {len(snapshot.rows)} rows for site '{args.site_id}'")
    print(f"       Checksum SHA256: {snapshot.checksum_sha256}")


if __name__ == "__main__":
    main()
