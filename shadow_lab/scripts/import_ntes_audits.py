#!/usr/bin/env python3
"""CLI utility to import and parse offline NTES audit JSONL logs into shadow archive.
"""
import argparse
import sys
from pathlib import Path

# Setup sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "shadow_lab" / "src"))

from shadow_identity.ntes_archive import NtesArchiveReplayer


def main():
    parser = argparse.ArgumentParser(description="Import NTES audit logs into shadow archive")
    parser.add_argument("--audit-dir", required=True, help="Directory containing ntes_*.jsonl files")
    args = parser.parse_args()

    audit_path = Path(args.audit_dir)
    if not audit_path.exists():
        print(f"[FAIL] Audit directory not found: {audit_path}")
        sys.exit(1)

    replayer = NtesArchiveReplayer(audit_dir=audit_path)
    print(f"[PASS] Successfully loaded {len(replayer.snapshots)} NTES snapshots into shadow archive.")


if __name__ == "__main__":
    main()
