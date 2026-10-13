#!/usr/bin/env python3
"""Verification script to ensure ARDS-main files have not been modified.
"""
import os
import sys
import hashlib
import json
from pathlib import Path


def verify_originals(manifest_path: str = "shadow_lab/workspace/fixtures/original_hashes.json") -> bool:
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        print(f"[FAIL] Baseline manifest not found: {manifest_path}")
        return False

    with open(manifest_file, "r") as fp:
        baseline_hashes = json.load(fp)

    mismatches = []
    missing = []
    checked = 0

    for file_path, expected_hash in baseline_hashes.items():
        p = Path(file_path)
        if not p.exists():
            missing.append(file_path)
            continue
        with open(p, "rb") as fp:
            current_hash = hashlib.sha256(fp.read()).hexdigest()
        if current_hash != expected_hash:
            mismatches.append((file_path, expected_hash, current_hash))
        checked += 1

    if missing:
        print(f"[FAIL] {len(missing)} files missing from production baseline:")
        for m in missing[:10]:
            print(f"  - {m}")
        return False

    if mismatches:
        print(f"[FAIL] {len(mismatches)} files have modified hashes:")
        for f, exp, curr in mismatches[:10]:
            print(f"  - {f}: expected {exp[:8]}... got {curr[:8]}...")
        return False

    print(f"[PASS] All {checked} original ARDS files verified unchanged (zero modifications).")
    return True


if __name__ == "__main__":
    manifest = sys.argv[1] if len(sys.argv) > 1 else "shadow_lab/workspace/fixtures/original_hashes.json"
    success = verify_originals(manifest)
    sys.exit(0 if success else 1)
