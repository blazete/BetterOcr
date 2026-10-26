#!/usr/bin/env python3
"""Verification script to ensure ARDS-main files have not been modified.
"""
import os
import sys
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def verify_originals_structured(
    manifest_path: str = "shadow_lab/workspace/fixtures/original_hashes.json",
) -> Dict[str, Any]:
    """Read-only structured integrity check of production ARDS files.

    Returns a dict with:
        passed (bool): True if all files match their baseline hashes.
        total_expected (int): Number of files in the baseline manifest.
        verified_count (int): Number of files successfully verified.
        missing_count (int): Number of files not found on disk.
        mismatch_count (int): Number of files with modified hashes.
        missing_files (list): Paths of missing files (first 10).
        mismatched_files (list): Paths of mismatched files (first 10).
    """
    manifest_file = Path(manifest_path)
    result: Dict[str, Any] = {
        "passed": False,
        "total_expected": 0,
        "verified_count": 0,
        "missing_count": 0,
        "mismatch_count": 0,
        "missing_files": [],
        "mismatched_files": [],
    }

    if not manifest_file.exists():
        return result

    with open(manifest_file, "r") as fp:
        baseline_hashes = json.load(fp)

    result["total_expected"] = len(baseline_hashes)
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
            mismatches.append(file_path)
        checked += 1

    result["verified_count"] = checked
    result["missing_count"] = len(missing)
    result["mismatch_count"] = len(mismatches)
    result["missing_files"] = missing[:10]
    result["mismatched_files"] = mismatches[:10]
    result["passed"] = (len(missing) == 0 and len(mismatches) == 0 and checked > 0)
    return result


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

