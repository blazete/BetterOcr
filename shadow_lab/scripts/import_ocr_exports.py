#!/usr/bin/env python3
"""Imports historical per-frame OCR CSVs from train-identity-lab into shadow fixtures.
"""
import argparse
import json
import sys
from pathlib import Path

# Setup sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "shadow_lab" / "src"))

from shadow_identity.ocr_adapter import OcrAdapter


def main():
    parser = argparse.ArgumentParser(description="Import per-frame OCR CSV into shadow fixture")
    parser.add_argument("--csv", required=True, help="Path to per_frame_ocr.csv")
    parser.add_argument("--passage-id", required=True, help="Passage ID")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    adapter = OcrAdapter(confirmed_threshold=3)
    evidence = adapter.load_from_per_frame_csv(Path(args.csv), args.passage_id)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(evidence.to_dict(), f, indent=2)

    print(f"[PASS] Successfully imported OCR evidence to: {out_p}")
    print(f"       Numeric status: {evidence.passage_consensus.numeric_status}")
    print(f"       Stable pair: {evidence.passage_consensus.stable_pair}")


if __name__ == "__main__":
    main()
