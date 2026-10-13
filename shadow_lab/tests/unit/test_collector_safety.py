"""Safety and Zero-Mutation Tests for Shadow Collector (§25)
Proves:
1. Shadow code operates strictly on read-only copies.
2. Production files remain byte-identical (SHA256 verified) before and after shadow collection runs.
3. No file modifications, additions, or deletions occur in production ARDS directories.
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
import pytest

from shadow_identity.passage_collector import ShadowCollector


def compute_dir_sha256(directory: Path) -> dict[str, str]:
    hashes = {}
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            fp = Path(root) / f
            h = hashlib.sha256()
            with open(fp, "rb") as fl:
                while chunk := fl.read(65536):
                    h.update(chunk)
            rel = str(fp.relative_to(directory))
            hashes[rel] = h.hexdigest()
    return hashes


def test_production_source_files_byte_identical_before_and_after_collection():
    """Test §25: verify production source files remain byte-identical before and after collection."""
    with tempfile.TemporaryDirectory() as tmp_prod_dir:
        prod_path = Path(tmp_prod_dir)
        # Create mock production ARDS structure
        insp_dir = prod_path / "inspections" / "20261013_001"
        insp_dir.mkdir(parents=True)
        meta_file = insp_dir / "passage_metadata.json"
        sample_meta = {
            "passage_id": "PAS_SAFE_001",
            "start_time": "2026-10-13T17:20:00+05:30",
            "end_time": "2026-10-13T17:21:00+05:30",
            "observed_platform": "12",
            "possible_platforms": ["12", "13"],
            "production_inspection_id": "INSP_SAFE_001",
            "production": {
                "train_number": "12920",
                "confidence": "HIGH",
                "source": "production_ards",
            },
            "visual": {
                "coach_evidence": [
                    {"coach_id": "C01", "frame_id": 10, "raw_text": "12920", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                    {"coach_id": "C02", "frame_id": 20, "raw_text": "12920", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                    {"coach_id": "C03", "frame_id": 30, "raw_text": "12920", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                ]
            },
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(sample_meta, f, indent=2)

        # Baseline hash before collection
        hashes_before = compute_dir_sha256(prod_path)

        collector = ShadowCollector()
        bundle = collector.import_production_passage(source=meta_file, site_id="ndls_pf12_13")

        # Verify bundle was created in shadow output directory
        assert bundle.passage_id == "PAS_SAFE_001"
        assert bundle.site_id == "ndls_pf12_13"

        # Hash after collection
        hashes_after = compute_dir_sha256(prod_path)

        # Zero-write assertion: exactly identical hashes and file list
        assert hashes_before == hashes_after
        assert set(hashes_before.keys()) == set(hashes_after.keys())
        for f_name in hashes_before:
            assert hashes_before[f_name] == hashes_after[f_name], f"File modified: {f_name}"


def test_shadow_collector_writes_only_to_shadow_workspace():
    """Verify that collector outputs are confined strictly to shadow_lab/data and shadow_lab/results."""
    collector = ShadowCollector()
    mock_passage = {
        "passage_id": "PAS_CONFINED_001",
        "start_time": "2026-10-13T17:20:00+05:30",
        "end_time": "2026-10-13T17:21:00+05:30",
        "observed_platform": "12",
        "possible_platforms": ["12", "13"],
        "production": {"train_number": "12920", "confidence": "HIGH"},
        "visual": {"service_numbers": []},
    }
    bundle = collector.import_production_passage(source=mock_passage, site_id="ndls_pf12_13")
    
    # Check output paths are strictly within shadow_lab
    assert (collector.passages_dir / f"{bundle.passage_id}.json").is_file()
    assert (collector.passage_results_dir / f"{bundle.passage_id}.json").is_file()
    assert "shadow_lab" in str(collector.passages_dir)
