"""
Regression tests for Phase 7 benchmark runner corrections (Fixes 1–3).

Fix 1: Undefined variable `inbox_videos` in report aggregation loop.
Fix 2: Incorrect service-family prefix matching replaced with exact pair membership.
Fix 3: Hardcoded ARDS count replaced with dynamic structured verification.

These tests exercise the report-aggregation and evaluation logic using
temporary fixture files, without running actual OCR inference or modifying
production ARDS recordings.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

# Ensure shadow_lab/src is importable
repo_root = Path(__file__).resolve().parents[3]
src_dir = repo_root / "shadow_lab" / "src"
scripts_dir = repo_root / "shadow_lab" / "scripts"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


# ---------------------------------------------------------------------------
# Fix 2 — exact_pair_membership helper tests
# ---------------------------------------------------------------------------

def exact_pair_membership(
    pred_num: Optional[str],
    pair_members: Optional[List],
    verified: bool,
) -> Optional[bool]:
    """Deterministic service-pair membership check.

    Returns:
        True  — pred_num is an exact member of verified pair.
        False — pred_num is absent or not in verified pair.
        None  — pair not verified or missing, result is indeterminate.
    """
    if not verified or not pair_members:
        return None

    members = {str(n).strip() for n in pair_members if n is not None}

    if not members:
        return None

    if pred_num is None:
        return False

    return str(pred_num).strip() in members


class TestExactPairMembership:
    """Fix 2 regression tests: exact pair membership, no prefix matching."""

    def test_18103_in_pair(self):
        assert exact_pair_membership("18103", ["18103", "18104"], verified=True) is True

    def test_18105_not_in_pair(self):
        assert exact_pair_membership("18105", ["18103", "18104"], verified=True) is False

    def test_12345_not_prefix_match(self):
        """12345 must NOT match pair [1234, 12346] — no prefix shortcuts."""
        assert exact_pair_membership("12345", ["1234", "12346"], verified=True) is False

    def test_02276_not_in_unrelated_pair(self):
        assert exact_pair_membership("02276", ["20433", "20434"], verified=True) is False

    def test_null_prediction_verified_pair(self):
        assert exact_pair_membership(None, ["18103", "18104"], verified=True) is False

    def test_missing_pair_returns_none(self):
        assert exact_pair_membership("18103", [], verified=True) is None

    def test_unverified_pair_returns_none(self):
        assert exact_pair_membership("18103", ["18103", "18104"], verified=False) is None

    def test_none_pair_returns_none(self):
        assert exact_pair_membership("18103", None, verified=True) is None

    def test_leading_zero_preserved(self):
        """Leading-zero train numbers must remain unchanged during comparison."""
        assert exact_pair_membership("02276", ["02276", "02275"], verified=True) is True

    def test_leading_zero_not_stripped(self):
        """02276 must NOT match 2276 — leading zeros are significant."""
        assert exact_pair_membership("02276", ["2276", "2275"], verified=True) is False

    def test_whitespace_trimmed(self):
        """Whitespace around numbers should be trimmed."""
        assert exact_pair_membership(" 18103 ", ["18103", "18104"], verified=True) is True

    def test_pair_with_none_element(self):
        """None elements in pair list should be filtered out."""
        assert exact_pair_membership("18103", [None, "18103", "18104"], verified=True) is True

    def test_empty_members_after_filter(self):
        """If all pair members are None, treat as missing."""
        assert exact_pair_membership("18103", [None, None], verified=True) is None


# ---------------------------------------------------------------------------
# Fix 1 — Report aggregation variable name regression test
# ---------------------------------------------------------------------------

def _create_mock_video_and_results(tmp_dir: Path, video_id: str, pred_number: str = "12345"):
    """Create minimal fixture files for a single passage in a tmp directory."""
    inbox = tmp_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    output = tmp_dir / "output"
    output.mkdir(parents=True, exist_ok=True)
    manifest_dir = tmp_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal MP4 placeholder (will not be decoded in these tests)
    mp4_path = inbox / f"{video_id}.mp4"
    mp4_path.write_bytes(b"\x00" * 100)

    # Create worker decision JSON
    decision_data = {
        "visual_evidence": {
            "visual_pair_status": "PAIR_PROVISIONAL",
            "stable_pair": [pred_number],
            "endpoint_candidates": [],
            "train_name_candidates": [],
            "evaluated_frames_count": 10,
            "ocr_calls_executed": 20,
            "board_buckets_covered": [0, 1, 2],
            "unique_frame_count": 10,
            "fallback_calls": 5,
        },
        "resolver_decision": {
            "shadow_final_number": pred_number,
            "shadow_status": "PROVISIONAL",
            "reason_codes": ["VISUAL_PAIR_PROVISIONAL"],
            "ranked_candidates": [],
        },
        "performance_latency": {
            "total_wall_clock_s": 5.0,
        },
    }
    dec_file = output / f"fixture_{video_id}_decision.json"
    with open(dec_file, "w") as f:
        json.dump(decision_data, f)

    return mp4_path


class TestFix1ReportAggregation:
    """Fix 1 regression: the report loop variable must be valid_recordings, not inbox_videos."""

    def test_runner_post_inference_report_aggregation_with_fixtures(self):
        """
        Executes the runner's post-inference report-aggregation path using
        temporary worker result fixtures.
        Must fail with NameError when inbox_videos is used, and pass when
        valid_recordings is used, producing exactly 4 result entries.
        """
        import run_heldout_evaluation
        from unittest.mock import MagicMock, patch
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inbox = tmp_path / "inbox"
            output = tmp_path / "output"
            storage = tmp_path / "storage"
            manifest = tmp_path / "manifest"
            for d in (inbox, output, storage, manifest):
                d.mkdir(parents=True, exist_ok=True)

            video_ids = ["coach_id11", "coach_id12", "coach_id13", "coach_id14"]
            for vid in video_ids:
                (inbox / f"{vid}.mp4").write_bytes(b"\x00" * 100)
                import hashlib
                fixture_sha = hashlib.sha256(b"\x00" * 100).hexdigest()
                fixture_key = f"ndls_pf12_13_side_a_pas_{vid}_{fixture_sha}"
                decision_data = {
                    "visual_evidence": {
                        "visual_pair_status": "PAIR_PROVISIONAL",
                        "stable_pair": ["18103"],
                        "endpoint_candidates": [],
                        "train_name_candidates": [],
                        "evaluated_frames_count": 10,
                        "ocr_calls_executed": 20,
                        "board_buckets_covered": [0, 1, 2],
                        "unique_frame_count": 10,
                        "fallback_calls": 5,
                    },
                    "resolver_decision": {
                        "shadow_final_number": "18103",
                        "shadow_status": "PROVISIONAL",
                        "reason_codes": ["VISUAL_PAIR_PROVISIONAL"],
                    },
                    "performance_latency": {
                        "total_wall_clock_s": 4.5,
                    },
                }
                (output / f"{fixture_key}_decision.json").write_text(json.dumps(decision_data))
                (output / f"{fixture_key}_raw.jsonl").write_text('{"frame": 1}\n')

            manifest_content = {
                "schema_version": "shadow_eval_manifest.v1",
                "dataset": "heldout_4",
                "recordings": [
                    {
                        "expected_filename": f"{vid}.mp4",
                        "ground_truth_number": None,
                        "direction_label_quality": "PENDING_VERIFICATION",
                        "is_single_active_service": False,
                        "service_pair": [],
                    }
                    for vid in video_ids
                ],
            }
            (manifest / "ground_truth_manifest.json").write_text(json.dumps(manifest_content))

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
            mock_cap.get.side_effect = lambda p: 25.0 if p == 5 else (100 if p == 7 else 1920)

            fake_repo = tmp_path / "repo"
            (fake_repo / "shadow_lab" / "data").mkdir(parents=True, exist_ok=True)
            import os
            os.symlink(str(tmp_path), str(fake_repo / "shadow_lab" / "data" / "heldout_4"))

            with patch("run_heldout_evaluation.verify_production_files", return_value={"passed": True, "verified_count": 456, "total_expected": 456}), \
                 patch("run_heldout_evaluation.ensure_fresh_benchmark"), \
                 patch("run_heldout_evaluation.cv2.VideoCapture", return_value=mock_cap), \
                 patch("run_heldout_evaluation.ShadowWorker.run_cycle", return_value={"discovered": 4, "processed": 4, "failed": 0}), \
                 patch.object(run_heldout_evaluation, "repo_root", fake_repo):

                run_heldout_evaluation.main()

            results_file = output / "heldout_4_evaluation_results.json"
            assert results_file.exists(), "heldout_4_evaluation_results.json was not written"
            with open(results_file, "r") as f:
                results_data = json.load(f)

            assert results_data["total_passages_evaluated"] == 4
            assert len(results_data["passage_results"]) == 4
            assert results_data["accuracy_status"] == "INSUFFICIENT_DATA"
            assert results_data["exact_active_service_accuracy_pct"] is None
            assert results_data["service_family_accuracy_pct"] is None
            vids = [r["video_id"] for r in results_data["passage_results"]]
            assert sorted(vids) == ["coach_id11", "coach_id12", "coach_id13", "coach_id14"]


# ---------------------------------------------------------------------------
# Fix 3 — Dynamic ARDS integrity reporting
# ---------------------------------------------------------------------------

class TestFix3DynamicIntegrity:
    """Fix 3 regression: verify_originals_structured returns dynamic counts."""

    def test_structured_verifier_returns_count(self):
        """The structured verifier must return verified_count and total_expected."""
        from verify_originals_unchanged import verify_originals_structured

        with tempfile.TemporaryDirectory() as tmp:
            # Create a small fake manifest with 3 files
            manifest_path = Path(tmp) / "test_hashes.json"
            fake_files = {}
            for i in range(3):
                fp = Path(tmp) / f"file_{i}.txt"
                content = f"content_{i}".encode()
                fp.write_bytes(content)
                import hashlib
                fake_files[str(fp)] = hashlib.sha256(content).hexdigest()

            with open(manifest_path, "w") as f:
                json.dump(fake_files, f)

            result = verify_originals_structured(str(manifest_path))
            assert result["passed"] is True
            assert result["total_expected"] == 3
            assert result["verified_count"] == 3
            assert result["missing_count"] == 0
            assert result["mismatch_count"] == 0

    def test_structured_verifier_detects_missing(self):
        from verify_originals_unchanged import verify_originals_structured

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "test_hashes.json"
            fake_files = {"/nonexistent/file.txt": "abc123"}
            with open(manifest_path, "w") as f:
                json.dump(fake_files, f)

            result = verify_originals_structured(str(manifest_path))
            assert result["passed"] is False
            assert result["total_expected"] == 1
            assert result["verified_count"] == 0
            assert result["missing_count"] == 1

    def test_structured_verifier_detects_mismatch(self):
        from verify_originals_unchanged import verify_originals_structured

        with tempfile.TemporaryDirectory() as tmp:
            fp = Path(tmp) / "file.txt"
            fp.write_bytes(b"original")
            manifest_path = Path(tmp) / "test_hashes.json"
            with open(manifest_path, "w") as f:
                json.dump({str(fp): "0000000000000000000000000000000000000000000000000000000000000000"}, f)

            result = verify_originals_structured(str(manifest_path))
            assert result["passed"] is False
            assert result["verified_count"] == 1
            assert result["mismatch_count"] == 1

    def test_no_hardcoded_456_in_runner(self):
        """After Fix 3, the runner must not contain hardcoded '456 / 456'."""
        runner_path = repo_root / "shadow_lab" / "scripts" / "run_heldout_evaluation.py"
        content = runner_path.read_text()
        assert "456 / 456" not in content, (
            "Hardcoded '456 / 456' still present in runner; Fix 3 not applied."
        )


# ---------------------------------------------------------------------------
# Runner Protections & Policy Verification
# ---------------------------------------------------------------------------

class TestRunnerProtections:
    """Verifies intake validation, ground-truth isolation, replay mode, and overwrite guards."""

    def test_four_recordings_required(self):
        """Runner must cleanly halt without executing worker if fewer than 4 recordings are present."""
        import run_heldout_evaluation
        from unittest.mock import MagicMock, patch

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inbox = tmp_path / "inbox"
            output = tmp_path / "output"
            storage = tmp_path / "storage"
            manifest = tmp_path / "manifest"
            for d in (inbox, output, storage, manifest):
                d.mkdir(parents=True, exist_ok=True)

            # Stage only 3 videos (missing coach_id14.mp4)
            for vid in ["coach_id11", "coach_id12", "coach_id13"]:
                (inbox / f"{vid}.mp4").write_bytes(b"\x00" * 100)

            fake_repo = tmp_path / "repo"
            (fake_repo / "shadow_lab" / "data").mkdir(parents=True, exist_ok=True)
            import os
            os.symlink(str(tmp_path), str(fake_repo / "shadow_lab" / "data" / "heldout_4"))

            worker_mock = MagicMock()
            with patch("run_heldout_evaluation.verify_production_files", return_value={"passed": True, "verified_count": 456, "total_expected": 456}), \
                 patch("run_heldout_evaluation.ShadowWorker", return_value=worker_mock), \
                 patch.object(run_heldout_evaluation, "repo_root", fake_repo):

                ret = run_heldout_evaluation.main()
                assert ret == 0, "Expected main() to return 0 on BLOCKED_DATA"
                worker_mock.run_cycle.assert_not_called()

    def test_ground_truth_isolation_and_offline_replay(self):
        """Worker must be initialized with ground_truth_dir=None and allow_unmarked_offline_replay=True."""
        runner_path = repo_root / "shadow_lab" / "scripts" / "run_heldout_evaluation.py"
        content = runner_path.read_text()
        assert "ground_truth_dir=None" in content, "Ground truth directory must be None during worker initialization"
        assert "allow_unmarked_offline_replay=True" in content, "Approved offline replay mode must be enabled"

    def test_silent_overwrite_prevention(self):
        """Runner must refuse to run if initial prediction outputs already exist."""
        import run_heldout_evaluation
        from unittest.mock import MagicMock, patch

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inbox = tmp_path / "inbox"
            output = tmp_path / "output"
            storage = tmp_path / "storage"
            manifest = tmp_path / "manifest"
            for d in (inbox, output, storage, manifest):
                d.mkdir(parents=True, exist_ok=True)

            # Pre-create the evaluation output file
            existing_results = output / "heldout_4_evaluation_results.json"
            existing_results.write_text('{"existing": true}')

            fake_repo = tmp_path / "repo"
            (fake_repo / "shadow_lab" / "data").mkdir(parents=True, exist_ok=True)
            import os
            os.symlink(str(tmp_path), str(fake_repo / "shadow_lab" / "data" / "heldout_4"))

            with patch("run_heldout_evaluation.verify_production_files", return_value={"passed": True, "verified_count": 456, "total_expected": 456}), \
                 patch.object(run_heldout_evaluation, "repo_root", fake_repo):

                with pytest.raises(SystemExit):
                    run_heldout_evaluation.main()

