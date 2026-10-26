"""
Regression tests for Package P0.3: Correct metrics; no more 'exact' from a direction pair.
"""
from pathlib import Path
import pytest
from shadow_identity.shadow_ocr.worker import ShadowWorker, WorkerJob


def test_evaluate_against_ground_truth_separates_exact_and_pair(tmp_path):
    # Setup temporary ground truth CSV
    gt_dir = tmp_path / "ground_truth"
    gt_dir.mkdir()
    gt_csv = gt_dir / "ground_truth.csv"
    gt_csv.write_text(
        "video_id,filename,actual_train_number,actual_train_name\n"
        "video_001,video_001.mp4,15708,AMRAPALI EXPRESS\n"
        "video_007,video_007.mp4,18103/18104,JALLIANWALA BAGH EXP\n"
    )

    worker = ShadowWorker(
        inbox_dir=tmp_path / "inbox",
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "output",
        ground_truth_dir=gt_dir,
    )

    # Job 1: video_001, predicted 15708 (matches single active GT 15708)
    job1 = WorkerJob(
        job_id="job_001",
        passage_id="video_001",
        site_id="ndls",
        camera_id="cam1",
        original_mp4_path=str(tmp_path / "video_001.mp4"),
        mp4_sha256="sha001",
        file_size_bytes=1000,
        status="COMPLETED",
        discovered_at_epoch=100.0,
        resolved_train_number="15708",
        resolver_status="PROVISIONAL",
    )
    worker.jobs["job_001"] = job1

    # Job 2: video_007, predicted 18103 (GT is pair 18103/18104)
    job2 = WorkerJob(
        job_id="job_007",
        passage_id="video_007",
        site_id="ndls",
        camera_id="cam1",
        original_mp4_path=str(tmp_path / "video_007.mp4"),
        mp4_sha256="sha007",
        file_size_bytes=1000,
        status="COMPLETED",
        discovered_at_epoch=100.0,
        resolved_train_number="18103",
        resolver_status="PROVISIONAL",
    )
    worker.jobs["job_007"] = job2

    evals = worker.evaluate_against_ground_truth()
    eval_by_id = {e["passage_id"]: e for e in evals}

    # Video 001 has single active service GT:
    e1 = eval_by_id["video_001"]
    assert e1["exact_active_service_match"] is True
    assert e1["pair_membership_match"] is True
    assert e1["direction_label_quality"] == "SINGLE_ACTIVE_SERVICE"

    # Video 007 has pair GT (18103/18104):
    e2 = eval_by_id["video_007"]
    # Exact active service MUST be None because active direction is unverified
    assert e2["exact_active_service_match"] is None, (
        f"Expected exact_active_service_match to be None for pair GT, got {e2['exact_active_service_match']}"
    )
    assert e2["pair_membership_match"] is True
    assert e2["direction_label_quality"] == "UNVERIFIED_ACTIVE_DIRECTION"


def test_first_four_digits_shortcut_removed(tmp_path):
    # Testing that 12345 matching 12349 via first 4 digits does NOT produce service_family_match without family map
    gt_dir = tmp_path / "ground_truth"
    gt_dir.mkdir()
    gt_csv = gt_dir / "ground_truth.csv"
    gt_csv.write_text(
        "video_id,filename,actual_train_number,actual_train_name\n"
        "video_999,video_999.mp4,12345,SOME EXP\n"
    )
    worker = ShadowWorker(
        inbox_dir=tmp_path / "inbox",
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "output",
        ground_truth_dir=gt_dir,
    )
    # Predicted 12349 (different train in same number block)
    job = WorkerJob(
        job_id="job_999",
        passage_id="video_999",
        site_id="ndls",
        camera_id="cam1",
        original_mp4_path=str(tmp_path / "video_999.mp4"),
        mp4_sha256="sha999",
        file_size_bytes=1000,
        status="COMPLETED",
        discovered_at_epoch=100.0,
        resolved_train_number="12349",
        resolver_status="PROVISIONAL",
    )
    worker.jobs["job_999"] = job

    evals = worker.evaluate_against_ground_truth()
    assert evals[0]["service_family_match"] is False, (
        "First 4 digits prefix match must NOT automatically confer service_family_match"
    )
