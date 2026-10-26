"""
Regression tests for Task 2: Discovery-to-Staging Source Identity & Integrity.

Requirements:
- Preserve the discovery hash in immutable job metadata.
- Revalidate source finalization and source identity immediately before staging.
- Copy exclusively into shadow-owned temporary storage.
- Compare the staged file hash and size with original discovery hash and size.
- Reject changed sources ("SOURCE_CHANGED_SINCE_DISCOVERY", "SOURCE_SIZE_CHANGED_SINCE_DISCOVERY").
- Ensure rejected files never enter OCR inference.
- Handle a different hash for the same passage as a distinct, auditable event without overwriting previous passage.
- Prevent symlink replacement or source-file swapping.
"""

from pathlib import Path
import json
import time
import pytest
import cv2
import numpy as np
from unittest.mock import patch

from shadow_identity.shadow_ocr.worker import ShadowWorker, compute_file_sha256
from shadow_identity.continuous_evaluation import generate_worker_daily_review_report


def _create_dummy_mp4(path: Path, frames: int = 5):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 64))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(frames):
        out.write(frame)
    out.release()


def test_staging_rejects_source_hash_changed_since_discovery(tmp_path):
    """File content modified between discovery and staging must be rejected."""
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    storage = tmp_path / "storage"
    out_dir = tmp_path / "out"

    mp4 = inbox / "pas_test_01.mp4"
    _create_dummy_mp4(mp4, frames=5)
    (inbox / "pas_test_01.done").write_text("DONE")

    worker = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=storage,
        output_dir=out_dir,
        size_stability_interval_sec=0.0,
    )

    # 1. Discovery
    discovered = worker.discover_new_recordings()
    assert len(discovered) == 1
    assert discovered[0].resolve() == mp4.resolve()

    # 2. Deliberately modify file content before staging (change content)
    _create_dummy_mp4(mp4, frames=10)

    # 3. Staging must detect modification and reject job
    job = worker.stage_job(discovered[0])
    assert job.status == "FAILED"
    assert "SOURCE_CHANGED_SINCE_DISCOVERY" in (job.error_message or "") or "SOURCE_SIZE_CHANGED_SINCE_DISCOVERY" in (job.error_message or "")

    # 4. Job must never enter OCR inference
    claimed = worker.claim_job_for_processing(job.job_id)
    assert claimed is None
    assert worker.process_job(job) is False


def test_staging_rejects_source_size_changed_since_discovery(tmp_path):
    """File size modified between discovery and staging must be rejected."""
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    storage = tmp_path / "storage"
    out_dir = tmp_path / "out"

    mp4 = inbox / "pas_test_02.mp4"
    _create_dummy_mp4(mp4, frames=5)
    (inbox / "pas_test_02.done").write_text("DONE")

    worker = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=storage,
        output_dir=out_dir,
        size_stability_interval_sec=0.0,
    )

    discovered = worker.discover_new_recordings()
    assert len(discovered) == 1

    # Append bytes to change file size
    with open(mp4, "ab") as f:
        f.write(b"\x00" * 128)

    job = worker.stage_job(discovered[0])
    assert job.status == "FAILED"
    assert "SOURCE_SIZE_CHANGED_SINCE_DISCOVERY" in (job.error_message or "") or "SOURCE_CHANGED_SINCE_DISCOVERY" in (job.error_message or "")


def test_staging_rejects_symlink_replacement(tmp_path):
    """File replaced by symlink between discovery and staging must be rejected."""
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir(parents=True)

    mp4 = inbox / "pas_test_03.mp4"
    _create_dummy_mp4(mp4, frames=5)
    (inbox / "pas_test_03.done").write_text("DONE")

    other_mp4 = outside / "other.mp4"
    _create_dummy_mp4(other_mp4, frames=5)

    worker = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "out",
        size_stability_interval_sec=0.0,
    )

    discovered = worker.discover_new_recordings()
    assert len(discovered) == 1

    # Replace original file with symlink
    mp4.unlink()
    mp4.symlink_to(other_mp4)

    job = worker.stage_job(discovered[0])
    assert job.status == "FAILED"


def test_discovery_hash_preserved_in_metadata(tmp_path):
    """Discovery hash and size must be preserved in immutable job metadata."""
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)

    mp4 = inbox / "pas_test_04.mp4"
    _create_dummy_mp4(mp4, frames=5)
    (inbox / "pas_test_04.done").write_text("DONE")
    source_sha = compute_file_sha256(mp4)
    source_size = mp4.stat().st_size

    worker = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "out",
        size_stability_interval_sec=0.0,
    )

    discovered = worker.discover_new_recordings()
    assert len(discovered) == 1

    job = worker.stage_job(discovered[0])
    assert job.status == "STAGED"
    assert job.discovered_sha256 == source_sha
    assert job.discovered_size_bytes == source_size
    assert job.mp4_sha256 == source_sha


def test_different_hash_for_same_passage_distinct_event(tmp_path):
    """Different hash for same passage must create distinct event without overwriting previous."""
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    storage = tmp_path / "storage"
    out_dir = tmp_path / "out"

    mp4_v1 = inbox / "pas_test_05.mp4"
    _create_dummy_mp4(mp4_v1, frames=5)
    (inbox / "pas_test_05.done").write_text("DONE")

    worker = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=storage,
        output_dir=out_dir,
        size_stability_interval_sec=0.0,
    )

    # First version discovered, staged, processed
    disc1 = worker.discover_new_recordings()
    job1 = worker.stage_job(disc1[0])
    ok = worker.process_job(job1)
    assert ok is True
    assert job1.status == "COMPLETED"

    committed_decision_path = Path(job1.output_json_path)
    assert committed_decision_path.is_file()
    original_decision_bytes = committed_decision_path.read_bytes()

    # Now a second recording arrives for same passage name but different hash
    _create_dummy_mp4(mp4_v1, frames=12)
    (inbox / "pas_test_05.done").write_text("DONE")

    disc2 = worker.discover_new_recordings()
    assert len(disc2) == 1

    job2 = worker.stage_job(disc2[0])
    assert job2.job_id != job1.job_id
    assert job2.output_json_path != job1.output_json_path

    # First committed output must remain unchanged
    assert committed_decision_path.read_bytes() == original_decision_bytes


def _staged_worker(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    mp4 = inbox / "retry.mp4"
    _create_dummy_mp4(mp4)
    (inbox / "retry.done").write_text("DONE")
    worker = ShadowWorker(inbox, tmp_path / "storage", tmp_path / "out", size_stability_interval_sec=0)
    job = worker.stage_job(worker.discover_new_recordings()[0])
    return worker, job, mp4


def test_existing_staged_file_tampered_is_rejected_before_inference(tmp_path):
    worker, job, _ = _staged_worker(tmp_path)
    Path(job.staged_mp4_path).write_bytes(b"tampered")
    with patch("shadow_identity.shadow_ocr.worker.execute_pipeline", return_value={"resolver_decision": {}}) as execute:
        assert worker.process_job(job) is False
    execute.assert_not_called()
    assert job.error_message in {"SOURCE_CHANGED_SINCE_DISCOVERY", "SOURCE_SIZE_CHANGED_SINCE_DISCOVERY"}


def test_missing_staged_file_does_not_restage_changed_original(tmp_path):
    worker, job, source = _staged_worker(tmp_path)
    Path(job.staged_mp4_path).unlink()
    source.write_bytes(b"changed after discovery")
    with patch("shadow_identity.shadow_ocr.worker.execute_pipeline", return_value={"resolver_decision": {}}) as execute:
        assert worker.process_job(job) is False
    execute.assert_not_called()
    assert job.error_message in {"SOURCE_CHANGED_SINCE_DISCOVERY", "SOURCE_SIZE_CHANGED_SINCE_DISCOVERY"}


def test_missing_staged_file_safely_restages_unchanged_original(tmp_path):
    worker, job, _ = _staged_worker(tmp_path)
    Path(job.staged_mp4_path).unlink()
    with patch("shadow_identity.shadow_ocr.worker.execute_pipeline", return_value={"resolver_decision": {}}) as execute:
        assert worker.process_job(job) is True
    execute.assert_called_once()


def test_restart_retry_keeps_discovery_identity_and_rejects_tampered_stage(tmp_path):
    worker, job, _ = _staged_worker(tmp_path)
    job.status = "PROCESSING"
    worker._save_state()
    restarted = ShadowWorker(worker.inbox_dir, worker.storage_dir, worker.output_dir, size_stability_interval_sec=0)
    recovered = restarted.jobs[job.job_id]
    Path(recovered.staged_mp4_path).write_bytes(b"changed")
    with patch("shadow_identity.shadow_ocr.worker.execute_pipeline", return_value={"resolver_decision": {}}) as execute:
        assert restarted.process_job(recovered) is False
    execute.assert_not_called()
    assert recovered.discovered_sha256 == job.discovered_sha256


def test_symlinked_staged_file_is_rejected_before_inference(tmp_path):
    worker, job, _ = _staged_worker(tmp_path)
    staged = Path(job.staged_mp4_path)
    staged.unlink()
    staged.symlink_to(Path(job.original_mp4_path))
    with patch("shadow_identity.shadow_ocr.worker.execute_pipeline", return_value={"resolver_decision": {}}) as execute:
        assert worker.process_job(job) is False
    execute.assert_not_called()


def test_identity_mismatch_preserves_existing_committed_evidence(tmp_path):
    worker, job, _ = _staged_worker(tmp_path)
    decision, raw = Path(job.output_json_path), Path(job.raw_jsonl_path)
    decision.write_bytes(b"committed decision")
    raw.write_bytes(b"committed raw evidence")
    Path(job.staged_mp4_path).write_bytes(b"tampered")
    with patch("shadow_identity.shadow_ocr.worker.execute_pipeline", return_value={"resolver_decision": {}}) as execute:
        assert worker.process_job(job) is False
    execute.assert_not_called()
    assert decision.read_bytes() == b"committed decision"
    assert raw.read_bytes() == b"committed raw evidence"


def test_permanent_inference_failure_enters_retention_cleanup(tmp_path):
    worker, job, source = _staged_worker(tmp_path)
    source_hash = compute_file_sha256(source)
    job.max_attempts = 1
    with patch("shadow_identity.shadow_ocr.worker.execute_pipeline", side_effect=RuntimeError("inference failed")):
        assert worker.process_job(job) is False
    lifecycle_record = worker.lifecycle_mgr._jobs[job.lifecycle_job_id]
    assert job.status == lifecycle_record.status == "FAILED"
    assert worker.lifecycle_mgr.run_cleanup_due(now_epoch=time.time() + 86401) == [job.lifecycle_job_id]
    assert not Path(job.staged_mp4_path).exists()
    assert source.is_file() and compute_file_sha256(source) == source_hash


def test_worker_persisted_unknown_flows_into_daily_report(tmp_path):
    worker, job, _ = _staged_worker(tmp_path)

    def persist_unknown(**kwargs):
        source_sha = compute_file_sha256(Path(kwargs["mp4_path"]))
        decision = {
            "site_id": kwargs["site_id"], "passage_id": kwargs["passage_id"],
            "input_video_sha256": source_sha, "identity_status": "UNKNOWN",
            "active_service_number": None, "active_direction_verified": False,
            "reason_codes": ["NO_VERIFIED_NUMBER"], "contradictions": [],
            "resolver_decision": {"shadow_status": "UNKNOWN"},
            "performance_latency": {"total_wall_clock_s": 0.1, "ocr_calls_executed": 1},
            "passage_metadata": {"passage_start": "2026-10-26T10:00:00+05:30"},
        }
        Path(kwargs["output_path"]).write_text(json.dumps(decision))
        Path(kwargs["raw_observations_path"]).write_text("")
        return {"resolver_decision": {"shadow_status": "UNKNOWN", "shadow_final_number": None}}

    with patch("shadow_identity.shadow_ocr.worker.execute_pipeline", side_effect=persist_unknown):
        assert worker.process_job(job) is True

    md_path, json_path = generate_worker_daily_review_report(
        report_date="2026-10-26", worker_state_path=worker.state_file,
        ground_truth_dir=tmp_path / "ground_truth", output_dir=tmp_path / "reports",
        site_id="ndls_pf12_13",
    )
    report = json.loads(json_path.read_text())
    assert md_path.is_file()
    assert report["passages_count"] == 1
    assert report["metrics"]["unknown_count"] == 1
    assert report["passages"][0]["input_video_sha256"] == job.discovered_sha256
