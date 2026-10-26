"""
Regression tests for Package P0.4: Fix worker idempotency, safety, and retry.
"""
import json
import time
import uuid
from pathlib import Path
import pytest

from shadow_identity.shadow_ocr.worker import ShadowWorker, WorkerJob, compute_file_sha256
from shadow_identity.shadow_ocr.temp_mp4_lifecycle import TempMp4LifecycleManager


def test_require_finalization_signal_default_true(tmp_path):
    # Verify default is require_finalization_signal=True
    worker = ShadowWorker(
        inbox_dir=tmp_path / "inbox",
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "output",
    )
    assert worker.require_finalization_signal is True, (
        f"Worker default require_finalization_signal must be True for safety, got {worker.require_finalization_signal}"
    )


def test_job_id_uses_full_sha256_and_job_id_named_outputs(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    dummy_mp4 = inbox / "test_video.mp4"
    import cv2
    import numpy as np
    vw = cv2.VideoWriter(str(dummy_mp4), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 64))
    for _ in range(5):
        vw.write(np.zeros((64, 64, 3), dtype=np.uint8))
    vw.release()
    # Write authoritative finalization signal
    dummy_mp4.with_suffix(".done").write_text("FINALIZED")

    worker = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "output",
        require_finalization_signal=True,
    )

    discovered = worker.discover_new_recordings()
    assert len(discovered) == 1
    job = worker.stage_job(discovered[0])

    full_sha = compute_file_sha256(dummy_mp4)
    # job_id must contain full 64-char sha, not truncated to 16
    assert full_sha in job.job_id, f"Expected full sha {full_sha} in job_id, got {job.job_id}"
    # Output file paths must be named after job_id, not passage_id alone
    assert job.job_id in Path(job.output_json_path).name
    assert job.job_id in Path(job.raw_jsonl_path).name


def test_retry_on_failure_and_exhaustion(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    dummy_mp4 = inbox / "test_retry.mp4"
    import cv2
    import numpy as np
    vw = cv2.VideoWriter(str(dummy_mp4), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 64))
    for _ in range(5):
        vw.write(np.zeros((64, 64, 3), dtype=np.uint8))
    vw.release()
    dummy_mp4.with_suffix(".done").write_text("FINALIZED")

    worker = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "output",
        require_finalization_signal=True,
    )
    job = worker.stage_job(dummy_mp4)
    job.max_attempts = 2

    # Simulate execution failure on attempt 1
    # Under P0.4: first failure should set RETRY_WAIT with exponential backoff
    def failing_execute(*args, **kwargs):
        raise RuntimeError("Simulated processing error")

    import shadow_identity.shadow_ocr.worker as w_mod
    orig_exec = w_mod.execute_pipeline
    w_mod.execute_pipeline = failing_execute
    try:
        success = worker.process_job(job)
        assert success is False
        assert job.attempts == 1
        assert job.status == "RETRY_WAIT", f"Expected RETRY_WAIT on attempt 1, got {job.status}"
        assert job.retry_after_epoch is not None and job.retry_after_epoch > time.time()

        # Immediate cycle should NOT pick up job while retry_after_epoch is in future
        ready_before = [j for j in worker.jobs.values() if j.status in ("STAGED", "PROCESSING") or (j.status == "RETRY_WAIT" and time.time() >= (j.retry_after_epoch or 0))]
        assert job not in ready_before, "Job in backoff must not be selected prematurely"

        # Advance time to expire backoff
        job.retry_after_epoch = time.time() - 1
        # Attempt 2 failure: attempts reaches max_attempts (2) -> terminal FAILED
        success2 = worker.process_job(job)
        assert success2 is False
        assert job.attempts == 2
        assert job.status == "FAILED", f"Expected FAILED after reaching max_attempts, got {job.status}"
    finally:
        w_mod.execute_pipeline = orig_exec


def test_atomic_claim_prevents_duplicate_processing(tmp_path):
    worker = ShadowWorker(
        inbox_dir=tmp_path / "inbox",
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "output",
        require_finalization_signal=False,
    )
    job = WorkerJob(
        job_id="claim_test_job",
        passage_id="p1",
        site_id="ndls",
        camera_id="cam1",
        original_mp4_path=str(tmp_path / "dummy.mp4"),
        mp4_sha256="dummy_sha",
        file_size_bytes=100,
        status="STAGED",
        discovered_at_epoch=time.time(),
    )
    worker.jobs[job.job_id] = job

    # First claim succeeds
    claimed = worker.claim_job_for_processing(job.job_id)
    assert claimed is not None
    assert claimed.status == "PROCESSING"
    assert claimed.attempts == 1

    # Concurrent second claim must be rejected (return None)
    second_claim = worker.claim_job_for_processing(job.job_id)
    assert second_claim is None, "Concurrent claim on PROCESSING job must return None"
