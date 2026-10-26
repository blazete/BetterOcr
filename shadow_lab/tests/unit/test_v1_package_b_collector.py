"""
Regression tests for BetterOcr / ShadowOCR V1 Package B:
Continuous Passage Collector.

Requirements:
B1 — Passage Input Contract (PassageContext dataclass).
B2 — Controlled polling discovery, job_id incorporating full SHA-256, no writes to ARDS.
B3 — Strict State Machine:
     DISCOVERED -> VALIDATED -> STAGED -> PROCESSING -> EVIDENCE_PERSISTED -> DECISION_PERSISTED -> COMPLETED
     Transient: RETRY_WAIT -> STAGED
     Permanent: FAILED
     Crash recovery rules.
B4 — Staging Lifecycle:
     Verify source & staged hash & byte count.
     Cleanup only after completion & retention time.
"""

from pathlib import Path
import json
import pytest
import cv2
import numpy as np

from shadow_identity.models import PassageContext
from shadow_identity.shadow_ocr.worker import (
    ShadowWorker,
    WorkerJob,
    compute_file_sha256,
    transition_job_status,
    LEGAL_TRANSITIONS,
)


def _create_dummy_mp4(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 64))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(5):
        out.write(frame)
    out.release()


def test_b1_passage_context_contract():
    """B1: PassageContext contract preserves None for missing metadata and is frozen."""
    ctx = PassageContext(
        site_id="ndls_pf12_13",
        installation_id="inst_01",
        camera_id="side_a",
        passage_id="pas_001",
        source_mp4_path="/path/to/source.mp4",
        source_sha256="abc123sha256",
        passage_start_utc=None,
        passage_end_utc=None,
        timestamp_source="NONE",
        observed_platform=None,
        possible_platforms=("12", "13"),
        track_id=None,
        travel_direction=None,
        direction_source=None,
        coach_intervals_verified=False,
        finalization_source="DONE_MARKER",
    )
    assert ctx.passage_start_utc is None
    assert ctx.track_id is None
    assert ctx.travel_direction is None
    # Frozen contract: immutable
    with pytest.raises(AttributeError):
        ctx.site_id = "other"


def test_b3_state_machine_legal_transitions():
    """B3: State machine enforces exact progression and rejects invalid transitions."""
    job = WorkerJob(
        job_id="test_job",
        passage_id="pas_test",
        site_id="ndls_pf12_13",
        camera_id="side_a",
        original_mp4_path="/tmp/test.mp4",
        mp4_sha256="hash123",
        file_size_bytes=1000,
        status="DISCOVERED",
        discovered_at_epoch=100.0,
    )
    # Valid progression
    transition_job_status(job, "VALIDATED")
    assert job.status == "VALIDATED"
    transition_job_status(job, "STAGED")
    assert job.status == "STAGED"
    transition_job_status(job, "PROCESSING")
    assert job.status == "PROCESSING"
    transition_job_status(job, "EVIDENCE_PERSISTED")
    assert job.status == "EVIDENCE_PERSISTED"
    transition_job_status(job, "DECISION_PERSISTED")
    assert job.status == "DECISION_PERSISTED"
    transition_job_status(job, "COMPLETED")
    assert job.status == "COMPLETED"

    # Illegal transition: cannot jump from COMPLETED to PROCESSING
    with pytest.raises(ValueError):
        transition_job_status(job, "PROCESSING")


def test_b3_state_machine_retry_and_failed_transitions():
    """B3: Retry and failure state transitions."""
    job = WorkerJob(
        job_id="test_job_retry",
        passage_id="pas_test",
        site_id="ndls_pf12_13",
        camera_id="side_a",
        original_mp4_path="/tmp/test.mp4",
        mp4_sha256="hash123",
        file_size_bytes=1000,
        status="PROCESSING",
        discovered_at_epoch=100.0,
    )
    # Transient error -> RETRY_WAIT -> STAGED
    transition_job_status(job, "RETRY_WAIT")
    assert job.status == "RETRY_WAIT"
    transition_job_status(job, "STAGED")
    assert job.status == "STAGED"

    # From STAGED back to PROCESSING
    transition_job_status(job, "PROCESSING")
    # Permanent error -> FAILED
    transition_job_status(job, "FAILED")
    assert job.status == "FAILED"


def test_b3_crash_recovery_matching_artifacts_recovers_completed(tmp_path):
    """B3: Crash recovery with matching decision and raw evidence recovers COMPLETED."""
    storage = tmp_path / "storage"
    out_dir = tmp_path / "output"
    inbox = tmp_path / "inbox"
    out_dir.mkdir(parents=True, exist_ok=True)
    inbox.mkdir(parents=True, exist_ok=True)

    mp4 = inbox / "test.mp4"
    _create_dummy_mp4(mp4)
    sha = compute_file_sha256(mp4)

    job_id = f"ndls_pf12_13_side_a_pas_test_{sha}"
    decision_file = out_dir / f"{job_id}_decision.json"
    raw_file = out_dir / f"{job_id}_raw.jsonl"

    # Create matching persisted artifacts
    decision_file.write_text(json.dumps({
        "input_video_sha256": sha,
        "resolver_decision": {"shadow_status": "PROVISIONAL", "shadow_final_number": "12445"},
    }))
    raw_file.write_text(json.dumps({"frame_id": 10, "raw_text": "12445"}) + "\n")

    # Worker state file saved with job in PROCESSING
    storage.mkdir(parents=True, exist_ok=True)
    state_file = storage / "worker_jobs.json"
    state_file.write_text(json.dumps({
        job_id: {
            "job_id": job_id,
            "passage_id": "pas_test",
            "site_id": "ndls_pf12_13",
            "camera_id": "side_a",
            "original_mp4_path": str(mp4),
            "mp4_sha256": sha,
            "file_size_bytes": mp4.stat().st_size,
            "status": "PROCESSING",
            "discovered_at_epoch": 100.0,
            "output_json_path": str(decision_file),
            "raw_jsonl_path": str(raw_file),
        }
    }))

    worker = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=storage,
        output_dir=out_dir,
        size_stability_interval_sec=0.0,
    )
    recovered_job = worker.jobs[job_id]
    assert recovered_job.status == "COMPLETED"


def test_b3_crash_recovery_raw_only_does_not_destroy_evidence(tmp_path):
    """B3: Raw evidence exists but decision missing -> recover without destroying durable raw evidence."""
    storage = tmp_path / "storage"
    out_dir = tmp_path / "output"
    inbox = tmp_path / "inbox"
    out_dir.mkdir(parents=True, exist_ok=True)
    inbox.mkdir(parents=True, exist_ok=True)

    mp4 = inbox / "test.mp4"
    _create_dummy_mp4(mp4)
    sha = compute_file_sha256(mp4)

    job_id = f"ndls_pf12_13_side_a_pas_test_{sha}"
    raw_file = out_dir / f"{job_id}_raw.jsonl"
    raw_file.write_text(json.dumps({"frame_id": 10, "raw_text": "12445"}) + "\n")

    storage.mkdir(parents=True, exist_ok=True)
    state_file = storage / "worker_jobs.json"
    state_file.write_text(json.dumps({
        job_id: {
            "job_id": job_id,
            "passage_id": "pas_test",
            "site_id": "ndls_pf12_13",
            "camera_id": "side_a",
            "original_mp4_path": str(mp4),
            "mp4_sha256": sha,
            "file_size_bytes": mp4.stat().st_size,
            "status": "PROCESSING",
            "discovered_at_epoch": 100.0,
            "output_json_path": str(out_dir / f"{job_id}_decision.json"),
            "raw_jsonl_path": str(raw_file),
        }
    }))

    worker = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=storage,
        output_dir=out_dir,
        size_stability_interval_sec=0.0,
    )
    # Raw file must NOT be deleted
    assert raw_file.is_file()
    assert worker.jobs[job_id].status in ("EVIDENCE_PERSISTED", "STAGED")


def test_b4_staging_lifecycle_verification(tmp_path):
    """B4: Staging verifies hash, byte count, and exclusive creation."""
    inbox = tmp_path / "inbox"
    storage = tmp_path / "storage"
    out_dir = tmp_path / "output"
    inbox.mkdir(parents=True, exist_ok=True)

    mp4 = inbox / "sample.mp4"
    _create_dummy_mp4(mp4)
    (inbox / "sample.done").write_text("DONE")

    worker = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=storage,
        output_dir=out_dir,
        size_stability_interval_sec=0.0,
    )
    job = worker.stage_job(mp4)
    assert job.status == "STAGED"
    assert Path(job.staged_mp4_path).is_file()
    # Verify staged hash equals source hash
    staged_hash = compute_file_sha256(Path(job.staged_mp4_path))
    assert staged_hash == job.mp4_sha256
    assert Path(job.staged_mp4_path).stat().st_size == job.file_size_bytes
    # Original MP4 must remain unchanged
    assert mp4.is_file()
