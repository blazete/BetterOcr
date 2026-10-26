"""
Unit and Integration Tests for Autonomous Shadow Worker (Part 2).
Covers:
1. Discovery of finalized MP4s.
2. Rejection of unfinalized / zero-byte / growing files.
3. Idempotent staging and lifecycle tracking.
4. Raw JSONL and final decision atomic persistence before lifecycle completion.
5. Simulated crash recovery and restart idempotency.
6. Original MP4 immutability check.
"""

from datetime import datetime
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import pytest

from shadow_identity.shadow_ocr.worker import ShadowWorker, WorkerJob


@pytest.fixture
def dummy_mp4(tmp_path: Path) -> Path:
    """Creates a tiny valid MP4 file for fast worker pipeline testing."""
    p = tmp_path / "passage_test_dummy.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(p), fourcc, 10.0, (160, 120))
    for i in range(10):
        frame = np.full((120, 160, 3), 128, dtype=np.uint8)
        out.write(frame)
    out.release()
    p.with_suffix(".done").write_text("DONE")
    return p


def test_worker_discovery_and_finalization_validation(tmp_path: Path, dummy_mp4: Path):
    """Worker discovers valid MP4s and rejects zero-byte or non-video files."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    storage = tmp_path / "storage"
    output = tmp_path / "output"

    # Copy dummy MP4 to inbox with .done signal
    shutil.copy2(dummy_mp4, inbox / "video_ok.mp4")
    (inbox / "video_ok.done").write_text("DONE")

    # Create empty / zero-byte file
    (inbox / "incomplete.mp4").touch()

    # Create invalid text file with .mp4 extension
    (inbox / "corrupt.mp4").write_text("not a video")

    worker = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=storage,
        output_dir=output,
        size_stability_interval_sec=0.1,
    )

    discovered = worker.discover_new_recordings()
    names = [p.name for p in discovered]
    assert "video_ok.mp4" in names
    assert "incomplete.mp4" not in names
    assert "corrupt.mp4" not in names


def test_worker_idempotent_staging_and_lifecycle(tmp_path: Path, dummy_mp4: Path):
    """Worker creates idempotent jobs and stages copies without touching original."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    storage = tmp_path / "storage"
    output = tmp_path / "output"

    orig_mp4 = inbox / "orig_passage.mp4"
    shutil.copy2(dummy_mp4, orig_mp4)
    (inbox / "orig_passage.done").write_text("DONE")
    orig_stat_before = orig_mp4.stat()

    worker = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=storage,
        output_dir=output,
        size_stability_interval_sec=0.0,
    )

    # First cycle stages
    stats1 = worker.run_cycle()
    assert stats1["discovered"] == 1
    assert stats1["processed"] == 1

    # Verify original file untouched
    orig_stat_after = orig_mp4.stat()
    assert orig_stat_before.st_size == orig_stat_after.st_size
    assert orig_stat_before.st_mtime == orig_stat_after.st_mtime

    # Output files exist and are durable named by job_id
    job = list(worker.jobs.values())[0]
    decision_file = Path(job.output_json_path)
    raw_file = Path(job.raw_jsonl_path)
    assert decision_file.is_file()
    assert raw_file.is_file()

    # Second cycle on same inbox should not re-discover or re-process
    stats2 = worker.run_cycle()
    assert stats2["discovered"] == 0
    assert stats2["processed"] == 0


def test_worker_crash_recovery_and_restart(tmp_path: Path, dummy_mp4: Path):
    """Simulates a worker crash during PROCESSING state and verifies restart recovery."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    storage = tmp_path / "storage"
    output = tmp_path / "output"

    shutil.copy2(dummy_mp4, inbox / "crash_test.mp4")
    (inbox / "crash_test.done").write_text("DONE")

    worker1 = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=storage,
        output_dir=output,
        size_stability_interval_sec=0.0,
    )

    discovered = worker1.discover_new_recordings()
    assert len(discovered) == 1
    job = worker1.stage_job(discovered[0])
    assert job.status == "STAGED"

    # Simulate crash mid-processing: set status to PROCESSING in persisted state
    job.status = "PROCESSING"
    worker1._save_state()

    # Create new worker instance simulating process restart
    worker2 = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=storage,
        output_dir=output,
        size_stability_interval_sec=0.0,
    )

    # Crash recovery should have reset PROCESSING back to STAGED
    assert worker2.jobs[job.job_id].status == "STAGED"

    # Worker finishes the job cleanly
    stats = worker2.run_cycle()
    assert stats["processed"] == 1
    assert worker2.jobs[job.job_id].status == "COMPLETED"


def test_collision_resistant_keys_and_duplicate_filenames(tmp_path: Path, dummy_mp4: Path):
    """
    Test duplicate filenames and same-size different recordings:
    - Same filename in different directories or overwrites must key on SHA-256.
    - Two different files having the exact same size must NOT collide.
    """
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    storage = tmp_path / "storage"
    output = tmp_path / "output"

    # File 1: dummy_mp4
    f1 = inbox / "recording_a.mp4"
    shutil.copy2(dummy_mp4, f1)
    (inbox / "recording_a.done").write_text("DONE")

    # File 2: Same size as dummy_mp4, but different content (pad byte at end)
    f2 = inbox / "recording_b.mp4"
    data = dummy_mp4.read_bytes()
    # Flip last byte to keep exact same size but different SHA-256
    mutated = bytearray(data)
    mutated[-1] = (mutated[-1] + 1) % 256
    f2.write_bytes(bytes(mutated))
    (inbox / "recording_b.done").write_text("DONE")

    assert f1.stat().st_size == f2.stat().st_size
    assert f1.stat().st_size > 0

    worker = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=storage,
        output_dir=output,
        size_stability_interval_sec=0.0,
    )

    discovered = worker.discover_new_recordings()
    assert len(discovered) == 2

    job1 = worker.stage_job(f1)
    job2 = worker.stage_job(f2)

    # Collision check: Jobs must have distinct collision-resistant keys despite same size
    assert job1.job_id != job2.job_id
    assert job1.mp4_sha256 != job2.mp4_sha256
    assert job1.job_id.endswith(job1.mp4_sha256)
    assert job2.job_id.endswith(job2.mp4_sha256)


def test_authoritative_finalization_signal_required(tmp_path: Path, dummy_mp4: Path):
    """When require_finalization_signal is True, files without .done or finalized=true are rejected."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    storage = tmp_path / "storage"
    output = tmp_path / "output"

    # File without signal
    f_nosignal = inbox / "video_nosignal.mp4"
    shutil.copy2(dummy_mp4, f_nosignal)

    # File with .done signal
    f_done = inbox / "video_with_done.mp4"
    shutil.copy2(dummy_mp4, f_done)
    (inbox / "video_with_done.done").write_text("DONE")

    # File with sidecar finalized=true
    f_meta = inbox / "video_with_meta.mp4"
    shutil.copy2(dummy_mp4, f_meta)
    (inbox / "video_with_meta.json").write_text(json.dumps({"finalized": True}))

    worker = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=storage,
        output_dir=output,
        size_stability_interval_sec=0.0,
        require_finalization_signal=True,
    )

    discovered = worker.discover_new_recordings()
    names = [p.name for p in discovered]
    assert "video_nosignal.mp4" not in names
    assert "video_with_done.mp4" in names
    assert "video_with_meta.mp4" in names


def test_corrupted_partial_output_cleanup_on_restart(tmp_path: Path, dummy_mp4: Path):
    """Partially written / corrupt output files left by a crash are purged on restart."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    storage = tmp_path / "storage"
    output = tmp_path / "output"

    f = inbox / "partial_test.mp4"
    shutil.copy2(dummy_mp4, f)
    f.with_suffix(".done").write_text("DONE")

    worker1 = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=storage,
        output_dir=output,
        size_stability_interval_sec=0.0,
    )
    job = worker1.stage_job(f)
    job.status = "PROCESSING"

    # Write truncated corrupt output JSON
    partial_json = Path(job.output_json_path)
    partial_json.parent.mkdir(parents=True, exist_ok=True)
    partial_json.write_text('{"partial": "corrupt')  # invalid json
    worker1._save_state()

    assert partial_json.is_file()

    # Restart worker
    worker2 = ShadowWorker(
        inbox_dir=inbox,
        storage_dir=storage,
        output_dir=output,
        size_stability_interval_sec=0.0,
    )

    # Recovery must reset status and purge truncated output
    assert worker2.jobs[job.job_id].status == "STAGED"
    assert not partial_json.is_file()

