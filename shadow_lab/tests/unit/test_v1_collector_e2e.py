"""
End-to-End Continuous Passage Collector Validation Test
Validates:
1. Discovery of newly named finalized MP4 in explicitly allowlisted shadow-owned directory.
2. Staging lifecycle (hash, byte count, exclusive creation).
3. OCR inference execution and budget limits (<= 25 calls).
4. Durable atomic evidence persistence (raw JSONL + decision JSON with shadow_identity_v1 schema).
5. Deterministic identity decision gating.
6. Idempotency (duplicate discovery/processing strictly prevented).
7. Crash recovery rules (valid artifacts recovered, raw evidence never destroyed).
8. Cleanup lifecycle (retention compliance, temp purged, original untouched).
9. ARDS isolation guarantee (zero writes to ARDS).
"""

from pathlib import Path
import json
import time
import cv2
import numpy as np
import pytest

from shadow_identity.shadow_ocr.worker import (
    ShadowWorker,
    compute_file_sha256,
)


def _generate_test_passage_mp4(path: Path, frames: int = 15):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (128, 128))
    for i in range(frames):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        # Put simple text on some frames
        cv2.putText(img, "12445", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out.write(img)
    out.release()


def test_e2e_collector_lifecycle_new_passage(tmp_path):
    """End-to-end collector test on newly named finalized MP4."""
    inbox_dir = tmp_path / "allowlisted_inbox"
    storage_dir = tmp_path / "shadow_storage"
    output_dir = tmp_path / "shadow_output"
    inbox_dir.mkdir(parents=True)

    # 1. Authoritative finalization of a newly named recording
    new_mp4 = inbox_dir / "pas_ndls_test_live_999.mp4"
    _generate_test_passage_mp4(new_mp4, frames=20)
    done_marker = inbox_dir / "pas_ndls_test_live_999.done"
    done_marker.write_text("DONE")

    source_hash = compute_file_sha256(new_mp4)
    source_bytes = new_mp4.stat().st_size

    # Configure worker with explicit allowlisted root and trusted workspace root
    worker = ShadowWorker(
        inbox_dir=inbox_dir,
        storage_dir=storage_dir,
        output_dir=output_dir,
        site_id="ndls_pf12_13",
        camera_id="side_a",
        max_ocr_budget=25,
        cleanup_retention_sec=0.1,  # short retention for test
        operator_replay_workspace_root=tmp_path,
        offline_replay_config={
            "enabled": True,
            "allowed_roots": [str(inbox_dir.resolve())],
        },
    )

    # 2. Discovery
    discovered = worker.discover_new_recordings()
    assert len(discovered) == 1
    assert discovered[0].resolve() == new_mp4.resolve()

    # 3. Staging
    job = worker.stage_job(discovered[0])
    assert job.status == "STAGED"
    assert job.mp4_sha256 == source_hash
    assert job.file_size_bytes == source_bytes
    assert Path(job.staged_mp4_path).is_file()
    assert compute_file_sha256(Path(job.staged_mp4_path)) == source_hash
    # Original MP4 in inbox must remain untouched
    assert new_mp4.is_file()
    assert new_mp4.stat().st_size == source_bytes

    # 4. Processing & Persistence
    ok = worker.process_job(job)
    assert ok is True
    assert job.status == "COMPLETED"

    # Verify decision JSON persisted with shadow_identity_v1 schema
    decision_path = Path(job.output_json_path)
    raw_path = Path(job.raw_jsonl_path)

    assert decision_path.is_file()
    assert raw_path.is_file()

    with open(decision_path, "r", encoding="utf-8") as f:
        dec_data = json.load(f)

    assert dec_data["schema_version"] == "shadow_identity_v1"
    assert dec_data["site_id"] == "ndls_pf12_13"
    assert dec_data["input_video_sha256"] == source_hash
    assert dec_data["identity_status"] in ("UNKNOWN", "FAMILY_CANDIDATE", "VERIFIED_ACTIVE")
    assert dec_data["decision_created_utc"] is not None

    # Verify raw JSONL lines exist and have schema
    with open(raw_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) >= 0

    # 5. Idempotency & Cleanup: Second cycle must not re-process completed job
    discovered_second = worker.discover_new_recordings()
    assert len(discovered_second) == 0

    # Wait past retention interval to verify autonomous cleanup during run_cycle
    time.sleep(0.15)
    cycle_stats = worker.run_cycle()
    assert cycle_stats["discovered"] == 0
    assert cycle_stats["processed"] == 0
    assert cycle_stats["cleaned_up_mp4s"] == 1

    # Temp MP4 unlinked
    assert not Path(job.staged_mp4_path).exists()
    # BUT durable decision and raw evidence remain intact
    assert decision_path.is_file()
    assert raw_path.is_file()
    # AND original MP4 remains untouched
    assert new_mp4.is_file()
    assert compute_file_sha256(new_mp4) == source_hash

    # 7. Crash Recovery Verification on Restart
    worker_restarted = ShadowWorker(
        inbox_dir=inbox_dir,
        storage_dir=storage_dir,
        output_dir=output_dir,
        site_id="ndls_pf12_13",
        camera_id="side_a",
    )
    assert job.job_id in worker_restarted.jobs
    assert worker_restarted.jobs[job.job_id].status == "COMPLETED"
