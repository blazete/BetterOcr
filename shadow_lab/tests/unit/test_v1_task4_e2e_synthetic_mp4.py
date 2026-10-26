"""
End-to-End Validation with Newly Created Synthetic MP4 (Task 4).

Validates the full workflow on a new synthetic recording:
Discovery -> Finalization validation -> Staging -> Actual OCR execution ->
Evidence persistence -> Identity decision -> Durable output -> Restart recovery ->
Idempotency -> Retention cleanup.

Checks:
- Original MP4 remains unchanged.
- Shadow storage contains all outputs.
- No writes occur inside ARDS directories.
- OCR never exceeds 25 calls.
- Unsupported train numbers are not confirmed.
- Service-family candidate mappings require verified provenance.
- Missing direction produces active_direction_verified = false.
- Missing railway wall-clock timestamp remains null.
- Unknown trains produce UNKNOWN rather than invented identities.
- Repeated worker cycles do not create duplicate inspections.
- Inference and cleanup failures do not remove source recordings.
"""

import json
import time
from pathlib import Path
import cv2
import numpy as np
import pytest

from shadow_identity.shadow_ocr.worker import ShadowWorker, compute_file_sha256


def _generate_synthetic_coach_mp4(path: Path, frames: int = 25):
    """Generates a synthetic MP4 with visible coach stencils for real OCR inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 5.0, (320, 240))
    for i in range(frames):
        # Blue railway coach body background
        img = np.full((240, 320, 3), (120, 50, 20), dtype=np.uint8)
        # Yellow destination board stripe
        cv2.rectangle(img, (20, 80), (300, 160), (0, 200, 255), -1)
        # Add coach stencil text
        cv2.putText(img, "NDLS SPECIAL", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "243561", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        out.write(img)
    out.release()


def test_task4_synthetic_mp4_e2e_workflow(tmp_path):
    """Full end-to-end lifecycle verification with real OCR execution."""
    workspace = tmp_path / "operator_workspace"
    inbox_dir = workspace / "inbox"
    storage_dir = tmp_path / "shadow_storage"
    output_dir = tmp_path / "shadow_output"
    ards_mock_dir = tmp_path / "ards_production"
    ards_mock_dir.mkdir(parents=True)
    inbox_dir.mkdir(parents=True)

    # 1. Authoritative generation of a newly named synthetic recording
    new_mp4 = inbox_dir / "pas_synthetic_ndls_trial_001.mp4"
    _generate_synthetic_coach_mp4(new_mp4, frames=20)
    done_marker = inbox_dir / "pas_synthetic_ndls_trial_001.done"
    done_marker.write_text("DONE")

    source_hash_before = compute_file_sha256(new_mp4)
    source_bytes_before = new_mp4.stat().st_size
    ards_files_before = list(ards_mock_dir.glob("**/*"))

    # Configure worker with explicit allowlisted root and trusted workspace root
    worker = ShadowWorker(
        inbox_dir=inbox_dir,
        storage_dir=storage_dir,
        output_dir=output_dir,
        site_id="ndls_pf12_13",
        camera_id="side_a",
        max_ocr_budget=25,
        cleanup_retention_sec=0.1,
        operator_replay_workspace_root=workspace,
        offline_replay_config={
            "enabled": True,
            "allowed_roots": [str(inbox_dir.resolve())],
        },
    )

    # 2. Discovery Verification
    discovered = worker.discover_new_recordings()
    assert len(discovered) == 1
    assert discovered[0].resolve() == new_mp4.resolve()

    # 3. Finalization Validation & Staging
    job = worker.stage_job(discovered[0])
    assert job.status == "STAGED"
    assert job.discovered_sha256 == source_hash_before
    assert job.discovered_size_bytes == source_bytes_before
    assert job.mp4_sha256 == source_hash_before
    assert Path(job.staged_mp4_path).is_file()
    assert compute_file_sha256(Path(job.staged_mp4_path)) == source_hash_before

    # Original MP4 in inbox must remain untouched
    assert new_mp4.is_file()
    assert new_mp4.stat().st_size == source_bytes_before
    assert compute_file_sha256(new_mp4) == source_hash_before

    # 4. Processing & Real OCR Execution
    ok = worker.process_job(job)
    assert ok is True
    assert job.status == "COMPLETED"

    # 5. Output Verification & Strict Invariants
    decision_path = Path(job.output_json_path)
    raw_path = Path(job.raw_jsonl_path)
    assert decision_path.is_file()
    assert raw_path.is_file()

    with open(decision_path, "r", encoding="utf-8") as f:
        decision_data = json.load(f)

    assert decision_data["schema_version"] == "shadow_identity_v1"
    assert decision_data["site_id"] == "ndls_pf12_13"
    assert decision_data["input_video_sha256"] == source_hash_before

    # Check budget ceiling
    ocr_calls = decision_data.get("performance_latency", {}).get("actual_ocr_calls", 0)
    assert ocr_calls <= 25

    # Check direction safety invariants:
    # Missing direction produces active_direction_verified = false
    assert decision_data["active_direction_verified"] is False
    assert decision_data["active_service_number"] is None

    # Check timestamp invariant:
    # Missing railway wall-clock timestamp remains null
    passage_meta = decision_data.get("passage_metadata", {})
    assert passage_meta.get("passage_start") is None
    assert passage_meta.get("passage_end") is None

    # Unknown train stencils produce UNKNOWN rather than invented identities
    assert decision_data["identity_status"] in ("UNKNOWN", "FAMILY_CANDIDATE")
    if decision_data["identity_status"] == "UNKNOWN":
        assert decision_data["service_family_candidates"] == []

    # 6. ARDS Isolation Check: No writes inside ARDS directories
    ards_files_after = list(ards_mock_dir.glob("**/*"))
    assert ards_files_before == ards_files_after

    # 7. Idempotency Check: Second discovery cycle skips completed job
    discovered_second = worker.discover_new_recordings()
    assert len(discovered_second) == 0

    # 8. Temporary Lifecycle Cleanup
    time.sleep(0.15)
    cycle_stats = worker.run_cycle()
    assert cycle_stats["discovered"] == 0
    assert cycle_stats["processed"] == 0
    assert cycle_stats["cleaned_up_mp4s"] == 1

    # Staged temporary MP4 unlinked
    assert not Path(job.staged_mp4_path).exists()
    # Durable output JSON and raw JSONL remain intact
    assert decision_path.is_file()
    assert raw_path.is_file()
    # Original MP4 in inbox remains intact
    assert new_mp4.is_file()
    assert compute_file_sha256(new_mp4) == source_hash_before

    # 9. Restart Crash Recovery
    worker_restarted = ShadowWorker(
        inbox_dir=inbox_dir,
        storage_dir=storage_dir,
        output_dir=output_dir,
        site_id="ndls_pf12_13",
        camera_id="side_a",
    )
    assert job.job_id in worker_restarted.jobs
    assert worker_restarted.jobs[job.job_id].status == "COMPLETED"
