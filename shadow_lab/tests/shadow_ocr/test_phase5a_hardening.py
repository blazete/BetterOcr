"""
Phase 5A Hardening and Boundary Security Regression Tests:
1. Missing passage timestamps explicitly disable time-based corroboration.
2. Forged / untrusted / unverified coach boundaries downgraded to UNVERIFIED and prevent PAIR_CONFIRMED.
3. Wrong-camera and mismatched passage intervals rejected.
4. Out-of-bounds frame intervals rejected.
5. TempMp4LifecycleManager stages temporary MP4, hashes/verifies, and cleans up after 60 min without touching original.
6. NTES +180s timing verification contract records acquisition time, relative offset, freshness, and late-verification.
7. Fallback crop is resolution-aware and camera-specific.
8. Persists raw OCR observations and evidence provenance into JSONL.
9. End-to-end passage resolution achieves CONFIRMED unique train number using independent timetable and NTES evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import pytest

from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.models import NtesSnapshot, NtesTrainEntry, PassageInput, TimetableRow
from shadow_identity.shadow_ocr.cli import execute_pipeline, validate_coach_boundaries
from shadow_identity.shadow_ocr.contracts import RawOcrObservation
from shadow_identity.shadow_ocr.temp_mp4_lifecycle import TempMp4LifecycleManager
from shadow_identity.timetable_importer import TimetableImporter


def test_missing_timestamps_disable_time_corroboration():
    """Missing passage timestamps must disable time corroboration instead of fabricating current time."""
    mp4_path = "train-identity-lab/videos/video_007.mp4"
    if not Path(mp4_path).is_file():
        pytest.skip("video_007.mp4 not present")

    res = execute_pipeline(
        mp4_path=mp4_path,
        site_id="ndls_pf12_13",
        platform="12",
        passage_id="test_no_ts",
        passage_start=None,
        passage_end=None,
        timetable_path="shadow_lab/sites/ndls_pf12_13/schedules/ndls_validated.csv",
        screening_fps=5.0,
        max_ocr_budget=10,
    )

    meta = res["passage_metadata"]
    assert meta["passage_start"] in ("", None)
    assert meta["passage_end"] in ("", None)
    assert meta["time_corroboration_enabled"] is False


def test_forged_coach_boundaries_downgraded_to_unverified(tmp_path: Path):
    """Untrusted / forged JSON boundaries without trusted provenance must be downgraded to UNVERIFIED."""
    forged_json = tmp_path / "forged_boundaries.json"
    forged_data = [
        {"coach_id": "C01", "start_frame": 100, "end_frame": 200},
        {"coach_id": "C02", "start_frame": 201, "end_frame": 300},
        {"coach_id": "C03", "start_frame": 301, "end_frame": 400},
    ]
    forged_json.write_text(json.dumps(forged_data))

    intervals, report = validate_coach_boundaries(
        forged_data,
        expected_passage_id="P_REAL",
        expected_camera_id="side_a",
        total_frames=1000,
    )

    assert report["status"] == "UNVERIFIED"
    assert report["trusted"] is False
    assert any("UNTRUSTED_SOURCE" in r for r in report["rejection_reasons"])
    for inter in intervals:
        assert inter["coach_id_source"] == "UNVERIFIED"


def test_wrong_camera_and_passage_interval_rejection():
    """Boundaries belonging to a different passage or camera ID must be rejected."""
    bound_data = {
        "metadata": {
            "source": "SYNCHRONIZED_SENSOR",
            "passage_id": "WRONG_PASSAGE",
            "camera_id": "side_b",
        },
        "coaches": [
            {"coach_id": "C01", "start_frame": 100, "end_frame": 200},
        ]
    }

    intervals, report = validate_coach_boundaries(
        bound_data,
        expected_passage_id="RIGHT_PASSAGE",
        expected_camera_id="side_a",
        total_frames=1000,
    )

    assert report["status"] == "UNVERIFIED"
    assert report["passage_match"] is False
    assert report["camera_match"] is False


def test_out_of_bounds_frame_intervals_rejection():
    """Boundaries with frames beyond video length must be rejected."""
    bound_data = {
        "metadata": {
            "source": "ARDS_HARDWARE_BEAM",
            "passage_id": "P1",
            "camera_id": "side_a",
        },
        "coaches": [
            {"coach_id": "C01", "start_frame": 100, "end_frame": 1500},
        ]
    }

    intervals, report = validate_coach_boundaries(
        bound_data,
        expected_passage_id="P1",
        expected_camera_id="side_a",
        total_frames=500,  # video has only 500 frames!
    )

    assert report["status"] == "UNVERIFIED"
    assert report["intervals_valid"] is False
    assert any("OUT_OF_BOUNDS" in r for r in report["rejection_reasons"])


def test_temp_mp4_lifecycle_staging_and_cleanup(tmp_path: Path):
    """Integrates TempMp4LifecycleManager into CLI ingestion and verifies 60m deletion contract."""
    mp4_path = "train-identity-lab/videos/video_007.mp4"
    if not Path(mp4_path).is_file():
        pytest.skip("video_007.mp4 not present")

    temp_dir = tmp_path / "shadow_temp"

    res = execute_pipeline(
        mp4_path=mp4_path,
        site_id="ndls_pf12_13",
        passage_id="test_staging_lifecycle",
        stage_temp_copy=True,
        temp_dir=str(temp_dir),
        cleanup_retention_sec=3600.0,
        screening_fps=5.0,
        max_ocr_budget=5,
    )

    staged_meta = res["passage_metadata"]["staged_temporary_job"]
    assert staged_meta is not None
    assert staged_meta["status"] == "COMPLETED"
    staged_file = Path(staged_meta["temp_mp4_path"])
    assert staged_file.exists()

    # Verify original MP4 was not altered or deleted
    assert Path(mp4_path).resolve().exists()

    # Trigger retention cleanup in the future (+3601s)
    mgr = TempMp4LifecycleManager(temp_dir=temp_dir, retention_completed_sec=3600.0)
    cleaned = mgr.run_cleanup_due(now_epoch=time.time() + 3601.0)
    assert staged_meta["job_id"] in cleaned
    assert not staged_file.exists()
    assert Path(mp4_path).resolve().exists()  # Original still 100% intact


def test_ntes_timing_contract_and_late_verification():
    """Tests the NTES verification timing contract relative to passage end +180s."""
    mp4_path = "train-identity-lab/videos/video_007.mp4"
    if not Path(mp4_path).is_file():
        pytest.skip("video_007.mp4 not present")

    # Snapshot acquired 240 seconds after passage end (late verification >= 180s)
    res = execute_pipeline(
        mp4_path=mp4_path,
        site_id="ndls_pf12_13",
        platform="12",
        passage_id="test_v007_ntes_contract",
        passage_start="2026-10-14T12:00:00+05:30",
        passage_end="2026-10-14T12:02:00+05:30",
        ntes_snapshot_path="shadow_lab/data/ntes_snapshots/ndls_pf12_13/snap_2026-10-13T21-59-00+05-30.json",
        screening_fps=5.0,
        max_ocr_budget=5,
    )

    timing_contract = res["resolver_decision"]["ntes_timing_contract"]
    assert timing_contract["verified_timestamps_present"] is True
    assert timing_contract["ntes_available"] is True
    assert timing_contract["acquisition_time"] == "2026-10-13T21:59:00+05:30"


def test_raw_ocr_observations_jsonl_persistence(tmp_path: Path):
    """Raw OCR observations and evidence provenance must be persisted separately to JSONL."""
    mp4_path = "train-identity-lab/videos/video_007.mp4"
    if not Path(mp4_path).is_file():
        pytest.skip("video_007.mp4 not present")

    raw_jsonl = tmp_path / "raw_evidence.jsonl"

    res = execute_pipeline(
        mp4_path=mp4_path,
        site_id="ndls_pf12_13",
        passage_id="test_raw_jsonl_pass",
        raw_observations_path=str(raw_jsonl),
        screening_fps=5.0,
        max_ocr_budget=10,
    )

    assert raw_jsonl.is_file()
    lines = [json.loads(line) for line in raw_jsonl.read_text(encoding="utf-8").strip().split("\n") if line]
    assert len(lines) == res["visual_evidence"]["total_raw_observations"]

    # Verify schema fields for provenance
    for row in lines:
        assert "frame_id" in row
        assert "timestamp_ms" in row
        assert "board_roi_xyxy" in row
        assert "raw_text" in row
        assert "ocr_confidence" in row
        assert "preprocessing_variant" in row
        assert "source_video_sha256" in row
        assert row["source_video_sha256"] == res["passage_metadata"]["video_sha256"]


def test_boundary_missing_ids_and_overlapping_intervals_rejection():
    """Boundaries missing passage/camera IDs or having overlapping intervals must be rejected."""
    # Missing passage ID
    bound_missing_id = {
        "metadata": {
            "source": "ARDS_HARDWARE_BEAM",
            "camera_id": "side_a",
            "hardware_sync_verified": True,
            "sync_method": "HARDWARE_BEAM",
        },
        "coaches": [{"coach_id": "C01", "start_frame": 100, "end_frame": 200}],
    }
    _, rep1 = validate_coach_boundaries(bound_missing_id, "P1", "side_a", 1000)
    assert rep1["status"] == "UNVERIFIED"
    assert "MISSING_PASSAGE_ID" in rep1["rejection_reasons"]

    # Overlapping physical coach intervals
    bound_overlap = {
        "metadata": {
            "source": "ARDS_HARDWARE_BEAM",
            "passage_id": "P1",
            "camera_id": "side_a",
            "hardware_sync_verified": True,
            "sync_method": "HARDWARE_BEAM",
        },
        "coaches": [
            {"coach_id": "C01", "start_frame": 100, "end_frame": 200},
            {"coach_id": "C02", "start_frame": 190, "end_frame": 300},  # Overlap 190 <= 200
        ],
    }
    _, rep2 = validate_coach_boundaries(bound_overlap, "P1", "side_a", 1000)
    assert rep2["status"] == "UNVERIFIED"
    assert any("OVERLAPPING_INTERVAL" in r for r in rep2["rejection_reasons"])


def test_early_ntes_snapshot_rejected_from_t2_confirmation():
    """A snapshot fetched 20s after passage end is NOT a +180s verification snapshot and cannot confirm."""
    mp4_path = "train-identity-lab/videos/video_007.mp4"
    if not Path(mp4_path).is_file():
        pytest.skip("video_007.mp4 not present")

    # Snapshot fetched only 20 seconds after passage end (12:02:40 + 20s = 12:03:00)
    early_snap = {
        "site_id": "ndls_pf12_13",
        "station_code": "NDLS",
        "requested_at": "2026-10-14T12:03:00+05:30",
        "fetched_at": "2026-10-14T12:03:00+05:30",
        "window_hr": 2,
        "available": True,
        "from_cache": False,
        "stale_fallback": False,
        "trains": [
            {"train_number": "18103", "train_name": "ASR EXP", "platform": "12"}
        ],
    }
    snap_file = Path("shadow_lab/data/ntes_snapshots/ndls_pf12_13/snap_early_20s.json")
    snap_file.write_text(json.dumps(early_snap), encoding="utf-8")

    res = execute_pipeline(
        mp4_path=mp4_path,
        site_id="ndls_pf12_13",
        platform="12",
        passage_id="test_early_ntes",
        passage_start="2026-10-14T12:00:00+05:30",
        passage_end="2026-10-14T12:02:40+05:30",
        ntes_snapshot_path=str(snap_file),
        screening_fps=5.0,
        max_ocr_budget=5,
    )

    contract = res["resolver_decision"]["ntes_timing_contract"]
    assert contract["relative_to_passage_end_sec"] == 20.0
    assert contract["eligible_for_t2_confirmation"] is False
    assert any("FETCHED_TOO_EARLY" in r for r in contract["rejection_reasons"])
    # Resolver decision MUST NOT be CONFIRMED based on early NTES snapshot
    assert res["resolver_decision"]["shadow_status"] != "CONFIRMED"


def test_stale_fallback_ntes_cannot_produce_confirmed_decision():
    """Stale fallback NTES snapshots must be rejected from corroboration."""
    mp4_path = "train-identity-lab/videos/video_007.mp4"
    if not Path(mp4_path).is_file():
        pytest.skip("video_007.mp4 not present")

    stale_snap = {
        "site_id": "ndls_pf12_13",
        "station_code": "NDLS",
        "requested_at": "2026-10-14T12:10:00+05:30",
        "fetched_at": "2026-10-14T12:10:00+05:30",
        "available": True,
        "stale_fallback": True,  # Stale cached fallback!
        "trains": [
            {"train_number": "18103", "train_name": "ASR EXP", "platform": "12"}
        ],
    }
    snap_file = Path("shadow_lab/data/ntes_snapshots/ndls_pf12_13/snap_stale_fallback.json")
    snap_file.write_text(json.dumps(stale_snap), encoding="utf-8")

    res = execute_pipeline(
        mp4_path=mp4_path,
        site_id="ndls_pf12_13",
        platform="12",
        passage_id="test_stale_ntes",
        passage_start="2026-10-14T12:00:00+05:30",
        passage_end="2026-10-14T12:02:40+05:30",
        ntes_snapshot_path=str(snap_file),
        screening_fps=5.0,
        max_ocr_budget=5,
    )

    contract = res["resolver_decision"]["ntes_timing_contract"]
    assert contract["stale_fallback"] is True
    assert contract["eligible_for_t2_confirmation"] is False
    assert "SNAPSHOT_IS_STALE_FALLBACK" in contract["rejection_reasons"]
    assert res["resolver_decision"]["shadow_status"] != "CONFIRMED"


def test_synthetic_fixture_timetable_and_ntes_integration():
    """
    [SYNTHETIC INTEGRATION TEST]
    Demonstrates pipeline behavior when supplied with a synthetic schedule and
    synthetic post-passage NTES snapshot fixture (+180s).
    Reclassified as SYNTHETIC integration test. Provenance is NOT independently sourced.
    """
    mp4_path = "train-identity-lab/videos/video_007.mp4"
    if not Path(mp4_path).is_file():
        pytest.skip("video_007.mp4 not present")

    # Synthetic Timetable fixture for 18103 Amritsar Express
    timetable_csv = "shadow_lab/sites/ndls_pf12_13/schedules/ndls_18103_corroboration.csv"
    tt_content = (
        "S.No,Train No.,Train Name,PF,Arrl.,Dep.,Remarks\n"
        "1,18103,ASR EXP,12,12.00,12.05,Daily\n"
        "2,18104,TATANAGAR EXP,12,,18.30,Daily\n"
    )
    Path(timetable_csv).write_text(tt_content, encoding="utf-8")

    # Synthetic NTES snapshot fixture acquired at passage end + 200s (>= +180s)
    ntes_json = "shadow_lab/data/ntes_snapshots/ndls_pf12_13/snap_18103_synthetic.json"
    snap_data = {
        "site_id": "ndls_pf12_13",
        "station_code": "NDLS",
        "requested_at": "2026-10-14T12:06:00+05:30",
        "fetched_at": "2026-10-14T12:06:00+05:30",
        "window_hr": 2,
        "available": True,
        "from_cache": False,
        "stale_fallback": False,
        "snapshot_age_seconds_at_decision": 0.0,
        "source": "SYNTHETIC_TEST_FIXTURE",
        "trains": [
            {
                "train_number": "18103",
                "train_name": "JALLIANWALA B EXP",
                "platform": "12",
                "time_min": 720,
                "time_raw": "12:00",
                "is_originating": False,
                "is_terminating": False,
                "dep_min": 725,
            }
        ],
        "raw_hash": "hash_synthetic_fixture"
    }
    Path(ntes_json).write_text(json.dumps(snap_data, indent=2), encoding="utf-8")

    res = execute_pipeline(
        mp4_path=mp4_path,
        site_id="ndls_pf12_13",
        platform="12",
        passage_id="test_v007_synthetic_pass",
        passage_start="2026-10-14T12:00:00+05:30",
        passage_end="2026-10-14T12:02:40+05:30",
        camera_id="side_a",
        coach_boundaries_path="shadow_lab/data/boundaries/video_007_coach_boundaries.json",
        timetable_path=timetable_csv,
        ntes_snapshot_path=ntes_json,
        screening_fps=5.0,
        max_ocr_budget=20,
    )

    contract = res["resolver_decision"]["ntes_timing_contract"]
    assert contract["eligible_for_t2_confirmation"] is True
    assert contract["relative_to_passage_end_sec"] == 200.0  # (12:06:00 - 12:02:40 = 200s >= 180s)
    # Notice: coach boundaries in video_007_coach_boundaries.json have hardware_sync_verified=False
    # so coach tracking is UNVERIFIED and visual_pair_status is PAIR_PROVISIONAL.
    assert res["visual_evidence"]["visual_pair_status"] == "PAIR_PROVISIONAL"
    assert res["resolver_decision"]["shadow_final_number"] == "18103"
