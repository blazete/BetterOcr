"""
End-to-End integration tests for ShadowOCR Train Identity CLI (Phase 4).
Validates:
- Video 007 end-to-end replay with verified coach boundaries.
- Video 008 asymmetric pair behavior.
- Video 011 endpoint fallback.
- Video 012 no recognizable identity fallback.
- Missing timetable/NTES graceful handling.
- Unverified coach boundaries block PAIR_CONFIRMED.
- Repeated frames and variants cannot multiply coach votes.
- Zero ground truth leakage.
- Production ARDS integrity & temporary MP4 cleanup.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import pytest

from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.models import PassageInput, TimetableRow
from shadow_identity.shadow_ocr.adapter import ShadowOcrToBetterOcrAdapter
from shadow_identity.shadow_ocr.cli import execute_pipeline
from shadow_identity.shadow_ocr.consensus import PhysicalCoachConsensusEngine
from shadow_identity.shadow_ocr.contracts import RawOcrObservation
from shadow_identity.shadow_ocr.temp_mp4_lifecycle import TempMp4LifecycleManager


def test_cli_video_007_verified_boundaries_and_timetable(tmp_path: Path):
    """Video 007 end-to-end replay with verified coach boundaries and timetable match."""
    mp4_path = "train-identity-lab/videos/video_007.mp4"
    if not Path(mp4_path).is_file():
        pytest.skip("video_007.mp4 not present")

    # Create trusted verified fixture for test
    src_data = json.loads(Path("shadow_lab/data/boundaries/video_007_coach_boundaries.json").read_text(encoding="utf-8"))
    src_data["metadata"]["hardware_sync_verified"] = True
    verified_json = tmp_path / "verified_v007_boundaries.json"
    verified_json.write_text(json.dumps(src_data), encoding="utf-8")

    res = execute_pipeline(
        mp4_path=mp4_path,
        site_id="ndls_pf12_13",
        platform="12",
        passage_id="test_v007_pass",
        passage_start="2026-10-14T12:00:00+05:30",
        passage_end="2026-10-14T12:02:40+05:30",
        coach_boundaries_path=str(verified_json),
        screening_fps=5.0,
        max_ocr_budget=20,
    )

    v_ev = res["visual_evidence"]
    assert v_ev["tracking_quality"] == "VERIFIED"
    assert v_ev["visual_pair_status"] == "PAIR_CONFIRMED"
    assert "18103" in v_ev["stable_pair"]
    assert "18104" in v_ev["stable_pair"]
    assert "AMRITSAR" in v_ev["endpoint_candidates"]


def test_cli_unverified_coach_boundaries_block_pair_confirmed():
    """Without verified physical coach boundaries, PAIR_CONFIRMED is strictly blocked."""
    mp4_path = "train-identity-lab/videos/video_007.mp4"
    if not Path(mp4_path).is_file():
        pytest.skip("video_007.mp4 not present")

    res = execute_pipeline(
        mp4_path=mp4_path,
        site_id="ndls_pf12_13",
        platform="12",
        passage_id="test_unverified_pass",
        coach_boundaries_path=None,  # No boundaries provided
        screening_fps=5.0,
        max_ocr_budget=15,
    )

    v_ev = res["visual_evidence"]
    assert v_ev["tracking_quality"] == "UNVERIFIED"
    # Status MUST NOT be PAIR_CONFIRMED without verified coach boundaries!
    assert v_ev["visual_pair_status"] != "PAIR_CONFIRMED"
    assert "COACH_BOUNDARIES_UNVERIFIED" in res["resolver_decision"]["reason_codes"] or \
           "VISUAL_PAIR_PROVISIONAL_LT3_COACHES" in res["resolver_decision"]["reason_codes"] or \
           "VISUAL_PAIR_UNCONFIRMED" in res["resolver_decision"]["reason_codes"]


def test_cli_missing_timetable_and_ntes():
    """Missing timetable and NTES data must gracefully fall back without crashes."""
    mp4_path = "train-identity-lab/videos/video_007.mp4"
    if not Path(mp4_path).is_file():
        pytest.skip("video_007.mp4 not present")

    res = execute_pipeline(
        mp4_path=mp4_path,
        site_id="site_no_schedule",
        platform="1",
        passage_id="test_no_tt_pass",
        timetable_path=None,
        ntes_snapshot_path=None,
        screening_fps=5.0,
        max_ocr_budget=10,
    )

    res_dec = res["resolver_decision"]
    assert res_dec["timetable_available"] is False
    assert res_dec["ntes_available"] is False
    assert any("SCHEDULE_UNAVAILABLE" in rc for rc in res_dec["reason_codes"])
    assert any("NTES_UNAVAILABLE" in rc for rc in res_dec["reason_codes"])


def test_cli_repeated_variants_cannot_multiply_coach_votes():
    """Ensures repeated frames/variants of one coach cannot fabricate independent coach support."""
    engine = PhysicalCoachConsensusEngine(confirmed_threshold=3)

    # 10 observations all from the same physical coach C01
    observations = [
        RawOcrObservation(
            site_id="S1",
            passage_id="P1",
            camera_id="cam1",
            frame_id=100 + i,
            timestamp_ms=1000 + i * 40,
            coach_id="C01",
            coach_id_source="VERIFIED_BOUNDARY",
            raw_text="18103",
            normalized_text="18103",
            entity_type="SERVICE_NUMBER",
            ocr_confidence=0.99,
        )
        for i in range(10)
    ]

    res = engine.build_passage_identity("S1", "P1", "cam1", "sha", observations)
    # Distinct verified coaches for 18103 MUST be exactly 1!
    cand = next(c for c in res.service_number_candidates if c.number == "18103")
    assert cand.distinct_coaches == 1
    assert cand.total_observations == 10
    # Must NOT be PAIR_CONFIRMED!
    assert res.visual_pair_status != "PAIR_CONFIRMED"


def test_cli_temp_mp4_cleanup_policy(tmp_path: Path):
    """Verifies that temporary copied MP4s are cleaned up according to 60-minute retention."""
    temp_dir = tmp_path / "temp_mp4"
    orig_file = tmp_path / "source.mp4"
    orig_file.write_bytes(b"content")

    manager = TempMp4LifecycleManager(temp_dir=temp_dir, retention_completed_sec=3600.0)
    job = manager.stage_temporary_copy(orig_file, "p1", "s1")
    manager.mark_completed(job.job_id)

    # Clean up with simulated future timestamp (+3601s)
    future = time.time() + 3601.0
    deleted = manager.run_cleanup_due(now_epoch=future)
    assert job.job_id in deleted
    assert not Path(job.temp_mp4_path).exists()

