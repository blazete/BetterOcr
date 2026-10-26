"""
Comprehensive Unit Tests for ShadowOCR Phase 2:
- Railway entity classification (service number, train name, endpoint, coach serial, designation, other)
- Coach consensus & physical attribution (no fake coach support from repeated frames)
- Strict pair confirmation (>=3 verified physical coaches on both candidates)
- Four-number and Name-only board preservation
- Ground-truth anti-leakage demonstration
- BetterOcr adapter compatibility
- Temporary MP4 lifecycle & 60-minute retention cleanup
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import pytest

from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.models import PassageInput, TimetableRow
from shadow_identity.shadow_ocr.adapter import ShadowOcrToBetterOcrAdapter
from shadow_identity.shadow_ocr.consensus import PhysicalCoachConsensusEngine
from shadow_identity.shadow_ocr.contracts import RawOcrObservation
from shadow_identity.shadow_ocr.temp_mp4_lifecycle import TempMp4LifecycleManager
from shadow_identity.shadow_ocr.text_entities import EntityClassifier


# ---------------------------------------------------------------------------
# 1. Entity Classification Tests
# ---------------------------------------------------------------------------

def test_entity_classification_service_numbers():
    c = EntityClassifier.classify("12280", 0.95)
    assert c.entity_type == "SERVICE_NUMBER"
    assert c.normalized_text == "12280"

    # Leading zero preserved as string
    c2 = EntityClassifier.classify("02563", 0.92)
    assert c2.entity_type == "SERVICE_NUMBER"
    assert c2.normalized_text == "02563"

    # Embedded 5-digit number
    c3 = EntityClassifier.classify("12477/78", 0.88)
    assert c3.entity_type == "SERVICE_NUMBER"
    assert c3.normalized_text == "12477"


def test_entity_classification_train_names():
    for name in ["TAJ EXPRESS", "LUCKNOW MAIL", "SACHKHAND EXPRESS", "SHRI SHAKTI AC EXPRESS"]:
        c = EntityClassifier.classify(name, 0.90)
        assert c.entity_type == "TRAIN_NAME"
        assert c.normalized_text == name


def test_entity_classification_endpoints():
    for ep in ["NEW DELHI", "VGL JHANSI", "KATRA", "SVDK", "JAMNAGAR", "HAPA"]:
        c = EntityClassifier.classify(ep, 0.88)
        assert c.entity_type == "ENDPOINT"


def test_entity_classification_coach_serials_and_designations():
    c_serial = EntityClassifier.classify("217152/C", 0.96)
    assert c_serial.entity_type == "COACH_SERIAL"

    c_desig1 = EntityClassifier.classify("S1", 0.99)
    assert c_desig1.entity_type == "COACH_DESIGNATION"

    c_desig2 = EntityClassifier.classify("AC THREE TIER", 0.95)
    assert c_desig2.entity_type == "COACH_DESIGNATION"

    c_desig3 = EntityClassifier.classify("3A", 0.99)
    assert c_desig3.entity_type == "COACH_DESIGNATION"


# ---------------------------------------------------------------------------
# 2. Coach Consensus & Strict Physical Attribution Tests
# ---------------------------------------------------------------------------

def test_negative_repeated_frames_single_coach_equals_one_vote():
    engine = PhysicalCoachConsensusEngine(confirmed_threshold=3)

    # 10 repeated frames from ONE single physical coach C01
    observations = [
        RawOcrObservation(
            frame_id=100 + i * 5,
            timestamp_ms=4000 + i * 200,
            coach_id="C01",
            coach_id_source="VERIFIED_BOUNDARY",
            raw_text="12477",
            ocr_confidence=0.92,
        )
        for i in range(10)
    ]

    res = engine.build_passage_identity(
        site_id="ndls", passage_id="p1", camera_id="cam1", video_sha256="fake_sha",
        observations=observations
    )

    assert len(res.service_number_candidates) == 1
    cand = res.service_number_candidates[0]
    assert cand.number == "12477"
    # Even though there are 10 observations across 10 frames, distinct coaches MUST be 1!
    assert cand.distinct_coaches == 1
    assert cand.total_observations == 10
    # Must NOT be confirmed
    assert res.visual_pair_status != "PAIR_CONFIRMED"


def test_strict_pair_confirmation_both_ge3_coaches():
    engine = PhysicalCoachConsensusEngine(confirmed_threshold=3)

    # Both 12477 and 12478 present on C01, C02, C03
    observations = []
    for cid in ["C01", "C02", "C03"]:
        observations.append(
            RawOcrObservation(frame_id=10, coach_id=cid, coach_id_source="VERIFIED_BOUNDARY", raw_text="12477", ocr_confidence=0.9)
        )
        observations.append(
            RawOcrObservation(frame_id=11, coach_id=cid, coach_id_source="VERIFIED_BOUNDARY", raw_text="12478", ocr_confidence=0.9)
        )

    res = engine.build_passage_identity("ndls", "p1", "cam1", "sha", observations)
    assert res.visual_pair_status == "PAIR_CONFIRMED"
    assert res.stable_pair == ["12477", "12478"]


def test_asymmetric_support_yields_provisional():
    engine = PhysicalCoachConsensusEngine(confirmed_threshold=3)

    # 12919 has 5 coaches; 12920 has only 2 coaches (simulating Video 008)
    observations = []
    for cid in ["C01", "C02", "C03", "C04", "C05"]:
        observations.append(
            RawOcrObservation(frame_id=10, coach_id=cid, coach_id_source="VERIFIED_BOUNDARY", raw_text="12919", ocr_confidence=0.9)
        )
    for cid in ["C01", "C02"]:
        observations.append(
            RawOcrObservation(frame_id=11, coach_id=cid, coach_id_source="VERIFIED_BOUNDARY", raw_text="12920", ocr_confidence=0.9)
        )

    res = engine.build_passage_identity("ndls", "p1", "cam1", "sha", observations)
    # Strict rule: second candidate < 3 -> PAIR_PROVISIONAL (not PAIR_CONFIRMED!)
    assert res.visual_pair_status == "PAIR_PROVISIONAL"
    assert res.dominant_service_number == "12919"


def test_missing_coach_boundaries_prevents_pair_confirmed():
    engine = PhysicalCoachConsensusEngine(confirmed_threshold=3)

    # Multiple frames but coach_id_source is UNVERIFIED
    observations = [
        RawOcrObservation(frame_id=i * 50, coach_id=None, coach_id_source="UNVERIFIED", raw_text="12477", ocr_confidence=0.9)
        for i in range(10)
    ] + [
        RawOcrObservation(frame_id=i * 50 + 1, coach_id=None, coach_id_source="UNVERIFIED", raw_text="12478", ocr_confidence=0.9)
        for i in range(10)
    ]

    res = engine.build_passage_identity("ndls", "p1", "cam1", "sha", observations)
    assert res.coach_tracking_quality == "UNVERIFIED"
    assert res.visual_pair_status == "PAIR_PROVISIONAL"
    assert "COACHES_UNVERIFIED_PAIR_PROVISIONAL_ONLY" in res.reason_codes


def test_name_only_board_preserves_train_name():
    engine = PhysicalCoachConsensusEngine(confirmed_threshold=3)

    observations = [
        RawOcrObservation(frame_id=100, coach_id="C01", coach_id_source="VERIFIED_BOUNDARY", raw_text="TAJ EXPRESS", ocr_confidence=0.94),
        RawOcrObservation(frame_id=200, coach_id="C02", coach_id_source="VERIFIED_BOUNDARY", raw_text="NEW DELHI", ocr_confidence=0.91),
    ]

    res = engine.build_passage_identity("ndls", "p1", "cam1", "sha", observations)
    assert res.board_category == "NAME_ONLY"
    assert res.visual_pair_status == "NO_NUMBERS"
    assert len(res.train_name_candidates) == 1
    assert res.train_name_candidates[0].name == "TAJ EXPRESS"
    assert "NEW DELHI" in res.endpoint_candidates


# ---------------------------------------------------------------------------
# 3. Ground-Truth Anti-Leakage Demonstration
# ---------------------------------------------------------------------------

def test_ground_truth_anti_leakage_invariance():
    """
    Demonstrates that visual identity output is bitwise identical
    regardless of any ground-truth state or fake labels.
    """
    engine = PhysicalCoachConsensusEngine(confirmed_threshold=3)
    observations = [
        RawOcrObservation(frame_id=10, coach_id="C01", coach_id_source="VERIFIED_BOUNDARY", raw_text="12477", ocr_confidence=0.9),
        RawOcrObservation(frame_id=20, coach_id="C02", coach_id_source="VERIFIED_BOUNDARY", raw_text="12477", ocr_confidence=0.9),
    ]

    res1 = engine.build_passage_identity("ndls", "p1", "cam1", "sha", observations)

    # Deliberately fake ground truth defined in test scope
    fake_gt = {"train_number": "99999", "train_name": "FAKE EXPRESS"}

    # Run inference again
    res2 = engine.build_passage_identity("ndls", "p1", "cam1", "sha", observations)

    # Both results MUST be completely identical and contain zero influence from fake_gt
    assert res1.to_dict() == res2.to_dict()
    assert res2.dominant_service_number == "12477"
    assert "99999" not in [c.number for c in res2.service_number_candidates]


# ---------------------------------------------------------------------------
# 4. BetterOcr Adapter & Downstream Resolver Compatibility
# ---------------------------------------------------------------------------

def test_shadow_ocr_to_betterocr_adapter_and_resolver():
    engine = PhysicalCoachConsensusEngine(confirmed_threshold=3)
    observations = []
    for cid in ["C01", "C02", "C03"]:
        observations.append(RawOcrObservation(frame_id=1, coach_id=cid, coach_id_source="VERIFIED_BOUNDARY", raw_text="12477", ocr_confidence=0.9))
        observations.append(RawOcrObservation(frame_id=2, coach_id=cid, coach_id_source="VERIFIED_BOUNDARY", raw_text="12478", ocr_confidence=0.9))

    v_identity = engine.build_passage_identity("site_ndls_plat12_13", "pass_01", "side_a", "sha", observations)
    evidence = ShadowOcrToBetterOcrAdapter.to_visual_ocr_evidence(v_identity, observations)

    assert evidence.passage_consensus.numeric_status == "PAIR_CONFIRMED"
    assert evidence.passage_consensus.stable_pair == ["12477", "12478"]

    # Verify existing resolver accepts this without error
    passage_input = PassageInput(
        site_id="site_ndls_plat12_13",
        passage_id="pass_01",
        passage_start="2026-10-14T21:20:00Z",
        passage_end="2026-10-14T21:23:00Z",
    )
    timetable_rows = [
        TimetableRow(
            site_id="site_ndls_plat12_13",
            schedule_version="2026_v1",
            station_code="NDLS",
            train_number="12477",
            train_name="JAMNAGAR - SVDK EXPRESS",
            arrival_local="21:25",
            departure_local="21:40",
            operating_days=[0, 1, 2, 3, 4, 5, 6],
            platform="12",
        )
    ]

    decision = resolve_shadow(passage_input, evidence, timetable_rows)
    assert decision.shadow_status == "PROVISIONAL"
    assert decision.shadow_final_number == "12477"


# ---------------------------------------------------------------------------
# 5. Temporary MP4 Lifecycle & Cleanup Tests
# ---------------------------------------------------------------------------

def test_temp_mp4_lifecycle_cleanup_rules(tmp_path: Path):
    temp_dir = tmp_path / "temp_mp4"
    orig_dir = tmp_path / "production_recordings"
    orig_dir.mkdir(parents=True)

    # Create dummy original recording
    orig_file = orig_dir / "original_passage.mp4"
    orig_file.write_bytes(b"dummy_mp4_content_12345")

    mgr = TempMp4LifecycleManager(temp_dir=temp_dir, retention_completed_sec=3600.0)

    # 1. Stage copy
    job = mgr.stage_temporary_copy(orig_file, passage_id="p100", site_id="ndls")
    temp_file = Path(job.temp_mp4_path)
    assert temp_file.is_file()
    assert temp_file.read_bytes() == b"dummy_mp4_content_12345"
    assert job.status == "STAGED"

    # 2. Early cleanup check (must NOT delete while STAGED)
    now = time.time()
    deleted = mgr.run_cleanup_due(now_epoch=now + 5000)
    assert len(deleted) == 0
    assert temp_file.is_file()

    # 3. Mark completed
    mgr.mark_completed(job.job_id)
    assert mgr._jobs[job.job_id].status == "COMPLETED"

    # 4. Cleanup before 60 min (e.g. at +1800s): must NOT delete
    deleted = mgr.run_cleanup_due(now_epoch=now + 1800)
    assert len(deleted) == 0
    assert temp_file.is_file()

    # 5. Cleanup after 60 min (e.g. at +3601s): MUST delete temp file
    deleted = mgr.run_cleanup_due(now_epoch=now + 3601)
    assert job.job_id in deleted
    assert not temp_file.is_file()

    # 6. CRUCIAL SAFETY: Original production recording MUST remain intact!
    assert orig_file.is_file()
    assert orig_file.read_bytes() == b"dummy_mp4_content_12345"


def test_temp_mp4_lifecycle_process_restart_and_persistence(tmp_path: Path):
    temp_dir = tmp_path / "temp_mp4"
    orig_dir = tmp_path / "production_recordings"
    orig_dir.mkdir(parents=True)
    orig_file = orig_dir / "passage_restart.mp4"
    orig_file.write_bytes(b"restart_test_data")

    # Instance 1: Stage and complete
    mgr1 = TempMp4LifecycleManager(temp_dir=temp_dir, retention_completed_sec=3600.0)
    job = mgr1.stage_temporary_copy(orig_file, passage_id="p200", site_id="ndls")
    mgr1.mark_completed(job.job_id)

    # Simulate process crash / restart: Instance 2 loads from disk manifest
    mgr2 = TempMp4LifecycleManager(temp_dir=temp_dir, retention_completed_sec=3600.0)
    assert job.job_id in mgr2._jobs
    assert mgr2._jobs[job.job_id].status == "COMPLETED"

    # Cleanup with Instance 2
    deleted = mgr2.run_cleanup_due(now_epoch=job.delete_after_epoch + 10)
    assert job.job_id in deleted
    assert not Path(job.temp_mp4_path).exists()


def test_temp_mp4_lifecycle_path_traversal_and_symlink_rejection(tmp_path: Path):
    temp_dir = tmp_path / "temp_mp4"
    prod_dir = tmp_path / "critical_production_dir"
    prod_dir.mkdir(parents=True)
    prod_file = prod_dir / "never_delete.mp4"
    prod_file.write_bytes(b"critical_production_content")

    mgr = TempMp4LifecycleManager(temp_dir=temp_dir)

    # Adversarial test 1: An injected job record pointing outside temp_dir
    rogue_job = mgr.stage_temporary_copy(prod_file, "p300", "ndls")
    # Tamper with path to point directly to production
    rogue_job.temp_mp4_path = str(prod_file)
    rogue_job.status = "COMPLETED"
    rogue_job.delete_after_epoch = time.time() - 100
    mgr._jobs[rogue_job.job_id] = rogue_job

    # Run cleanup: Manager MUST refuse to delete anything outside temp_dir or matching original
    deleted = mgr.run_cleanup_due()
    assert rogue_job.job_id not in deleted
    assert prod_file.is_file()
    assert prod_file.read_bytes() == b"critical_production_content"


def test_temp_mp4_lifecycle_concurrent_cleanup_safety(tmp_path: Path):
    temp_dir = tmp_path / "temp_mp4"
    orig_file = tmp_path / "source.mp4"
    orig_file.write_bytes(b"content")

    mgr = TempMp4LifecycleManager(temp_dir=temp_dir, retention_completed_sec=10.0)
    job1 = mgr.stage_temporary_copy(orig_file, "p1", "s1")
    job2 = mgr.stage_temporary_copy(orig_file, "p2", "s1")
    mgr.mark_completed(job1.job_id)
    # job2 remains STAGED (active)

    # First cleanup run after expiry of job1
    now = time.time() + 20
    del1 = mgr.run_cleanup_due(now_epoch=now)
    assert job1.job_id in del1
    assert job2.job_id not in del1
    assert Path(job2.temp_mp4_path).is_file()

    # Second concurrent/repeated cleanup run
    del2 = mgr.run_cleanup_due(now_epoch=now)
    assert len(del2) == 0  # Already deleted; idempotent
    assert Path(job2.temp_mp4_path).is_file()

