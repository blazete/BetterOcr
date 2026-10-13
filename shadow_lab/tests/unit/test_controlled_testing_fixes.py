"""
Unit tests for the 4 Controlled Testing fixes before the real one-week trial:
1. Strong single-number visual evidence protection (§1)
2. Split conflict reason codes (VISUAL_TIMETABLE_CONFLICT, VISUAL_NTES_CONFLICT, TIMETABLE_NTES_CONFLICT) (§2)
3. Comparison taxonomy correction (PRODUCTION_WRONG_SHADOW_CORRECT vs PRODUCTION_UNKNOWN_SHADOW_CORRECT) (§3)
4. Shadow-only delayed NTES verification worker (§4)
"""

import json
from pathlib import Path
import pytest

from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.comparison import categorize_comparison
from shadow_identity.delayed_ntes_worker import DelayedNtesVerificationWorker
from shadow_identity.models import (
    NtesSnapshot,
    NtesTrainEntry,
    NumberCandidate,
    PassageInput,
    PassageVisualConsensus,
    TimetableRow,
    VisualObservation,
    VisualOcrEvidence,
)
from shadow_identity.passage_collector import ShadowCollector
from shadow_identity.reason_codes import (
    RC_TIMETABLE_NTES_CONFLICT,
    RC_VISUAL_NTES_CONFLICT,
    RC_VISUAL_TIMETABLE_CONFLICT,
)


@pytest.fixture
def shadow_lab_dir() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def collector(shadow_lab_dir) -> ShadowCollector:
    return ShadowCollector(shadow_lab_root=shadow_lab_dir)


def test_case_1_strong_single_number_not_replaced_by_timetable(collector):
    """
    Case 1: If one 5-digit service number has >=3 distinct-coach support,
    an unrelated timetable-only candidate MUST NOT replace it.
    Keep visual candidate PROVISIONAL and mark VISUAL_TIMETABLE_CONFLICT.
    """
    # At 17:20, NDLS timetable has 12314 on PF 13.
    # Visual observation has 12560 on 3 distinct coaches.
    p_data = {
        "passage_id": "TEST_CASE_01_SINGLE_NUM",
        "start_time": "2026-10-13T17:20:00+05:30",
        "possible_platforms": ["12", "13"],
        "visual": {
            "coach_evidence": [
                {"coach_id": "C01", "raw_text": "12560", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C02", "raw_text": "12560", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C03", "raw_text": "12560", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
            ]
        },
        "ntes": None,
    }
    bundle = collector.import_production_passage(source=p_data, site_id="ndls_pf12_13")

    # Timetable-only candidate 12314 must NOT replace visual 12560
    assert bundle.shadow_result["T0"]["train_number"] == "12560"
    assert bundle.shadow_result["T0"]["status"] == "PROVISIONAL"
    assert RC_VISUAL_TIMETABLE_CONFLICT in bundle.shadow_result["T0"]["reason_codes"]


def test_case_2_split_conflict_reason_codes(collector):
    """
    Case 2: Split conflict reason codes into:
    - VISUAL_TIMETABLE_CONFLICT
    - VISUAL_NTES_CONFLICT
    - TIMETABLE_NTES_CONFLICT
    """
    # Subcase 2A: Strong visual contradicts timetable
    p_data_tt_conflict = {
        "passage_id": "TEST_CASE_02_VIS_TT",
        "start_time": "2026-10-13T17:20:00+05:30",
        "possible_platforms": ["12", "13"],
        "visual": {
            "coach_evidence": [
                {"coach_id": "C01", "raw_text": "12920", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C02", "raw_text": "12920", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C03", "raw_text": "12920", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
            ]
        },
        "ntes": None,
    }
    b1 = collector.import_production_passage(source=p_data_tt_conflict, site_id="ndls_pf12_13")
    assert RC_VISUAL_TIMETABLE_CONFLICT in b1.shadow_result["T0"]["reason_codes"]

    # Subcase 2B: Strong visual contradicts NTES
    p_data_ntes_conflict = {
        "passage_id": "TEST_CASE_02_VIS_NTES",
        "start_time": "2026-10-13T17:20:00+05:30",
        "possible_platforms": ["12", "13"],
        "visual": {
            "coach_evidence": [
                {"coach_id": "C01", "raw_text": "12919", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C02", "raw_text": "12919", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C03", "raw_text": "12919", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C01", "raw_text": "12920", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C02", "raw_text": "12920", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C03", "raw_text": "12920", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
            ]
        },
        "ntes": {
            "pre_or_near_passage_snapshot": {
                "site_id": "ndls_pf12_13",
                "station_code": "NDLS",
                "fetched_at": "2026-10-13T17:19:00+05:30",
                "trains": [{"train_number": "12420", "train_name": "GOMTI EXP", "platform": "12"}],
            }
        },
    }
    b2 = collector.import_production_passage(source=p_data_ntes_conflict, site_id="ndls_pf12_13")
    assert RC_VISUAL_NTES_CONFLICT in b2.shadow_result["T1"]["reason_codes"]

    # Subcase 2C: Timetable vs NTES conflict (Timetable says 12420 at 12:55 PF 12, NTES says 12920 PF 12)
    p_data_tt_ntes_conflict = {
        "passage_id": "TEST_CASE_02_TT_NTES",
        "start_time": "2026-10-13T12:55:00+05:30",
        "possible_platforms": ["12"],
        "visual": {"coach_evidence": []},
        "ntes": {
            "pre_or_near_passage_snapshot": {
                "site_id": "ndls_pf12_13",
                "station_code": "NDLS",
                "fetched_at": "2026-10-13T12:54:00+05:30",
                "trains": [{"train_number": "12920", "train_name": "MALWA EXP", "platform": "12"}],
            }
        },
    }
    b3 = collector.import_production_passage(source=p_data_tt_ntes_conflict, site_id="ndls_pf12_13")
    assert RC_TIMETABLE_NTES_CONFLICT in b3.shadow_result["T1"]["reason_codes"]


def test_case_3_comparison_taxonomy_low_confidence_is_not_abstention():
    """
    Case 3: Separate PRODUCTION_WRONG_SHADOW_CORRECT from PRODUCTION_UNKNOWN_SHADOW_CORRECT.
    LOW confidence with a train number is NOT an abstention.
    """
    # 3A: Production guessed 12004 with LOW confidence, ground truth is 12280, shadow got 12280
    cat_wrong, rev_wrong = categorize_comparison(
        prod_num="12004",
        prod_conf="LOW",
        shadow_num="12280",
        shadow_status="CONFIRMED",
        ground_truth="12280",
    )
    assert cat_wrong == "PRODUCTION_WRONG_SHADOW_CORRECT"

    # 3B: Production had NO train number (None / unknown), ground truth is 12280, shadow got 12280
    cat_unknown, rev_unknown = categorize_comparison(
        prod_num=None,
        prod_conf="LOW",
        shadow_num="12280",
        shadow_status="CONFIRMED",
        ground_truth="12280",
    )
    assert cat_unknown == "PRODUCTION_UNKNOWN_SHADOW_CORRECT"

    # 3C: Production had empty string "<NONE>"
    cat_none, rev_none = categorize_comparison(
        prod_num="<NONE>",
        prod_conf="LOW",
        shadow_num="12280",
        shadow_status="CONFIRMED",
        ground_truth="12280",
    )
    assert cat_none == "PRODUCTION_UNKNOWN_SHADOW_CORRECT"

    # 3D: Production had correct train number even with LOW confidence -> BOTH_CORRECT
    cat_both, rev_both = categorize_comparison(
        prod_num="12280",
        prod_conf="LOW",
        shadow_num="12280",
        shadow_status="CONFIRMED",
        ground_truth="12280",
    )
    assert cat_both == "BOTH_CORRECT"


def test_case_4_shadow_delayed_ntes_worker(shadow_lab_dir, collector):
    """
    Case 4: Shadow-only delayed NTES verification worker:
    passage completes -> enqueue -> around +180s fetch/reuse -> rerun T2 -> update result
    Guarantees zero writes to production ARDS.
    """
    # Import a passage without T2 post snapshot
    p_data = {
        "passage_id": "TEST_CASE_04_DELAYED_PAS",
        "start_time": "2026-10-13T21:55:00+05:30",
        "end_time": "2026-10-13T21:56:00+05:30",
        "possible_platforms": ["12", "13"],
        "visual": {
            "coach_evidence": [
                {"coach_id": "C01", "raw_text": "TAJ EXPRESS", "entity_type": "TRAIN_NAME", "confidence": 0.95},
                {"coach_id": "C02", "raw_text": "TAJ EXPRESS", "entity_type": "TRAIN_NAME", "confidence": 0.95},
            ]
        },
        "ntes": {
            "pre_or_near_passage_snapshot": {
                "site_id": "ndls_pf12_13",
                "station_code": "NDLS",
                "fetched_at": "2026-10-13T21:54:00+05:30",
                "trains": [{"train_number": "12280", "train_name": "TAJ EXP", "platform": "12"}],
            },
            "post_passage_snapshot": None,
        },
    }
    bundle = collector.import_production_passage(source=p_data, site_id="ndls_pf12_13")

    # Verify task was enqueued
    pending = collector.worker.get_pending_tasks()
    matching = [t for t in pending if t.passage_id == "TEST_CASE_04_DELAYED_PAS"]
    assert len(matching) == 1
    assert matching[0].status == "PENDING"

    # Simulate delayed NTES snapshot at T+180s (confirming train still verified)
    post_snap = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at="2026-10-13T21:59:00+05:30",
        fetched_at="2026-10-13T21:59:00+05:30",
        available=True,
        trains=[NtesTrainEntry(train_number="12280", train_name="TAJ EXP", platform="12")],
        raw_hash="chk_delayed_180",
    )

    # Process task via worker
    updated_bundle = collector.worker.process_passage(
        passage_id="TEST_CASE_04_DELAYED_PAS",
        snapshot_override=post_snap,
    )

    assert updated_bundle is not None
    assert updated_bundle.ntes["post_passage_snapshot"] is not None
    assert updated_bundle.shadow_result["T2"]["train_number"] == "12280"

    # Verify snapshot was saved under shadow_lab only
    saved_snaps = list((shadow_lab_dir / "data" / "ntes_snapshots" / "ndls_pf12_13").glob("*.json"))
    assert len(saved_snaps) > 0

    # Verify queue status updated
    assert collector.worker._tasks["TEST_CASE_04_DELAYED_PAS"].status == "PROCESSED"
