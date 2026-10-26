"""
Tests for Package P1.2: Fail-safe candidate decision gate and route evidence.
"""
from datetime import datetime
import pytest

from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.models import (
    CandidateService,
    NameCandidate,
    NumberCandidate,
    PassageInput,
    PassageVisualConsensus,
    TimetableRow,
    VisualObservation,
    VisualOcrEvidence,
)


def test_video004_unverified_single_number_yields_unknown_preserving_raw_candidate():
    """
    Video 004: OCR produces 02276, but coach tracking is UNVERIFIED and there is
    no independent timetable/NTES corroboration.
    Must return UNKNOWN, final_number=None, while preserving 02276 in candidate list.
    """
    passage = PassageInput(
        site_id="ndls_pf12_13",
        passage_id="video_004",
        camera_id="cam1",
        passage_start="2026-10-12T20:00:00",
        passage_end="2026-10-12T20:03:00",
    )

    consensus = PassageVisualConsensus(
        service_number_candidates=[
            NumberCandidate(number="02276", distinct_coaches=1, total_observations=2, mean_confidence=0.91)
        ],
        stable_pair=["02276"],
        train_name_candidates=[],
        endpoint_candidates=["SUBEDARGANJ"],
        number_board_type="TWO_NUMBER",
        numeric_status="PAIR_PROVISIONAL",
    )

    visual_evidence = VisualOcrEvidence(
        passage_id="video_004",
        ocr_model_version="PP-OCRv6",
        preprocessing_profile="P1",
        per_observation=[],
        passage_consensus=consensus,
    )

    # Empty schedule / timetable (no matching timetable for 02276)
    decision = resolve_shadow(
        passage=passage,
        visual_evidence=visual_evidence,
        site_schedule_rows=[],
        stage="T1",
    )

    assert decision.shadow_status in ("UNKNOWN", "CONFLICT"), (
        f"Expected UNKNOWN or CONFLICT without authentic corroboration, got {decision.shadow_status}"
    )
    assert decision.shadow_final_number is None, (
        f"Active train number must be None without independent corroboration, got {decision.shadow_final_number}"
    )
    # 02276 must be preserved in ranked candidates
    cand_nums = [c.train_number for c in decision.candidate_services]
    assert "02276" in cand_nums, f"Raw observed candidate 02276 must be retained, got {cand_nums}"


def test_bidirectional_pair_without_direction_yields_ambiguous():
    """
    Video 007 / Video 001 without timetable or direction evidence:
    When a pair like 18103/18104 or 15707/15708 is seen without independent direction proof,
    must return AMBIGUOUS with final_number=None, not picking an arbitrary first member.
    """
    passage = PassageInput(
        site_id="ndls_pf12_13",
        passage_id="video_007",
        camera_id="cam1",
        passage_start=None,
        passage_end=None,
    )

    consensus = PassageVisualConsensus(
        service_number_candidates=[
            NumberCandidate(number="18103", distinct_coaches=3, total_observations=5, mean_confidence=0.95),
            NumberCandidate(number="18104", distinct_coaches=3, total_observations=4, mean_confidence=0.93),
        ],
        stable_pair=["18103", "18104"],
        train_name_candidates=[],
        endpoint_candidates=["AMRITSAR"],
        number_board_type="TWO_NUMBER",
        numeric_status="PAIR_CONFIRMED",
    )

    visual_evidence = VisualOcrEvidence(
        passage_id="video_007",
        ocr_model_version="PP-OCRv6",
        preprocessing_profile="P1",
        per_observation=[],
        passage_consensus=consensus,
    )

    decision = resolve_shadow(
        passage=passage,
        visual_evidence=visual_evidence,
        site_schedule_rows=[],
        stage="T1",
    )

    assert decision.shadow_status == "AMBIGUOUS", f"Expected AMBIGUOUS, got {decision.shadow_status}"
    assert decision.shadow_final_number is None, f"Expected final_number=None, got {decision.shadow_final_number}"


def test_curated_route_endpoints_match():
    """
    Endpoints matching site_aliases entries (e.g. TAJ EXPRESS: NDLS + VGL JHANSI -> 12279/12280)
    produce ROUTE_ENDPOINT evidence without fabricating numbers.
    """
    passage = PassageInput(
        site_id="ndls_pf12_13",
        passage_id="test_taj",
        camera_id="cam1",
        passage_start="2026-10-12T21:40:00",
        passage_end="2026-10-12T21:45:00",
        observed_platform="12",
    )

    consensus = PassageVisualConsensus(
        service_number_candidates=[],
        stable_pair=[],
        train_name_candidates=[
            NameCandidate(name="TAJ EXPRESS", distinct_coaches=2, total_observations=3, mean_confidence=0.92)
        ],
        endpoint_candidates=["NEW DELHI", "VGL JHANSI"],
        number_board_type="NAME_ONLY",
        numeric_status="NOT_NUMBERED",
    )

    visual_evidence = VisualOcrEvidence(
        passage_id="test_taj",
        ocr_model_version="PP-OCRv6",
        preprocessing_profile="P1",
        per_observation=[],
        passage_consensus=consensus,
    )

    site_aliases = {
        "aliases": {
            "TAJ EXPRESS": {
                "canonical_name": "TAJ EXPRESS",
                "service_numbers": ["12279", "12280"],
                "endpoints": ["NEW DELHI", "VGL JHANSI", "GWALIOR"],
                "variations": ["TAJ EXP", "TAJ EXPRESS"],
            }
        }
    }

    tt_row = TimetableRow(
        site_id="ndls_pf12_13",
        schedule_version="1.0",
        train_number="12280",
        train_name="TAJ EXPRESS",
        station_code="NDLS",
        platform="12",
        arrival_local="21:40",
        departure_local=None,
        operating_days=[0, 1, 2, 3, 4, 5, 6],
    )

    decision = resolve_shadow(
        passage=passage,
        visual_evidence=visual_evidence,
        site_schedule_rows=[tt_row],
        site_aliases=site_aliases,
        stage="T1",
    )

    # 12280 should be top candidate with ROUTE_ENDPOINT evidence
    top = decision.candidate_services[0]
    assert top.train_number == "12280"
    assert any("ROUTE_ENDPOINT" in ev for ev in top.evidence_list), f"Expected ROUTE_ENDPOINT in {top.evidence_list}"
    # Must NOT be CONFIRMED (endpoint match alone never confers CONFIRMED)
    assert decision.shadow_status != "CONFIRMED"
