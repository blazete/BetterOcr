"""
Regression tests for BetterOcr / ShadowOCR V1 Package A1:
Correct Direction Verification Safety Defect.

Specification Rules:
active_direction_verified = (
    direction_evidence is not None
    and direction_evidence.authentic is True
    and direction_evidence.fresh is True
    and direction_evidence.passage_link_verified is True
    and direction_evidence.source_authorized is True
    and direction_evidence.direction_matches_candidate is True
)

1. Name only -> direction false.
2. Name + timetable match -> direction false.
3. Name + platform match -> direction false.
4. Name + stale NTES -> direction false.
5. Name + synthetic NTES -> direction false.
6. Contradictory directional evidence -> direction false and conflict represented.
7. Authenticated, fresh, passage-linked independent evidence -> eligible for direction verification.
8. No timestamp -> no invented timestamp and no time-based direction verification.
"""

import pytest
from datetime import datetime
from shadow_identity.models import (
    CandidateService,
    DirectionEvidence,
    NameCandidate,
    NtesSnapshot,
    NtesTrainEntry,
    NumberCandidate,
    PassageInput,
    PassageVisualConsensus,
    TimetableRow,
    VisualOcrEvidence,
)
from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.reason_codes import (
    RC_DIRECTION_CONFLICT,
    RC_NAME_ONLY_ACTIVE_DIRECTION_UNVERIFIED,
)


def _make_base_passage(start_dt="2026-10-14T17:15:00+05:30", platforms=None):
    return PassageInput(
        site_id="ndls_pf12_13",
        installation_id="inst_01",
        passage_id="P_DIR_TEST_01",
        camera_id="side_a",
        passage_start=start_dt,
        passage_end=start_dt,
        possible_platforms=platforms or ["12", "13"],
        observed_platform=None,
        direction="unknown",
    )


def _make_name_only_evidence(train_name="SHIV GANGA EXP"):
    v_cons = PassageVisualConsensus(
        service_number_candidates=[],
        train_name_candidates=[NameCandidate(name=train_name, distinct_coaches=3)],
        numeric_status="NOT_NUMBERED",
        recognized_train_name=train_name,
        service_family_candidates=["12559", "12560"],
    )
    return VisualOcrEvidence(
        passage_id="P_DIR_TEST_01",
        passage_consensus=v_cons,
        recognized_train_name=train_name,
        service_family_candidates=["12559", "12560"],
    )


def _make_sample_rows():
    return [
        TimetableRow(
            site_id="ndls_pf12_13",
            schedule_version="v1",
            station_code="NDLS",
            train_number="12560",
            train_name="SHIV GANGA EXP (NDLS-BSBS)",
            platform="12",
            departure_local="17:20",
            arrival_local=None,
            operating_days=[0, 1, 2, 3, 4, 5, 6],
            service_type="passenger",
        ),
        TimetableRow(
            site_id="ndls_pf12_13",
            schedule_version="v1",
            station_code="NDLS",
            train_number="12559",
            train_name="SHIV GANGA EXP (BSBS-NDLS)",
            platform="12",
            departure_local=None,
            arrival_local="08:30",
            operating_days=[0, 1, 2, 3, 4, 5, 6],
            service_type="passenger",
        ),
    ]


def test_1_name_only_direction_false():
    """1. Name only -> direction false."""
    passage = _make_base_passage()
    evidence = _make_name_only_evidence("SHIV GANGA EXP")
    dec = resolve_shadow(passage, evidence, [])
    assert dec.active_direction_verified is False
    assert dec.active_service_number is None


def test_2_name_plus_timetable_match_direction_false():
    """2. Name + timetable match -> direction false."""
    passage = _make_base_passage()
    evidence = _make_name_only_evidence("SHIV GANGA EXP")
    rows = _make_sample_rows()
    dec = resolve_shadow(passage, evidence, rows)
    # Timetable time-window match alone never verifies direction!
    assert dec.active_direction_verified is False
    assert dec.shadow_status != "CONFIRMED"


def test_3_name_plus_platform_match_direction_false():
    """3. Name + platform match -> direction false."""
    passage = _make_base_passage()
    passage.observed_platform = "12"
    passage.possible_platforms = ["12"]
    evidence = _make_name_only_evidence("SHIV GANGA EXP")
    rows = _make_sample_rows()
    dec = resolve_shadow(passage, evidence, rows)
    assert dec.active_direction_verified is False


def test_4_name_plus_stale_ntes_direction_false():
    """4. Name + stale NTES -> direction false."""
    passage = _make_base_passage()
    evidence = _make_name_only_evidence("SHIV GANGA EXP")
    rows = _make_sample_rows()
    stale_ntes = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at="2026-10-14T17:15:00+05:30",
        fetched_at="2026-10-14T16:00:00+05:30",
        stale_fallback=True,
        snapshot_age_seconds_at_decision=4500.0,
        trains=[NtesTrainEntry(train_number="12560", train_name="SHIV GANGA EXP", platform="12")],
    )
    dec = resolve_shadow(passage, evidence, rows, ntes_before=stale_ntes)
    assert dec.active_direction_verified is False


def test_5_name_plus_synthetic_ntes_direction_false():
    """5. Name + synthetic NTES -> direction false."""
    passage = _make_base_passage()
    evidence = _make_name_only_evidence("SHIV GANGA EXP")
    rows = _make_sample_rows()
    synthetic_ntes = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at="2026-10-14T17:15:00+05:30",
        fetched_at="2026-10-14T17:15:00+05:30",
        stale_fallback=False,
        snapshot_age_seconds_at_decision=0.0,
        source="SYNTHETIC_TEST_FIXTURE",
        raw_hash="hash_synthetic_fixture",
        trains=[NtesTrainEntry(train_number="12560", train_name="SHIV GANGA EXP", platform="12")],
    )
    dec = resolve_shadow(passage, evidence, rows, ntes_before=synthetic_ntes)
    assert dec.active_direction_verified is False


def test_6_contradictory_directional_evidence_direction_false_and_conflict():
    """6. Contradictory directional evidence -> direction false and conflict represented."""
    passage = _make_base_passage()
    evidence = _make_name_only_evidence("SHIV GANGA EXP")
    rows = _make_sample_rows()
    # Candidate 12560 is DN (NDLS-BSBS), but independent authentic direction is UP
    dir_ev = DirectionEvidence(
        direction="UP",
        authentic=True,
        fresh=True,
        passage_link_verified=True,
        source_authorized=True,
        direction_matches_candidate=False,
        has_conflict=True,
    )
    dec = resolve_shadow(passage, evidence, rows, direction_evidence=dir_ev)
    assert dec.active_direction_verified is False
    assert (
        RC_DIRECTION_CONFLICT in dec.reason_codes
        or dec.shadow_status in ("CONFLICT", "AMBIGUOUS", "UNKNOWN")
    )


def test_7_authenticated_fresh_passage_linked_evidence_eligible():
    """7. Authenticated, fresh, passage-linked independent evidence -> eligible for direction verification."""
    passage = _make_base_passage()
    evidence = _make_name_only_evidence("SHIV GANGA EXP")
    rows = _make_sample_rows()
    # Candidate 12560 is DN, independent authentic sensor verified DN passage
    dir_ev = DirectionEvidence(
        direction="DN",
        authentic=True,
        fresh=True,
        passage_link_verified=True,
        source_authorized=True,
        direction_matches_candidate=True,
        has_conflict=False,
    )
    dec = resolve_shadow(passage, evidence, rows, direction_evidence=dir_ev)
    assert dec.active_direction_verified is True
    assert dec.active_service_number == "12560"


def test_8_no_timestamp_no_time_based_direction_verification():
    """8. No timestamp -> no invented timestamp and no time-based direction verification."""
    passage = _make_base_passage(start_dt="")
    passage.passage_start = ""
    passage.passage_end = ""
    evidence = _make_name_only_evidence("SHIV GANGA EXP")
    rows = _make_sample_rows()
    dec = resolve_shadow(passage, evidence, rows)
    assert dec.active_direction_verified is False
    assert dec.shadow_status != "CONFIRMED"
