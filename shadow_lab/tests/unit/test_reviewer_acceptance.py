"""Reviewer Acceptance Criteria Tests
Validates the 9 critical acceptance criteria identified during static review:
1. NDLS passage, NTES contains unrelated PF1 train -> PF1 train must not influence resolution
2. NDLS passage, TESTX NTES snapshot -> Reject snapshot
3. NTES snapshot older than allowed TTL -> Mark stale; do not treat as fresh
4. Two numbers printed, no active-service corroboration -> Pair known; active service provisional/unknown
5. TAJ EXPRESS versus timetable TAJ EXP -> Alias-aware name match
6. OCR text NEW DELHI -> Endpoint, not train name
7. Timetable valid for wrong date/day -> Reject or mark schedule unavailable
8. Four/six service candidates with conflicting names -> Ambiguous or conflict, not forced identity
9. NTES train disappears after +3 minutes -> No negative inference from disappearance alone
"""

from datetime import datetime
import pytest

from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.models import (
    NameCandidate,
    NtesSnapshot,
    NtesTrainEntry,
    NumberCandidate,
    PassageInput,
    PassageVisualConsensus,
    TimetableRow,
    VisualObservation,
    VisualOcrEvidence,
)
from shadow_identity.ocr_adapter import OcrAdapter
from shadow_identity.reason_codes import (
    RC_AMBIGUOUS_MULTI_CANDIDATE,
    RC_NTES_CORROBORATED_FRESH,
    RC_NTES_CORROBORATED_STALE,
    RC_NTES_POST_180S_ABSENT_IGNORED,
    RC_NTES_WRONG_STATION_REJECTED,
    RC_VISUAL_PAIR_ACTIVE_SERVICE_UNCONFIRMED,
)
from shadow_identity.text_normalization import is_endpoint_text, match_name_with_aliases
from shadow_identity.timetable_matcher import TimetableMatcher


def make_ndls_passage(platforms=None, dt="2026-10-13T17:20:00+05:30", direction="unknown"):
    return PassageInput(
        site_id="ndls_pf12_13",
        passage_id="P_ACCEPTANCE_01",
        passage_start=dt,
        passage_end=dt,
        possible_platforms=platforms or ["12", "13"],
        direction=direction,
    )


def test_acceptance_1_unrelated_pf1_train_ignored():
    """Criterion 1: NDLS passage on PF 12/13, NTES contains unrelated PF1 train.
    The PF1 train must NOT influence candidate resolution or score."""
    passage = make_ndls_passage(platforms=["12", "13"])
    v_cons = PassageVisualConsensus(
        service_number_candidates=[NumberCandidate(number="12920", distinct_coaches=3)],
        stable_pair=["12920"],
        numeric_status="PAIR_PROVISIONAL",
    )
    evidence = VisualOcrEvidence(passage_id="P_ACCEPTANCE_01", passage_consensus=v_cons)

    # NTES contains unrelated train 12004 on Platform 1
    ntes = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at="2026-10-13T17:19:00+05:30",
        fetched_at="2026-10-13T17:19:00+05:30",
        trains=[
            NtesTrainEntry(train_number="12004", train_name="SHATABDI EXP", platform="1"),
            NtesTrainEntry(train_number="12920", train_name="MALWA EXP", platform="12"),
        ],
    )

    dec = resolve_shadow(passage, evidence, [], ntes_before=ntes)
    candidate_numbers = [c.train_number for c in dec.candidate_services]
    # PF 1 train must be strictly excluded
    assert "12004" not in candidate_numbers
    # PF 12 train is included
    assert "12920" in candidate_numbers


def test_acceptance_2_wrong_station_ntes_snapshot_rejected():
    """Criterion 2: NDLS passage, snapshot from TESTX station.
    The snapshot must be rejected and not influence resolution."""
    passage = make_ndls_passage()
    evidence = VisualOcrEvidence(passage_id="P_ACCEPTANCE_01", passage_consensus=PassageVisualConsensus())

    wrong_station_ntes = NtesSnapshot(
        site_id="testx_pf1_2",
        station_code="TESTX",
        requested_at="2026-10-13T17:19:00+05:30",
        fetched_at="2026-10-13T17:19:00+05:30",
        trains=[NtesTrainEntry(train_number="99999", train_name="WRONG STATION TRAIN", platform="12")],
    )

    dec = resolve_shadow(passage, evidence, [], ntes_before=wrong_station_ntes)
    assert RC_NTES_WRONG_STATION_REJECTED in dec.reason_codes
    assert not any(c.train_number == "99999" for c in dec.candidate_services)


def test_acceptance_3_ntes_snapshot_older_than_ttl_marked_stale():
    """Criterion 3: NTES snapshot older than allowed TTL (180s) must be marked stale, not fresh."""
    passage = make_ndls_passage(dt="2026-10-13T17:20:00+05:30")
    evidence = VisualOcrEvidence(passage_id="P_ACCEPTANCE_01", passage_consensus=PassageVisualConsensus())

    # Snapshot fetched 10 minutes (600s) before passage -> exceeds 180s TTL
    old_ntes = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at="2026-10-13T17:10:00+05:30",
        fetched_at="2026-10-13T17:10:00+05:30",
        snapshot_age_seconds_at_decision=600.0,
        trains=[NtesTrainEntry(train_number="12920", train_name="MALWA EXP", platform="12")],
    )

    dec = resolve_shadow(passage, evidence, [], ntes_before=old_ntes, config={"ntes_default_ttl_s": 180})
    assert RC_NTES_CORROBORATED_STALE in dec.reason_codes
    assert RC_NTES_CORROBORATED_FRESH not in dec.reason_codes


def test_acceptance_4_two_numbers_no_active_corroboration_is_provisional():
    """Criterion 4: Two numbers printed on board, but no fresh active-service corroboration.
    Pair is known, but active service direction must be PROVISIONAL/AMBIGUOUS, not CONFIRMED."""
    passage = make_ndls_passage()
    v_cons = PassageVisualConsensus(
        service_number_candidates=[
            NumberCandidate(number="12001", distinct_coaches=3),
            NumberCandidate(number="12002", distinct_coaches=3),
        ],
        stable_pair=["12001", "12002"],
        numeric_status="PAIR_CONFIRMED",
    )
    evidence = VisualOcrEvidence(passage_id="P_ACCEPTANCE_01", passage_consensus=v_cons)

    # Timetable only has static entries, but NO live NTES and NO direction corroboration
    rows = [
        TimetableRow(site_id="ndls_pf12_13", schedule_version="v1", station_code="NDLS", train_number="12001", train_name="SHATABDI", departure_local="17:20"),
    ]

    dec = resolve_shadow(passage, evidence, rows, ntes_before=None)
    # Must NOT be CONFIRMED without fresh active-service corroboration
    assert dec.shadow_status != "CONFIRMED"
    assert dec.shadow_status == "PROVISIONAL"
    assert RC_VISUAL_PAIR_ACTIVE_SERVICE_UNCONFIRMED in dec.reason_codes


def test_acceptance_5_alias_aware_name_match():
    """Criterion 5: OCR text TAJ EXPRESS versus timetable row TAJ EXP.
    Must match via alias-aware normalization."""
    site_aliases = {
        "aliases": {
            "TAJ EXPRESS": {
                "canonical_name": "TAJ EXPRESS",
                "variations": ["NDLS-VGLJ TAJ EXP", "TAJ EXP", "TAJ SUPERFAST"],
            }
        }
    }
    score = match_name_with_aliases("TAJ EXPRESS", "TAJ EXP", site_aliases=site_aliases)
    assert score >= 0.8


def test_acceptance_6_endpoint_classification():
    """Criterion 6: OCR text NEW DELHI or VGL JHANSI is classified as ENDPOINT, not train name."""
    assert is_endpoint_text("NEW DELHI")
    assert is_endpoint_text("VGL JHANSI")
    assert is_endpoint_text("NEW DELHI - VGL JHANSI")
    assert not is_endpoint_text("TAJ EXPRESS")
    assert not is_endpoint_text("SHIV GANGA EXPRESS")


def test_acceptance_7_wrong_operating_day_rejected():
    """Criterion 7: Timetable valid for wrong operating day is rejected."""
    # Passage on Tuesday (weekday = 1) at 14:00
    passage_dt = datetime.fromisoformat("2026-10-13T14:00:00+05:30")  # 2026-10-13 is Tuesday (1)
    
    # Train only runs on Sunday (day 6), daytime service at 14:00
    row_sunday_only = TimetableRow(
        site_id="ndls_pf12_13",
        schedule_version="v1",
        station_code="NDLS",
        train_number="19566",
        train_name="UTTARANCHAL EXP",
        departure_local="14:00",
        operating_days=[6],  # Only Sunday
    )

    matcher = TimetableMatcher([row_sunday_only], site_id="ndls_pf12_13")
    candidates = matcher.find_candidates(passage_dt)
    # Sunday-only train running on Tuesday must have zero candidates
    assert len(candidates) == 0


def test_acceptance_8_complex_board_conflicting_names_ambiguous():
    """Criterion 8: 4-number board with conflicting train names declares AMBIGUOUS, not forced identity."""
    passage = make_ndls_passage()
    v_cons = PassageVisualConsensus(
        service_number_candidates=[
            NumberCandidate(number="12559", distinct_coaches=3),
            NumberCandidate(number="12560", distinct_coaches=3),
            NumberCandidate(number="12451", distinct_coaches=3),
            NumberCandidate(number="12452", distinct_coaches=3),
        ],
        stable_pair=["12451", "12452", "12559", "12560"],
        number_board_type="FOUR_NUMBER",
        numeric_status="COMPLEX_BOARD",
        train_name_candidates=[],  # No train name observed to differentiate
    )
    evidence = VisualOcrEvidence(passage_id="P_ACCEPTANCE_01", passage_consensus=v_cons)

    rows = [
        TimetableRow(site_id="ndls_pf12_13", schedule_version="v1", station_code="NDLS", train_number="12559", train_name="SHIV GANGA EXP", departure_local="17:20"),
        TimetableRow(site_id="ndls_pf12_13", schedule_version="v1", station_code="NDLS", train_number="12451", train_name="SHRAM SHKTI EXP", departure_local="17:20"),
    ]

    dec = resolve_shadow(passage, evidence, rows)
    # Without name to disambiguate the complex board, resolver must declare AMBIGUOUS
    assert dec.shadow_status == "AMBIGUOUS"
    assert RC_AMBIGUOUS_MULTI_CANDIDATE in dec.reason_codes


def test_acceptance_9_ntes_train_disappearance_post_180s_non_negative():
    """Criterion 9: Train disappearance in NTES at +3 minutes is treated as possible departure (non-negative proof)."""
    passage = make_ndls_passage(dt="2026-10-13T17:20:00+05:30")
    v_cons = PassageVisualConsensus(
        service_number_candidates=[NumberCandidate(number="12920", distinct_coaches=3)],
        stable_pair=["12920"],
        numeric_status="PAIR_PROVISIONAL",
    )
    evidence = VisualOcrEvidence(passage_id="P_ACCEPTANCE_01", passage_consensus=v_cons)

    ntes_before = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at="2026-10-13T17:19:00+05:30",
        fetched_at="2026-10-13T17:19:00+05:30",
        trains=[NtesTrainEntry(train_number="12920", train_name="MALWA EXP", platform="12")],
    )

    # At +180s, the train departed so it is no longer listed in NTES snapshot
    ntes_after = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at="2026-10-13T17:23:00+05:30",
        fetched_at="2026-10-13T17:23:00+05:30",
        trains=[],  # departed
    )

    dec = resolve_shadow(passage, evidence, [], ntes_before=ntes_before, ntes_after=ntes_after, stage="T2")
    assert RC_NTES_POST_180S_ABSENT_IGNORED in dec.reason_codes
    assert dec.shadow_final_number == "12920"
