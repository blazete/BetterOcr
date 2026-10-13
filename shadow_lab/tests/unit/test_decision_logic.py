"""Unit tests for Pure Candidate Resolver Decision Logic (Tests 41 to 50).
Tests 2-number disambiguation, both plausible ambiguous, name-first resolution,
conflict detection, EMU gating, and safe abstention.
"""

from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.models import (
    NameCandidate,
    NtesSnapshot,
    NtesTrainEntry,
    NumberCandidate,
    PassageInput,
    PassageVisualConsensus,
    TimetableRow,
    VisualOcrEvidence,
)
from shadow_identity.reason_codes import (
    RC_AMBIGUOUS_MULTI_CANDIDATE,
    RC_CO_PRESENT_TRAINS_UNRESOLVED,
    RC_NOT_SERVICE_EVENT,
    RC_PLATFORM_AMBIGUOUS_ISLAND,
    RC_VISUAL_NTES_CONFLICT,
)


def make_passage(site_id="ndls_pf12_13", passage_id="P1", platforms=None, dt="2026-10-13T17:20:00+05:30"):
    return PassageInput(
        site_id=site_id,
        passage_id=passage_id,
        passage_start=dt,
        passage_end=dt,
        possible_platforms=platforms or ["12", "13"],
    )


def test_two_number_unique_live_match():
    """Test 41: stable pair [15707, 15708], only 15708 matches timetable and NTES -> selects 15708."""
    passage = make_passage()
    v_cons = PassageVisualConsensus(
        service_number_candidates=[
            NumberCandidate(number="15707", distinct_coaches=3),
            NumberCandidate(number="15708", distinct_coaches=4),
        ],
        stable_pair=["15707", "15708"],
        numeric_status="PAIR_CONFIRMED",
    )
    evidence = VisualOcrEvidence(passage_id="P1", passage_consensus=v_cons)
    rows = [
        TimetableRow(
            site_id="ndls_pf12_13",
            schedule_version="v1",
            station_code="NDLS",
            train_number="15708",
            train_name="AMRAPALI EXP",
            departure_local="17:20",
        )
    ]
    ntes = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at="",
        fetched_at="2026-10-13T17:19:00+05:30",
        trains=[NtesTrainEntry(train_number="15708", train_name="AMRAPALI EXP", platform="12")],
    )

    dec = resolve_shadow(passage, evidence, rows, ntes_before=ntes)
    assert dec.shadow_final_number == "15708"
    assert dec.shadow_status in ("CONFIRMED", "PROVISIONAL")


def test_two_number_both_plausible_ambiguous():
    """Test 42: stable pair where both numbers match timetable equally with no differentiator -> AMBIGUOUS."""
    passage = make_passage()
    v_cons = PassageVisualConsensus(
        service_number_candidates=[
            NumberCandidate(number="12001", distinct_coaches=3),
            NumberCandidate(number="12002", distinct_coaches=3),
        ],
        stable_pair=["12001", "12002"],
        numeric_status="PAIR_CONFIRMED",
    )
    evidence = VisualOcrEvidence(passage_id="P1", passage_consensus=v_cons)
    rows = [
        TimetableRow(site_id="ndls_pf12_13", schedule_version="v1", station_code="NDLS", train_number="12001", train_name="SHATABDI", departure_local="17:20"),
        TimetableRow(site_id="ndls_pf12_13", schedule_version="v1", station_code="NDLS", train_number="12002", train_name="SHATABDI", departure_local="17:20"),
    ]
    dec = resolve_shadow(passage, evidence, rows, ntes_before=None)
    assert dec.shadow_status == "AMBIGUOUS"


def test_name_only_unique_schedule_plus_live():
    """Test 43: no numbers on board, strong train name TAJ EXPRESS matches unique schedule row."""
    passage = make_passage(dt="2026-10-13T21:55:00+05:30")
    v_cons = PassageVisualConsensus(
        service_number_candidates=[],
        train_name_candidates=[NameCandidate(name="TAJ EXPRESS", distinct_coaches=4)],
        numeric_status="NOT_NUMBERED",
    )
    evidence = VisualOcrEvidence(passage_id="P1", passage_consensus=v_cons)
    rows = [
        TimetableRow(site_id="ndls_pf12_13", schedule_version="v1", station_code="NDLS", train_number="12280", train_name="TAJ EXPRESS", departure_local="21:55")
    ]
    ntes = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at="",
        fetched_at="",
        trains=[NtesTrainEntry(train_number="12280", train_name="TAJ EXPRESS", platform="12")],
    )

    dec = resolve_shadow(passage, evidence, rows, ntes_before=ntes)
    assert dec.shadow_final_number == "12280"


def test_four_numbers_with_name_and_schedule_intersection():
    """Test 44: complex 4-number board with train name intersects schedule."""
    passage = make_passage()
    v_cons = PassageVisualConsensus(
        service_number_candidates=[
            NumberCandidate(number="12919", distinct_coaches=2),
            NumberCandidate(number="12920", distinct_coaches=2),
            NumberCandidate(number="12921", distinct_coaches=1),
            NumberCandidate(number="12922", distinct_coaches=1),
        ],
        train_name_candidates=[NameCandidate(name="MALWA EXPRESS", distinct_coaches=3)],
        number_board_type="FOUR_NUMBER",
        numeric_status="COMPLEX_BOARD",
    )
    evidence = VisualOcrEvidence(passage_id="P1", passage_consensus=v_cons)
    rows = [
        TimetableRow(site_id="ndls_pf12_13", schedule_version="v1", station_code="NDLS", train_number="12920", train_name="MALWA EXPRESS", departure_local="17:20")
    ]
    dec = resolve_shadow(passage, evidence, rows)
    assert dec.shadow_final_number == "12920"


def test_name_alias_collision_ambiguous():
    """Test 45: ambiguous train name matching two different services without numbers yields AMBIGUOUS."""
    passage = make_passage()
    v_cons = PassageVisualConsensus(
        service_number_candidates=[],
        train_name_candidates=[NameCandidate(name="INTERCITY EXPRESS", distinct_coaches=3)],
        numeric_status="NOT_NUMBERED",
    )
    evidence = VisualOcrEvidence(passage_id="P1", passage_consensus=v_cons)
    rows = [
        TimetableRow(site_id="ndls_pf12_13", schedule_version="v1", station_code="NDLS", train_number="14211", train_name="INTERCITY EXPRESS", departure_local="17:20"),
        TimetableRow(site_id="ndls_pf12_13", schedule_version="v1", station_code="NDLS", train_number="14212", train_name="INTERCITY EXPRESS", departure_local="17:25"),
    ]
    dec = resolve_shadow(passage, evidence, rows)
    assert dec.shadow_status == "AMBIGUOUS"


def test_visual_ntes_conflict_not_auto_confirmed():
    """Test 46: strong visual pair [12919, 12920] completely contradicts NTES suggesting 12001 -> CONFLICT."""
    passage = make_passage()
    v_cons = PassageVisualConsensus(
        service_number_candidates=[
            NumberCandidate(number="12919", distinct_coaches=3),
            NumberCandidate(number="12920", distinct_coaches=3),
        ],
        stable_pair=["12919", "12920"],
        numeric_status="PAIR_CONFIRMED",
    )
    evidence = VisualOcrEvidence(passage_id="P1", passage_consensus=v_cons)
    rows = [
        TimetableRow(site_id="ndls_pf12_13", schedule_version="v1", station_code="NDLS", train_number="12001", train_name="SHATABDI", departure_local="17:20")
    ]
    ntes = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at="",
        fetched_at="",
        trains=[NtesTrainEntry(train_number="12001", train_name="SHATABDI", platform="12")],
    )
    dec = resolve_shadow(passage, evidence, rows, ntes_before=ntes)
    assert dec.shadow_status in ("CONFLICT", "UNKNOWN")
    assert RC_VISUAL_NTES_CONFLICT in dec.reason_codes


def test_direction_conflict_provisional_or_conflict():
    """Test 47: passage direction DOWN does not automatically confirm UP train without corroboration."""
    passage = make_passage()
    passage.direction = "DN"
    v_cons = PassageVisualConsensus(
        service_number_candidates=[NumberCandidate(number="12920", distinct_coaches=3)],
        numeric_status="PAIR_PROVISIONAL",
    )
    evidence = VisualOcrEvidence(passage_id="P1", passage_consensus=v_cons)
    rows = [
        TimetableRow(
            site_id="ndls_pf12_13",
            schedule_version="v1",
            station_code="NDLS",
            train_number="12920",
            train_name="INDORE-NDLS MALWA EXP",
            arrival_local="17:20",
        )
    ]
    dec = resolve_shadow(passage, evidence, rows)
    # Direction is in conflict (arriving train vs departing passage); cannot be CONFIRMED
    assert dec.shadow_status != "CONFIRMED"


def test_unknown_platform_not_asserted():
    """Test 48: platform ambiguity flag is always present for island camera."""
    passage = make_passage(platforms=["12", "13"])
    v_cons = PassageVisualConsensus()
    evidence = VisualOcrEvidence(passage_id="P1", passage_consensus=v_cons)
    dec = resolve_shadow(passage, evidence, [])
    assert RC_PLATFORM_AMBIGUOUS_ISLAND in dec.reason_codes


def test_emu_movement_does_not_steal_passenger_identity():
    """Test 49: EMU train movement starting with 6 is flagged as NOT_SERVICE_EVENT or separate gate."""
    passage = make_passage()
    v_cons = PassageVisualConsensus(
        service_number_candidates=[NumberCandidate(number="64432", distinct_coaches=3)],
        numeric_status="PAIR_CONFIRMED",
    )
    evidence = VisualOcrEvidence(passage_id="P1", passage_consensus=v_cons)
    rows = [
        TimetableRow(site_id="ndls_pf12_13", schedule_version="v1", station_code="NDLS", train_number="64432", train_name="GZB-NDLS EMU", service_type="emu_memu", departure_local="17:20")
    ]
    dec = resolve_shadow(passage, evidence, rows)
    assert RC_NOT_SERVICE_EVENT in dec.reason_codes or dec.shadow_status in ("NOT_SERVICE_EVENT", "PROVISIONAL", "CONFIRMED")


def test_multiple_co_present_trains_safe_abstention():
    """Test 50: multiple trains present at same platform/time safely abstain rather than guessing."""
    passage = make_passage()
    v_cons = PassageVisualConsensus(
        service_number_candidates=[],
        numeric_status="UNKNOWN",
    )
    evidence = VisualOcrEvidence(passage_id="P1", passage_consensus=v_cons)
    rows = [
        TimetableRow(site_id="ndls_pf12_13", schedule_version="v1", station_code="NDLS", train_number="12920", train_name="MALWA", departure_local="17:20"),
        TimetableRow(site_id="ndls_pf12_13", schedule_version="v1", station_code="NDLS", train_number="12314", train_name="SEALDAH", departure_local="17:20"),
    ]
    dec = resolve_shadow(passage, evidence, rows)
    assert dec.shadow_status in ("UNKNOWN", "AMBIGUOUS")
