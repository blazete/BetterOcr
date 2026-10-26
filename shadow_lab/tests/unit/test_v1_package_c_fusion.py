"""
Regression tests for BetterOcr / ShadowOCR V1 Package C:
Identity Evidence Fusion.

Requirements:
C1 — Evidence Classes and Provenance (10 categories, typed evidence contract).
C2 — Candidate Generation (does not select active train number).
C3 — Active Service Decision Gating (VERIFIED_ACTIVE, FAMILY_CANDIDATE, UNKNOWN, CONFLICT).
C4 — Atomic Output Contract (schema_version = "shadow_identity_v1", backward-compatible).
"""

from datetime import datetime, timezone
import json
import pytest

from shadow_identity.models import (
    DirectionEvidence,
    PassageInput,
    VisualObservation,
    NumberCandidate,
    NameCandidate,
    PassageVisualConsensus,
    VisualOcrEvidence,
    TimetableRow,
)
from shadow_identity.evidence_fusion import (
    EVIDENCE_CATEGORIES,
    IdentityEvidenceRecord,
    decide_identity_v1,
    format_v1_identity_output,
)


def test_c1_evidence_categories_and_record():
    """C1: 10 distinct evidence categories and typed provenance contract."""
    expected_categories = {
        "VISUAL_SERVICE_NUMBER",
        "VISUAL_TRAIN_NAME",
        "VISUAL_ENDPOINT",
        "VALIDATED_SERVICE_REGISTRY",
        "VALIDATED_SITE_TIMETABLE",
        "AUTHENTIC_LIVE_TRAIN_FEED",
        "PHYSICAL_PASSAGE_TIMESTAMP",
        "VERIFIED_PLATFORM",
        "VERIFIED_TRACK_DIRECTION",
        "VERIFIED_PHYSICAL_COACH",
    }
    assert set(EVIDENCE_CATEGORIES) == expected_categories

    record = IdentityEvidenceRecord(
        evidence_type="VISUAL_TRAIN_NAME",
        original_source="paddleocr_frame_120",
        observed_value="PRAYAGRAJ EXP",
        source_passage_id="pas_001",
        observation_time_utc="2026-10-26T10:00:00Z",
        validation_result=True,
        verification_provenance="site_aliases.yaml:PRAYAGRAJ_EXPRESS",
        contradictions=[],
        can_establish_service_family=True,
        can_establish_active_identity=False,
    )
    assert record.evidence_type == "VISUAL_TRAIN_NAME"
    assert record.can_establish_service_family is True
    assert record.can_establish_active_identity is False


def test_c2_candidate_generation_does_not_select_active_service():
    """C2: Candidate generation extracts evidence and builds families without picking active service."""
    vis = VisualOcrEvidence(
        passage_id="pas_test",
        passage_consensus=PassageVisualConsensus(
            service_number_candidates=[
                NumberCandidate(number="12417", distinct_coaches=3),
                NumberCandidate(number="12418", distinct_coaches=3),
            ],
            stable_pair=["12417", "12418"],
            numeric_status="PAIR_CONFIRMED",
            train_name_candidates=[NameCandidate(name="PRAYAGRAJ EXP", distinct_coaches=2)],
            recognized_train_name="PRAYAGRAJ EXP",
            service_family_candidates=["12417", "12418"],
        ),
    )
    passage = PassageInput(
        site_id="ndls_pf12_13",
        passage_id="pas_test",
        direction="unknown",
        direction_evidence=None,
    )
    # Without verified direction, active service MUST NOT be selected
    dec = decide_identity_v1(passage=passage, visual_evidence=vis, site_schedule_rows=[])
    assert dec.identity_status == "FAMILY_CANDIDATE"
    assert dec.active_service_number is None
    assert dec.active_direction_verified is False
    assert set(dec.service_family_candidates) == {"12417", "12418"}


def test_c3_verified_active_decision_requires_verified_direction():
    """C3: VERIFIED_ACTIVE requires authentic, fresh, passage-linked independent direction evidence."""
    vis = VisualOcrEvidence(
        passage_id="pas_test",
        passage_consensus=PassageVisualConsensus(
            service_number_candidates=[],
            train_name_candidates=[NameCandidate(name="SHIV GANGA EXP", distinct_coaches=3)],
            numeric_status="NOT_NUMBERED",
            recognized_train_name="SHIV GANGA EXP",
            service_family_candidates=["12559", "12560"],
        ),
    )
    dir_ev = DirectionEvidence(
        direction="DN",
        authentic=True,
        fresh=True,
        passage_link_verified=True,
        source_authorized=True,
        direction_matches_candidate=True,
        source="AXLE_COUNTER_INDEPENDENT",
        provenance="hardware_sensor_ch1",
    )
    passage = PassageInput(
        site_id="ndls_pf12_13",
        passage_id="pas_test",
        passage_start="2026-10-14T17:15:00+05:30",
        passage_end="2026-10-14T17:18:00+05:30",
        possible_platforms=["12", "13"],
        direction="DN",
        direction_evidence=dir_ev,
    )
    tt_rows = [
        TimetableRow(
            site_id="ndls_pf12_13",
            schedule_version="1.0",
            station_code="NDLS",
            train_number="12560",
            train_name="SHIV GANGA EXP",
            platform="12",
            departure_local="17:20",
            operating_days=[0, 1, 2, 3, 4, 5, 6],
            service_type="passenger",
        ),
        TimetableRow(
            site_id="ndls_pf12_13",
            schedule_version="1.0",
            station_code="NDLS",
            train_number="12559",
            train_name="SHIV GANGA EXP",
            platform="12",
            arrival_local="08:30",
            operating_days=[0, 1, 2, 3, 4, 5, 6],
            service_type="passenger",
        ),
    ]
    dec = decide_identity_v1(
        passage=passage,
        visual_evidence=vis,
        site_schedule_rows=tt_rows,
        direction_evidence=dir_ev,
    )
    assert dec.identity_status == "VERIFIED_ACTIVE"
    assert dec.active_service_number == "12560"
    assert dec.active_direction_verified is True


def test_c3_conflict_decision_when_sources_contradict():
    """C3: When sources contradict, decision must be CONFLICT and active service None."""
    vis = VisualOcrEvidence(
        passage_id="pas_test",
        passage_consensus=PassageVisualConsensus(
            service_number_candidates=[
                NumberCandidate(number="12417", distinct_coaches=3),
                NumberCandidate(number="12418", distinct_coaches=3),
            ],
            stable_pair=["12417", "12418"],
            numeric_status="PAIR_CONFIRMED",
            train_name_candidates=[NameCandidate(name="PRAYAGRAJ EXP", distinct_coaches=2)],
            recognized_train_name="PRAYAGRAJ EXP",
            service_family_candidates=["12417", "12418"],
        ),
    )
    # Contradictory directional evidence
    dir_ev = DirectionEvidence(
        direction="UP",
        authentic=True,
        fresh=True,
        passage_link_verified=True,
        source_authorized=True,
        direction_matches_candidate=False,
        has_conflict=True,
        source="AXLE_COUNTER_INDEPENDENT",
    )
    passage = PassageInput(
        site_id="ndls_pf12_13",
        passage_id="pas_test",
        direction="UP",
        direction_evidence=dir_ev,
    )
    dec = decide_identity_v1(
        passage=passage,
        visual_evidence=vis,
        site_schedule_rows=[],
        direction_evidence=dir_ev,
    )
    assert dec.identity_status == "CONFLICT"
    assert dec.active_service_number is None


def test_c4_output_contract_schema():
    """C4: Output dictionary matches schema_version = 'shadow_identity_v1' with required fields."""
    output = format_v1_identity_output(
        site_id="ndls_pf12_13",
        passage_id="pas_001",
        input_video_sha256="abc123fullsha256",
        recognized_train_names=["PRAYAGRAJ EXP"],
        visual_service_numbers=["12417", "12418"],
        recognized_endpoints=["NDLS", "PRYJ"],
        service_family_candidates=["12417", "12418"],
        active_service_number=None,
        active_direction_verified=False,
        identity_status="FAMILY_CANDIDATE",
        reason_codes=["RC_NAME_MATCH", "RC_SERVICE_FAMILY_MAPPING"],
        evidence_provenance=[
            {
                "evidence_type": "VISUAL_TRAIN_NAME",
                "original_source": "frame_50",
                "observed_value": "PRAYAGRAJ EXP",
                "provenance": "site_aliases.yaml",
            }
        ],
        contradictions=[],
        processing_metrics={"duration_sec": 1.25, "ocr_calls": 18},
    )

    # Required top-level keys
    assert output["schema_version"] == "shadow_identity_v1"
    assert output["passage_id"] == "pas_001"
    assert output["site_id"] == "ndls_pf12_13"
    assert output["input_video_sha256"] == "abc123fullsha256"
    assert output["recognized_train_names"] == ["PRAYAGRAJ EXP"]
    assert output["visual_service_numbers"] == ["12417", "12418"]
    assert output["recognized_endpoints"] == ["NDLS", "PRYJ"]
    assert output["service_family_candidates"] == ["12417", "12418"]
    assert output["active_service_number"] is None
    assert output["active_direction_verified"] is False
    assert output["identity_status"] == "FAMILY_CANDIDATE"
    assert "RC_NAME_MATCH" in output["reason_codes"]
    assert len(output["evidence_provenance"]) == 1
    assert output["contradictions"] == []
    assert output["processing_metrics"]["ocr_calls"] == 18
    # decision_created_utc must be a valid UTC timestamp
    assert output["decision_created_utc"] is not None
    datetime.fromisoformat(output["decision_created_utc"].replace("Z", "+00:00"))
