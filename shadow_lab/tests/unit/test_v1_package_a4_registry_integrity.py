"""
Regression tests for BetterOcr / ShadowOCR V1 Package A4:
Registry Mapping Integrity.

Specification Rules:
1. Normalize recognized name using Phase 7C rules.
2. Retrieve registry matches deterministically.
3. Require a unique approved alias match before assigning a canonical name.
4. If multiple canonical names match, preserve ambiguity.
5. Load service-number mappings only from explicitly approved records with provenance.
6. Detect contradictory service-number mappings and fail closed for the affected mapping.
7. Treat a global verified mapping as service-family knowledge only.
8. Treat site schedule separately; never infer station applicability from registry membership.
9. Reject stale or unvalidated schedule rows as active corroboration.
10. Never choose an active service solely by registry iteration order, alphabetical order, or first-member preference.

Tests:
1. Unique alias match -> canonical name, NAME_MATCH, SERVICE_FAMILY_MAPPING.
2. Competing aliases -> ambiguity preserved, no arbitrary canonical assignment.
3. Conflicting mappings -> fail closed, affected mapping unverified.
4. Absent site schedule -> registry match gives service-family candidates only, no schedule/station match asserted.
5. Missing provenance -> treated as unverified candidate, fails verified-mapping gate.
"""

import pytest
from shadow_identity.models import (
    NameCandidate,
    PassageInput,
    PassageVisualConsensus,
    TimetableRow,
    VisualOcrEvidence,
)
from shadow_identity.reason_codes import (
    RC_AMBIGUOUS_MULTI_CANDIDATE,
    RC_NAME_MATCH,
    RC_SCHEDULE_UNAVAILABLE,
    RC_SERVICE_FAMILY_MAPPING,
    RC_SERVICE_FAMILY_MATCHED,
    RC_SERVICE_MAPPING_UNVERIFIED,
)
from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.shadow_ocr.text_entities import (
    EntityClassifier,
    RegisteredTrainName,
    TRAIN_NAME_REGISTRY,
)


def _make_passage():
    return PassageInput(
        site_id="ndls_pf12_13",
        passage_id="P_REG_01",
        camera_id="side_a",
        passage_start="2026-10-14T17:15:00+05:30",
        passage_end="2026-10-14T17:15:00+05:30",
        possible_platforms=["12", "13"],
    )


def test_a4_unique_alias_match():
    """1. Unique alias match -> canonical name assigned with provenance, NAME_MATCH, SERVICE_FAMILY_MAPPING."""
    passage = _make_passage()
    v_cons = PassageVisualConsensus(
        train_name_candidates=[NameCandidate(name="UTTAR SAMPARK KRAN", distinct_coaches=2)],
        numeric_status="NOT_NUMBERED",
        recognized_train_name="UTTAR SAMPARK KRAN",
    )
    evidence = VisualOcrEvidence(passage_id="P_REG_01", passage_consensus=v_cons)
    dec = resolve_shadow(passage, evidence, [])

    assert dec.recognized_train_name == "UTTAR SAMPARK KRAN"
    assert dec.service_family_candidates == ["12445", "12446"]
    assert RC_NAME_MATCH in dec.reason_codes
    assert RC_SERVICE_FAMILY_MAPPING in dec.reason_codes


def test_a4_competing_aliases_preserve_ambiguity():
    """2. Competing aliases (multiple canonical names match) -> preserve ambiguity."""
    # Classify a query that matches multiple entries ambiguously
    norm, candidates, reason = EntityClassifier._match_train_name_against_registry("SAMPARK KRANTI")
    # Both UTTAR SAMPARK KRANTI and BIHAR SAMPARK KRANTI match "SAMPARK KRANTI"
    assert norm is None
    assert len(candidates) >= 2


def test_a4_conflicting_mappings_fail_closed():
    """3. Conflicting mappings -> fail closed for affected mapping."""
    passage = _make_passage()
    v_cons = PassageVisualConsensus(
        train_name_candidates=[NameCandidate(name="TEST EXPRESS", distinct_coaches=2)],
        numeric_status="NOT_NUMBERED",
        recognized_train_name="TEST EXPRESS",
    )
    evidence = VisualOcrEvidence(passage_id="P_REG_01", passage_consensus=v_cons)

    # Provide site aliases with conflicting / contradictory service numbers
    conflicting_aliases = {
        "aliases": {
            "TEST EXPRESS 1": {
                "canonical_name": "TEST EXPRESS",
                "service_numbers": ["11111", "11112"],
                "verification_state": "VERIFIED",
                "provenance": "source_a",
            },
            "TEST EXPRESS 2": {
                "canonical_name": "TEST EXPRESS",
                "service_numbers": ["99991", "99992"],
                "verification_state": "VERIFIED",
                "provenance": "source_b",
            },
        }
    }
    dec = resolve_shadow(passage, evidence, [], site_aliases=conflicting_aliases)
    # Contradictory service mappings must fail closed
    assert dec.service_family_candidates == []
    assert RC_SERVICE_MAPPING_UNVERIFIED in dec.reason_codes


def test_a4_absent_site_schedule_family_only():
    """4. Absent site schedule -> registry match provides family knowledge only, never station schedule."""
    passage = _make_passage()
    v_cons = PassageVisualConsensus(
        train_name_candidates=[NameCandidate(name="MALWA EXP", distinct_coaches=2)],
        numeric_status="NOT_NUMBERED",
        recognized_train_name="MALWA EXP",
    )
    evidence = VisualOcrEvidence(passage_id="P_REG_01", passage_consensus=v_cons)
    # Empty site schedule rows
    dec = resolve_shadow(passage, evidence, [])

    assert dec.service_family_candidates == ["12919", "12920"]
    assert RC_SERVICE_FAMILY_MATCHED in dec.reason_codes
    assert RC_SCHEDULE_UNAVAILABLE in dec.reason_codes
    # Active service number must NOT be asserted from global registry alone
    assert dec.active_service_number is None
    assert dec.active_direction_verified is False


def test_a4_missing_provenance_fails_verified_gate():
    """5. Missing provenance -> treated as unverified candidate, fails verified gate."""
    passage = _make_passage()
    v_cons = PassageVisualConsensus(
        train_name_candidates=[NameCandidate(name="UNVERIFIED TRAIN", distinct_coaches=2)],
        numeric_status="NOT_NUMBERED",
        recognized_train_name="UNVERIFIED TRAIN",
    )
    evidence = VisualOcrEvidence(passage_id="P_REG_01", passage_consensus=v_cons)

    unverified_aliases = {
        "aliases": {
            "UNVERIFIED TRAIN": {
                "canonical_name": "UNVERIFIED TRAIN",
                "service_numbers": ["33333", "33334"],
                "verification_state": "UNVERIFIED",  # Lacks verified provenance
                "provenance": None,
            }
        }
    }
    dec = resolve_shadow(passage, evidence, [], site_aliases=unverified_aliases)
    assert dec.service_family_candidates == []
    assert RC_SERVICE_MAPPING_UNVERIFIED in dec.reason_codes
