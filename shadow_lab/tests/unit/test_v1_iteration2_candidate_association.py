"""
Regression tests for V1 Optimization Loop Iteration 2:
Experiment E — Candidate Association & Identity Resolution.

Targeted Defect:
When valid train-name evidence reaches candidate_resolver (e.g. 'SACHKHAND EXPRESS'
or 'MALWA EXPRESS'), candidate association previously failed because the resolver
only checked site_aliases.yaml (which only contained 5 trains), resulting in
empty service_family_candidates and RC_SERVICE_MAPPING_UNVERIFIED despite having
an authentic, verified canonical train name.
"""

import pytest
from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.models import (
    PassageInput,
    PassageVisualConsensus,
    NameCandidate,
    VisualObservation,
    VisualOcrEvidence,
)
from shadow_identity.reason_codes import (
    RC_AMBIGUOUS_MULTI_CANDIDATE,
    RC_NAME_ONLY_ACTIVE_DIRECTION_UNVERIFIED,
    RC_SERVICE_FAMILY_MATCHED,
    RC_SERVICE_MAPPING_UNVERIFIED,
)


def _make_name_only_visual_evidence(train_name: str) -> VisualOcrEvidence:
    consensus = PassageVisualConsensus(
        service_number_candidates=[],
        stable_pair=[],
        train_name_candidates=[
            NameCandidate(name=train_name, distinct_coaches=2, total_observations=2, mean_confidence=0.95)
        ],
        endpoint_candidates=[],
        number_board_type="NOT_NUMBERED",
        numeric_status="NOT_NUMBERED",
        recognized_train_name=train_name,
    )
    return VisualOcrEvidence(
        passage_id="test_pas_assoc",
        passage_consensus=consensus,
    )


class TestIteration2CandidateAssociation:
    """Test candidate association resolves service families for all registered canonical names."""

    @pytest.mark.parametrize(
        "train_name,expected_family",
        [
            ("SACHKHAND EXPRESS", ["12715", "12716"]),
            ("MALWA EXPRESS", ["12919", "12920"]),
            ("UTTAR SAMPARK KRANTI", ["12445", "12446"]),
            ("LUCKNOW MAIL", ["12229", "12230"]),
            ("TAJ EXPRESS", ["12279", "12280"]),
            ("SHIV GANGA EXPRESS", ["12559", "12560"]),
            ("SHRAM SHAKTI EXPRESS", ["12451", "12452"]),
        ],
    )
    def test_registered_train_name_resolves_service_family(self, train_name: str, expected_family: list):
        """Resolver must associate verified train name with correct service family candidates."""
        passage = PassageInput(
            site_id="ndls_pf12_13",
            passage_id="test_pas_assoc",
            passage_start="2026-10-26T12:00:00+05:30",
            passage_end="2026-10-26T12:03:00+05:30",
            observed_platform=None,
            possible_platforms=["12", "13"],
        )
        visual_evidence = _make_name_only_visual_evidence(train_name)

        decision = resolve_shadow(
            passage=passage,
            visual_evidence=visual_evidence,
            site_schedule_rows=[],
        )

        assert decision.recognized_train_name == train_name
        assert sorted(decision.service_family_candidates) == sorted(expected_family), (
            f"Expected family {expected_family} for '{train_name}', got {decision.service_family_candidates}"
        )
        assert decision.active_service_number is None
        assert decision.active_direction_verified is False
        assert decision.shadow_final_number is None
        assert decision.shadow_status == "AMBIGUOUS"
        assert RC_SERVICE_FAMILY_MATCHED in decision.reason_codes
        assert RC_NAME_ONLY_ACTIVE_DIRECTION_UNVERIFIED in decision.reason_codes
        assert RC_SERVICE_MAPPING_UNVERIFIED not in decision.reason_codes
