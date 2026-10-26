"""
Unit and regression tests for Phase 7C: Train-Name Evidence Recovery.
Tests entity classification, consensus preservation, curated registry matching,
and candidate resolver separation of train-family evidence from active identity.
"""

import pytest
from shadow_identity.models import (
    NameCandidate,
    NumberCandidate,
    PassageInput,
    PassageVisualConsensus,
    VisualObservation,
    VisualOcrEvidence,
)
from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.shadow_ocr.text_entities import (
    ClassifiedEntity,
    EntityClassifier,
)
from shadow_identity.shadow_ocr.consensus import PhysicalCoachConsensusEngine
from shadow_identity.shadow_ocr.contracts import RawOcrObservation
from shadow_identity.reason_codes import (
    RC_AMBIGUOUS_MULTI_CANDIDATE,
    RC_NAME_ONLY_ACTIVE_DIRECTION_UNVERIFIED,
    RC_SERVICE_FAMILY_MATCHED,
    RC_SERVICE_MAPPING_UNVERIFIED,
    RC_VISUAL_BOARD_NOT_NUMBERED,
)


class TestTask1TrainNameClassifier:
    """Test 10-rule evidence-preserving name recognition algorithm."""

    def test_positive_uttar_sampark_kran_recovery(self):
        """Input 'UTTAR SAMPARK KRAN' maps to canonical 'UTTAR SAMPARK KRANTI'."""
        raw = "UTTAR SAMPARK KRAN"
        entity = EntityClassifier.classify(raw, ocr_confidence=0.993)

        assert entity.raw_text == raw
        assert entity.normalized_text == "UTTAR SAMPARK KRAN"
        assert entity.entity_type == "TRAIN_NAME"
        assert entity.canonical_name == "UTTAR SAMPARK KRANTI"
        assert entity.rejection_reason is None

    @pytest.mark.parametrize(
        "negative_input,expected_reason",
        [
            ("EMERGENCY WINDOW", "SEATING_SAFETY_SIGN"),
            ("TO SEAT 1-90", "SEATING_SAFETY_SIGN"),
            ("GENERAL", "COACH_CLASS_DESIGNATION"),
            ("LUGGAGE", "COACH_CLASS_DESIGNATION"),
            ("UTTAR", "INSUFFICIENT_NAME_TOKENS"),
            ("SAMPARK", "INSUFFICIENT_NAME_TOKENS"),
            ("RAJ", "INSUFFICIENT_NAME_TOKENS"),
        ],
    )
    def test_negative_controls_rejected(self, negative_input, expected_reason):
        """Negative control strings must not produce train name or service identities."""
        entity = EntityClassifier.classify(negative_input, ocr_confidence=0.995)
        assert entity.entity_type != "TRAIN_NAME"
        assert entity.canonical_name is None
        assert entity.rejection_reason == expected_reason


class TestTask1ConsensusPreservation:
    """Test that consensus receives train-name candidate and assigns no active number."""

    def test_consensus_receives_train_name_without_number(self):
        raw_obs = [
            RawOcrObservation(
                site_id="ndls_pf12_13",
                passage_id="PAS_TEST_12",
                camera_id="side_a",
                coach_id="coach_01",
                coach_id_source="VERIFIED",
                frame_id=500,
                raw_text="UTTAR SAMPARK KRAN",
                normalized_text="UTTAR SAMPARK KRAN",
                ocr_confidence=0.993,
                entity_type="TRAIN_NAME",
            )
        ]
        engine = PhysicalCoachConsensusEngine()
        res = engine.build_passage_identity(
            site_id="ndls_pf12_13",
            passage_id="PAS_TEST_12",
            camera_id="side_a",
            video_sha256="dummy_hash",
            observations=raw_obs,
        )
        assert len(res.train_name_candidates) == 1
        assert res.train_name_candidates[0].name == "UTTAR SAMPARK KRANTI"
        # Must not assign active train number from name alone
        assert res.stable_pair == []
        assert res.service_number_candidates == []


class TestTask2CuratedRegistry:
    """Test registry mappings in aliases.yaml."""

    def test_registry_contains_uttar_sampark_kranti(self):
        from shadow_identity.shadow_ocr.text_entities import TRAIN_NAME_REGISTRY
        assert "UTTAR SAMPARK KRANTI" in TRAIN_NAME_REGISTRY
        entry = TRAIN_NAME_REGISTRY["UTTAR SAMPARK KRANTI"]
        assert "UTTAR SAMPARK KRAN" in entry.aliases
        assert entry.allow_single_token is False


class TestTask3SeparateFamilyEvidenceFromActiveIdentity:
    """Test resolver handling of train name evidence without active direction verification."""

    def test_name_only_multiple_service_family_candidates_ambiguous(self):
        passage = PassageInput(
            passage_id="PAS_TEST_VIDEO_12",
            site_id="ndls_pf12_13",
            passage_start="2026-10-13T21:40:00+05:30",
            passage_end="2026-10-13T21:42:00+05:30",
            possible_platforms=["12", "13"],
        )
        v_cons = PassageVisualConsensus(
            service_number_candidates=[],
            stable_pair=[],
            train_name_candidates=[NameCandidate(name="UTTAR SAMPARK KRANTI", distinct_coaches=1)],
            numeric_status="NOT_NUMBERED",
            recognized_train_name="UTTAR SAMPARK KRANTI",
        )
        evidence = VisualOcrEvidence(
            passage_id="PAS_TEST_VIDEO_12",
            passage_consensus=v_cons,
            recognized_train_name="UTTAR SAMPARK KRANTI",
        )
        site_aliases = {
            "aliases": {
                "UTTAR SAMPARK KRANTI": {
                    "canonical_name": "UTTAR SAMPARK KRANTI",
                    "service_numbers": ["12445", "12446"],
                    "endpoints": ["NEW DELHI", "SHRI MATA VAISHNO DEVI KATRA"],
                }
            }
        }

        # Resolve without schedule or NTES direction confirmation
        decision = resolve_shadow(
            passage=passage,
            visual_evidence=evidence,
            site_schedule_rows=[],
            site_aliases=site_aliases,
            stage="T0",
        )

        # Expected semantics:
        # - Recognized train name preserved
        assert decision.recognized_train_name == "UTTAR SAMPARK KRANTI"
        # - Family candidates populated
        assert decision.service_family_candidates == ["12445", "12446"]
        # - shadow_final_number = null
        assert decision.shadow_final_number is None
        # - active_service_number = null
        assert decision.active_service_number is None
        # - active_direction_verified = false
        assert decision.active_direction_verified is False
        # - Status AMBIGUOUS (multiple verified service candidates remain)
        assert decision.shadow_status == "AMBIGUOUS"
        # - Explicit reason codes
        assert RC_NAME_ONLY_ACTIVE_DIRECTION_UNVERIFIED in decision.reason_codes
        assert RC_AMBIGUOUS_MULTI_CANDIDATE in decision.reason_codes
        assert RC_SERVICE_FAMILY_MATCHED in decision.reason_codes
        assert RC_VISUAL_BOARD_NOT_NUMBERED in decision.reason_codes

    def test_name_only_unverified_service_mapping_unknown(self):
        """If a train name is recognized but has no verified site service mapping, return UNKNOWN."""
        passage = PassageInput(
            passage_id="PAS_TEST_UNKNOWN_MAPPING",
            site_id="ndls_pf12_13",
        )
        v_cons = PassageVisualConsensus(
            service_number_candidates=[],
            stable_pair=[],
            train_name_candidates=[NameCandidate(name="UNREGISTERED EXPRESS", distinct_coaches=1)],
            numeric_status="NOT_NUMBERED",
            recognized_train_name="UNREGISTERED EXPRESS",
        )
        evidence = VisualOcrEvidence(
            passage_id="PAS_TEST_UNKNOWN_MAPPING",
            passage_consensus=v_cons,
            recognized_train_name="UNREGISTERED EXPRESS",
        )

        decision = resolve_shadow(
            passage=passage,
            visual_evidence=evidence,
            site_schedule_rows=[],
            site_aliases={"aliases": {}},
            stage="T0",
        )

        assert decision.recognized_train_name == "UNREGISTERED EXPRESS"
        assert decision.service_family_candidates == []
        assert decision.shadow_final_number is None
        assert decision.active_service_number is None
        assert decision.active_direction_verified is False
        assert decision.shadow_status == "UNKNOWN"
        assert RC_SERVICE_MAPPING_UNVERIFIED in decision.reason_codes
