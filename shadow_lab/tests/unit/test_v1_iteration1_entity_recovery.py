"""
Regression tests for V1 Optimization Loop Iteration 1:
Experiment D — Entity and name recovery.

Targeted Defect:
EntityClassifier previously rejected canonical train names possessing one informative
token + generic token (e.g. 'TAJ EXPRESS', 'LUCKNOW MAIL', 'MALWA EXPRESS') with
rejection_reason='LESS_THAN_TWO_INFORMATIVE_TOKENS', violating the requirement:
'Require at least two informative name tokens except for explicitly approved unique aliases.'
"""

import pytest
from shadow_identity.shadow_ocr.text_entities import EntityClassifier
from shadow_identity.shadow_ocr.consensus import PhysicalCoachConsensusEngine
from shadow_identity.shadow_ocr.contracts import RawOcrObservation
from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.models import PassageInput


class TestIteration1EntityRecovery:
    """Test approved unique aliases with generic suffixes are accepted while enforcing negatives."""

    @pytest.mark.parametrize(
        "raw_text,expected_canonical",
        [
            ("TAJ EXPRESS", "TAJ EXPRESS"),
            ("TAJ EXP", "TAJ EXPRESS"),
            ("LUCKNOW MAIL", "LUCKNOW MAIL"),
            ("SACHKHAND EXPRESS", "SACHKHAND EXPRESS"),
            ("MALWA EXPRESS", "MALWA EXPRESS"),
            ("GOA EXPRESS", "GOA EXPRESS"),
            ("POORVA EXPRESS", "POORVA EXPRESS"),
            ("GOMTI EXPRESS", "GOMTI EXPRESS"),
            ("UTTAR SAMPARK KRAN", "UTTAR SAMPARK KRANTI"),
            ("SHRI SHAKTI AC EXPRESS", "SHRI SHAKTI EXPRESS"),
        ],
    )
    def test_approved_aliases_accepted(self, raw_text: str, expected_canonical: str):
        """Explicitly approved aliases in registry must be classified as TRAIN_NAME."""
        entity = EntityClassifier.classify(raw_text, ocr_confidence=0.92)
        assert entity.entity_type == "TRAIN_NAME", (
            f"Expected TRAIN_NAME for '{raw_text}', got {entity.entity_type} (rej: {entity.rejection_reason})"
        )
        assert entity.canonical_name == expected_canonical
        assert entity.rejection_reason is None

    @pytest.mark.parametrize(
        "negative_text,expected_reason_substr",
        [
            ("TAJ", "INSUFFICIENT"),
            ("MALWA", "INSUFFICIENT"),
            ("EXPRESS", "INSUFFICIENT"),
            ("MAIL", "INSUFFICIENT"),
            ("UTTAR", "INSUFFICIENT"),
            ("SAMPARK", "INSUFFICIENT"),
            ("RAJ", "INSUFFICIENT"),
            ("EMERGENCY WINDOW", "SEATING_SAFETY_SIGN"),
            ("TO SEAT 1-90", "SEATING_SAFETY_SIGN"),
            ("1-90", "SEATING_SAFETY_SIGN"),
            ("GENERAL", "COACH_CLASS_DESIGNATION"),
            ("LUGGAGE", "COACH_CLASS_DESIGNATION"),
            ("SECOND CLASS", "COACH_CLASS_DESIGNATION"),
            ("AC THREE TIER", "COACH_CLASS_DESIGNATION"),
        ],
    )
    def test_negative_controls_strictly_rejected(self, negative_text: str, expected_reason_substr: str):
        """Single non-approved tokens and coach stencils must never produce TRAIN_NAME."""
        entity = EntityClassifier.classify(negative_text, ocr_confidence=0.95)
        assert entity.entity_type != "TRAIN_NAME"
        assert entity.canonical_name is None
        assert expected_reason_substr in (entity.rejection_reason or "")

    def test_coach_serial_distinction(self):
        """6-digit coach serials and slash serials must never be promoted to service numbers."""
        e1 = EntityClassifier.classify("243560", 0.99)
        assert e1.entity_type == "COACH_SERIAL"
        assert e1.entity_type != "SERVICE_NUMBER"

        e2 = EntityClassifier.classify("217152/C", 0.99)
        assert e2.entity_type == "COACH_SERIAL"
        assert e2.entity_type != "SERVICE_NUMBER"

    def test_consensus_name_only_preservation_without_active_promotion(self):
        """Consensus engine preserves train name and leaves active service number unassigned."""
        engine = PhysicalCoachConsensusEngine()
        obs = [
            RawOcrObservation(
                site_id="ndls_pf12_13",
                passage_id="test_pas",
                camera_id="cam_01",
                coach_id="C01",
                coach_id_source="VERIFIED_BOUNDARY",
                frame_id=100,
                raw_text="TAJ EXPRESS",
                ocr_confidence=0.95,
            )
        ]
        res = engine.build_passage_identity("ndls_pf12_13", "test_pas", "cam_01", "dummy_hash", obs)
        assert len(res.train_name_candidates) == 1
        assert res.train_name_candidates[0].name == "TAJ EXPRESS"
        assert res.stable_pair == []
        assert res.service_number_candidates == []
        assert res.dominant_service_number is None
