"""
Regression tests for BetterOcr / ShadowOCR V1 Package A3:
Evidence-Strength Reason Codes.

Specification Rules:
Separate:
- NAME_RECOGNIZED_SINGLE_OBSERVATION
- NAME_REPEATED_VERIFIED_COACHES
- NAME_RECOGNIZED_COACH_UNVERIFIED

Rules:
- One OCR observation is evidence of recognized text, not independent consensus.
- Multiple frames from the same coach do not count as multiple physical coaches.
- Distinct coaches must come from verified physical coach intervals or authenticated hardware passage events.
- Do not infer unique coach count by arbitrarily dividing video time.
- If coach identity is unverified, preserve the text but avoid verified multi-coach claims.
- Preserve the existing numeric pair confirmation requirement: at least 3 independently verified physical coaches.
- Never alter historical raw observations.
"""

import pytest
from shadow_identity.models import (
    NameCandidate,
    NumberCandidate,
    PassageInput,
    PassageVisualConsensus,
    VisualOcrEvidence,
)
from shadow_identity.reason_codes import (
    RC_NAME_RECOGNIZED_COACH_UNVERIFIED,
    RC_NAME_RECOGNIZED_SINGLE_OBSERVATION,
    RC_NAME_REPEATED_VERIFIED_COACHES,
    RC_VISUAL_NAME_STRONG,
    RC_VISUAL_NAME_WEAK,
    RC_VISUAL_PAIR_CONFIRMED,
    RC_VISUAL_PAIR_PROVISIONAL,
)
from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.shadow_ocr.consensus import PhysicalCoachConsensusEngine
from shadow_identity.shadow_ocr.contracts import RawOcrObservation


def test_a3_single_observation_tagged_single_observation():
    """Single OCR observation on 1 coach is not multi-coach consensus."""
    obs = [
        RawOcrObservation(
            frame_id=10,
            timestamp_ms=1000,
            raw_text="UTTAR SAMPARK KRAN",
            ocr_confidence=0.95,
            bbox_in_crop=[100, 100, 200, 50],
            coach_id="coach_01",
            coach_id_source="VERIFIED_BOUNDARY",
        )
    ]
    engine = PhysicalCoachConsensusEngine()
    res = engine.build_passage_identity(
        site_id="ndls_pf12_13",
        passage_id="P01",
        camera_id="side_a",
        video_sha256="sha256",
        observations=obs,
    )
    assert res.train_name_candidates[0].distinct_coaches == 1
    assert RC_NAME_RECOGNIZED_SINGLE_OBSERVATION in res.reason_codes
    assert RC_NAME_REPEATED_VERIFIED_COACHES not in res.reason_codes


def test_a3_multiple_frames_same_coach_not_counted_as_multiple_coaches():
    """Multiple frames from the same coach must NOT count as multiple physical coaches."""
    obs = [
        RawOcrObservation(
            frame_id=10,
            timestamp_ms=1000,
            raw_text="UTTAR SAMPARK KRAN",
            ocr_confidence=0.95,
            bbox_in_crop=[100, 100, 200, 50],
            coach_id="coach_01",
            coach_id_source="VERIFIED_BOUNDARY",
        ),
        RawOcrObservation(
            frame_id=15,
            timestamp_ms=1500,
            raw_text="UTTAR SAMPARK KRAN",
            ocr_confidence=0.96,
            bbox_in_crop=[110, 100, 210, 50],
            coach_id="coach_01",
            coach_id_source="VERIFIED_BOUNDARY",
        ),
        RawOcrObservation(
            frame_id=20,
            timestamp_ms=2000,
            raw_text="UTTAR SAMPARK",
            ocr_confidence=0.94,
            bbox_in_crop=[120, 100, 220, 50],
            coach_id="coach_01",
            coach_id_source="VERIFIED_BOUNDARY",
        ),
    ]
    engine = PhysicalCoachConsensusEngine()
    res = engine.build_passage_identity("ndls_pf12_13", "P01", "side_a", "sha256", obs)
    # Must only be 1 distinct coach
    assert res.train_name_candidates[0].distinct_coaches == 1
    assert res.train_name_candidates[0].total_observations == 3
    assert RC_NAME_RECOGNIZED_SINGLE_OBSERVATION in res.reason_codes
    assert RC_NAME_REPEATED_VERIFIED_COACHES not in res.reason_codes


def test_a3_unverified_coach_boundaries_preserves_text_without_multicoach_claim():
    """If coach boundaries are unverified, preserve text but tag coach unverified."""
    obs = [
        RawOcrObservation(
            frame_id=10,
            timestamp_ms=1000,
            raw_text="UTTAR SAMPARK KRAN",
            ocr_confidence=0.95,
            bbox_in_crop=[100, 100, 200, 50],
            coach_id=None,
            coach_id_source=None,
        ),
        RawOcrObservation(
            frame_id=50,
            timestamp_ms=5000,
            raw_text="UTTAR SAMPARK KRAN",
            ocr_confidence=0.95,
            bbox_in_crop=[100, 100, 200, 50],
            coach_id=None,
            coach_id_source=None,
        ),
    ]
    engine = PhysicalCoachConsensusEngine()
    res = engine.build_passage_identity("ndls_pf12_13", "P01", "side_a", "sha256", obs)
    assert len(res.train_name_candidates) == 1
    assert res.train_name_candidates[0].distinct_coaches == 1
    assert RC_NAME_RECOGNIZED_COACH_UNVERIFIED in res.reason_codes
    assert RC_NAME_REPEATED_VERIFIED_COACHES not in res.reason_codes


def test_a3_repeated_verified_coaches_tagged():
    """Two verified distinct coaches supporting train name receive repeated verified coaches code."""
    obs = [
        RawOcrObservation(
            frame_id=10,
            timestamp_ms=1000,
            raw_text="UTTAR SAMPARK KRAN",
            ocr_confidence=0.95,
            bbox_in_crop=[100, 100, 200, 50],
            coach_id="coach_01",
            coach_id_source="VERIFIED_BOUNDARY",
        ),
        RawOcrObservation(
            frame_id=60,
            timestamp_ms=6000,
            raw_text="UTTAR SAMPARK KRANTI",
            ocr_confidence=0.96,
            bbox_in_crop=[100, 100, 200, 50],
            coach_id="coach_02",
            coach_id_source="VERIFIED_BOUNDARY",
        ),
    ]
    engine = PhysicalCoachConsensusEngine()
    res = engine.build_passage_identity("ndls_pf12_13", "P01", "side_a", "sha256", obs)
    assert res.train_name_candidates[0].distinct_coaches == 2
    assert RC_NAME_REPEATED_VERIFIED_COACHES in res.reason_codes


def test_a3_resolver_reason_codes_differentiate():
    """Resolver accurately assigns evidence-strength codes based on consensus."""
    # 1. Unverified coach
    passage = PassageInput(site_id="ndls_pf12_13", passage_id="P1")
    v_cons_unverified = PassageVisualConsensus(
        train_name_candidates=[NameCandidate(name="TAJ EXPRESS", distinct_coaches=1, total_observations=2)],
        numeric_status="NOT_NUMBERED",
        recognized_train_name="TAJ EXPRESS",
        service_family_candidates=["12279", "12280"],
    )
    ev_unverified = VisualOcrEvidence(passage_id="P1", passage_consensus=v_cons_unverified)
    dec_unverified = resolve_shadow(passage, ev_unverified, [])
    assert RC_NAME_RECOGNIZED_COACH_UNVERIFIED in dec_unverified.reason_codes
    assert RC_VISUAL_NAME_STRONG not in dec_unverified.reason_codes

    # 2. Single verified coach
    passage_verified = PassageInput(site_id="ndls_pf12_13", passage_id="P2")
    setattr(passage_verified, "coach_intervals_verified", True)
    v_cons_single = PassageVisualConsensus(
        train_name_candidates=[NameCandidate(name="TAJ EXPRESS", distinct_coaches=1, total_observations=2)],
        numeric_status="NOT_NUMBERED",
        recognized_train_name="TAJ EXPRESS",
        service_family_candidates=["12279", "12280"],
    )
    ev_single = VisualOcrEvidence(passage_id="P2", passage_consensus=v_cons_single)
    dec_single = resolve_shadow(passage_verified, ev_single, [])
    assert RC_NAME_RECOGNIZED_SINGLE_OBSERVATION in dec_single.reason_codes
    assert RC_VISUAL_NAME_STRONG not in dec_single.reason_codes

    # 3. Multi-coach verified
    v_cons_multi = PassageVisualConsensus(
        train_name_candidates=[NameCandidate(name="TAJ EXPRESS", distinct_coaches=2, total_observations=2)],
        numeric_status="NOT_NUMBERED",
        recognized_train_name="TAJ EXPRESS",
        service_family_candidates=["12279", "12280"],
    )
    ev_multi = VisualOcrEvidence(passage_id="P2", passage_consensus=v_cons_multi)
    dec_multi = resolve_shadow(passage_verified, ev_multi, [])
    assert RC_NAME_REPEATED_VERIFIED_COACHES in dec_multi.reason_codes
    assert RC_VISUAL_NAME_STRONG in dec_multi.reason_codes
