"""
Regression test reproducing verified-family gating requirement:
Raw OCR numbers (such as 12920/12225 or 13307/13007) must NOT automatically
be presented as verified family pairs in service_family_candidates.
They must be preserved separately as visual OCR observations while
service_family_candidates remains empty and identity_status is UNKNOWN.
"""

from shadow_identity.evidence_fusion import decide_identity_v1
from shadow_identity.models import (
    NameCandidate,
    NumberCandidate,
    PassageInput,
    PassageVisualConsensus,
    VisualOcrEvidence,
)


def test_raw_unregistered_ocr_numbers_not_presented_as_verified_family():
    """Bug reproduction: Raw OCR numbers (12920/12225) must not become service_family_candidates."""
    vis = VisualOcrEvidence(
        passage_id="pas_test_v008",
        passage_consensus=PassageVisualConsensus(
            service_number_candidates=[
                NumberCandidate(number="12920", distinct_coaches=2),
                NumberCandidate(number="12225", distinct_coaches=2),
            ],
            stable_pair=["12920", "12225"],
            numeric_status="PAIR_PROVISIONAL",
            train_name_candidates=[],
            recognized_train_name=None,
            service_family_candidates=[],
        ),
    )
    passage = PassageInput(
        site_id="ndls_pf12_13",
        passage_id="pas_test_v008",
        direction="unknown",
        direction_evidence=None,
    )

    dec = decide_identity_v1(passage=passage, visual_evidence=vis, site_schedule_rows=[])

    # 1. service_family_candidates MUST be empty because 12920/12225 is NOT a registered family
    assert dec.service_family_candidates == [], (
        f"Expected empty service_family_candidates, got {dec.service_family_candidates}"
    )

    # 2. Raw OCR observations must be preserved separately
    assert "12920" in dec.visual_service_numbers
    assert "12225" in dec.visual_service_numbers

    # 3. Without active direction and without verified family, status must be UNKNOWN
    assert dec.identity_status == "UNKNOWN"
    assert dec.active_service_number is None


def test_raw_unregistered_disparate_trains_not_presented_as_verified_family():
    """Bug reproduction: Raw OCR numbers (13307/13007) must not become service_family_candidates."""
    vis = VisualOcrEvidence(
        passage_id="pas_test_v010",
        passage_consensus=PassageVisualConsensus(
            service_number_candidates=[
                NumberCandidate(number="13307", distinct_coaches=2),
                NumberCandidate(number="13007", distinct_coaches=2),
            ],
            stable_pair=["13307", "13007"],
            numeric_status="PAIR_PROVISIONAL",
            train_name_candidates=[],
            recognized_train_name=None,
            service_family_candidates=[],
        ),
    )
    passage = PassageInput(
        site_id="ndls_pf12_13",
        passage_id="pas_test_v010",
        direction="unknown",
        direction_evidence=None,
    )

    dec = decide_identity_v1(passage=passage, visual_evidence=vis, site_schedule_rows=[])

    assert dec.service_family_candidates == []
    assert "13307" in dec.visual_service_numbers
    assert "13007" in dec.visual_service_numbers
    assert dec.identity_status == "UNKNOWN"
    assert dec.active_service_number is None
