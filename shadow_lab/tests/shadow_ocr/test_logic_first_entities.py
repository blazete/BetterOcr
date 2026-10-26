"""
Regression tests for Package P0.1: Extract ALL observed service numbers without hallucinating.
"""
import pytest
from shadow_identity.shadow_ocr.text_entities import EntityClassifier, ClassifiedEntity
from shadow_identity.shadow_ocr.contracts import RawOcrObservation
from shadow_identity.shadow_ocr.consensus import PhysicalCoachConsensusEngine


def test_classify_all_multi_number_pairs():
    # 1. '18103/18104' returns two numbers
    res1 = EntityClassifier.classify_all("18103/18104", 0.99)
    nums1 = [e.normalized_text for e in res1 if e.entity_type == "SERVICE_NUMBER"]
    assert nums1 == ["18103", "18104"], f"Expected ['18103', '18104'], got {nums1}"

    # 2. '12477 / 12478' returns two
    res2 = EntityClassifier.classify_all("12477 / 12478", 0.95)
    nums2 = [e.normalized_text for e in res2 if e.entity_type == "SERVICE_NUMBER"]
    assert nums2 == ["12477", "12478"], f"Expected ['12477', '12478'], got {nums2}"

    # 3. '13307-13308' returns two
    res3 = EntityClassifier.classify_all("13307-13308", 0.92)
    nums3 = [e.normalized_text for e in res3 if e.entity_type == "SERVICE_NUMBER"]
    assert nums3 == ["13307", "13308"], f"Expected ['13307', '13308'], got {nums3}"


def test_classify_all_single_number_with_leading_zero():
    # '02276' returns one with leading zero preserved
    res = EntityClassifier.classify_all("02276", 0.91)
    nums = [e.normalized_text for e in res if e.entity_type == "SERVICE_NUMBER"]
    assert nums == ["02276"], f"Expected ['02276'], got {nums}"


def test_classify_all_negative_serials_and_short_numbers():
    # '153008/C' returns no service number
    res_serial = EntityClassifier.classify_all("153008/C", 0.95)
    nums_serial = [e.normalized_text for e in res_serial if e.entity_type == "SERVICE_NUMBER"]
    assert nums_serial == [], f"Expected no service number for coach serial, got {nums_serial}"

    # '2249' returns no service number (4 digits is not a 5-digit service number)
    res_short = EntityClassifier.classify_all("2249", 0.88)
    nums_short = [e.normalized_text for e in res_short if e.entity_type == "SERVICE_NUMBER"]
    assert nums_short == [], f"Expected no service number for 4-digit token, got {nums_short}"

    # Token mixing letters and digits: e.g. '12915ON' should not autocorrect to 5-digit service number
    res_mixed = EntityClassifier.classify_all("12915ON", 0.85)
    nums_mixed = [e.normalized_text for e in res_mixed if e.entity_type == "SERVICE_NUMBER"]
    assert nums_mixed == [], f"Expected no service number for mixed token, got {nums_mixed}"


def test_repeated_pair_sightings_on_one_coach_give_one_coach_vote():
    # Multiple pair observations on the same coach should only yield 1 coach vote per number
    engine = PhysicalCoachConsensusEngine()

    raw_text = "18103/18104"
    entities = EntityClassifier.classify_all(raw_text, 0.95)

    observations = []
    # Two frames on Coach C01, both seeing the pair
    for frame_id in [100, 110]:
        for ent in entities:
            obs = RawOcrObservation(
                schema_version="shadow_ocr.v1",
                site_id="ndls_pf12_13",
                passage_id="test_passage",
                camera_id="cam_01",
                frame_id=frame_id,
                timestamp_ms=frame_id * 40,
                board_roi_xyxy=[100, 100, 200, 150],
                coach_id="C01",
                coach_id_source="VERIFIED_BOUNDARY",
                ocr_model_id="paddle_v4",
                preprocessing_variant="P1_COLOR_2X_BICUBIC",
                raw_text=raw_text,
                normalized_text=ent.normalized_text,
                entity_type=ent.entity_type,
                ocr_confidence=0.95,
                source_video_sha256="test_sha",
            )
            observations.append(obs)

    res = engine.build_passage_identity(
        site_id="ndls_pf12_13",
        passage_id="test_passage",
        camera_id="cam_01",
        video_sha256="test_sha",
        observations=observations,
    )

    for cand in res.service_number_candidates:
        assert cand.distinct_coaches == 1, (
            f"Candidate {cand.number} should have 1 distinct coach vote, got {cand.distinct_coaches}"
        )
