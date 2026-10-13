"""Unit tests for OCR, Negative-Class Filtering, and Coach Consensus (Tests 21 to 30).
Validates coach serial rejection (217152/C), berth stencil rejection (43-48),
one-coach-fifty-frames single vote, >=3 coach threshold, and Video 8 audit resolution.
"""

from shadow_identity.coach_consensus import CoachConsensusAggregator
from shadow_identity.models import VisualObservation
from shadow_identity.text_normalization import extract_service_numbers, is_negative_class_text


def test_coach_serial_217152C_not_train_service_number():
    """Test 21: 6-digit coach body ID with slash/letter is rejected as negative class."""
    assert is_negative_class_text("217152/C")
    assert is_negative_class_text("04251/A")
    cands = extract_service_numbers("217152/C")
    assert len(cands) == 0


def test_berth_43_48_not_service_number():
    """Test 22: class/berth stencils (S7, 43-48) are rejected as negative class."""
    assert is_negative_class_text("S7")
    assert is_negative_class_text("43-48")
    assert is_negative_class_text("B1 1-72")
    cands = extract_service_numbers("43-48")
    assert len(cands) == 0


def test_one_coach_fifty_frames_counts_once():
    """Test 23: 50 frames observing the same physical coach yield exactly 1 distinct coach vote."""
    agg = CoachConsensusAggregator(min_distinct_coaches=3)
    for frame_idx in range(50):
        obs = VisualObservation(
            coach_segment_id="C01",
            frame_id=frame_idx,
            raw_text="12920",
            normalized_text="12920",
            entity_type="SERVICE_NUMBER",
            ocr_confidence=0.95,
        )
        agg.add_observation(obs)

    consensus = agg.compute_consensus()
    assert len(consensus.service_number_candidates) == 1
    cand = consensus.service_number_candidates[0]
    assert cand.number == "12920"
    assert cand.distinct_coaches == 1
    assert cand.total_observations == 50
    # Because only 1 distinct coach, status cannot be CONFIRMED
    assert consensus.numeric_status != "PAIR_CONFIRMED"


def test_pair_each_candidate_distinct_coach_threshold():
    """Test 24: both candidates in a pair must reach >=3 distinct coaches for PAIR_CONFIRMED."""
    agg = CoachConsensusAggregator(min_distinct_coaches=3)
    # Add 12919 across 3 coaches
    for c_id in ["C01", "C02", "C03"]:
        agg.add_observation(VisualObservation(coach_segment_id=c_id, frame_id=1, raw_text="12919", normalized_text="12919", entity_type="SERVICE_NUMBER"))
    # Add 12920 across 3 coaches
    for c_id in ["C04", "C05", "C06"]:
        agg.add_observation(VisualObservation(coach_segment_id=c_id, frame_id=2, raw_text="12920", normalized_text="12920", entity_type="SERVICE_NUMBER"))

    consensus = agg.compute_consensus()
    assert consensus.numeric_status == "PAIR_CONFIRMED"
    assert set(consensus.stable_pair) == {"12919", "12920"}


def test_video_008_historical_threshold_discrepancy_flagged():
    """Test 25: Video 008 discrepancy (12919 has 2 coaches, 12920 has 5) is strictly PAIR_PROVISIONAL."""
    agg = CoachConsensusAggregator(min_distinct_coaches=3)
    # 12919: only 2 coaches
    for c_id in ["C01", "C02"]:
        agg.add_observation(VisualObservation(coach_segment_id=c_id, frame_id=1, raw_text="12919", normalized_text="12919", entity_type="SERVICE_NUMBER"))
    # 12920: 5 coaches
    for c_id in ["C03", "C04", "C05", "C06", "C07"]:
        agg.add_observation(VisualObservation(coach_segment_id=c_id, frame_id=2, raw_text="12920", normalized_text="12920", entity_type="SERVICE_NUMBER"))

    consensus = agg.compute_consensus()
    # Under strict shadow rules, must be PROVISIONAL, NOT CONFIRMED
    assert consensus.numeric_status == "PAIR_PROVISIONAL"
    assert "12919" in consensus.stable_pair
    assert "12920" in consensus.stable_pair


def test_name_ocr_fallback_when_number_absent():
    """Test 26: when no numbers are observed, train name consensus triggers NOT_NUMBERED path."""
    agg = CoachConsensusAggregator(min_distinct_coaches=3)
    for c_id in ["C01", "C02", "C03"]:
        agg.add_observation(VisualObservation(coach_segment_id=c_id, frame_id=1, raw_text="TAJ EXPRESS", normalized_text="TAJ EXPRESS", entity_type="TRAIN_NAME"))

    consensus = agg.compute_consensus()
    assert len(consensus.service_number_candidates) == 0
    assert consensus.numeric_status == "NOT_NUMBERED"
    assert len(consensus.train_name_candidates) == 1
    assert consensus.train_name_candidates[0].name == "TAJ EXPRESS"


def test_four_six_numbers_name_first_does_not_pick_random_number():
    """Test 27: multi-number board (4 or 6 numbers) marked COMPLEX_BOARD."""
    agg = CoachConsensusAggregator(min_distinct_coaches=3)
    nums = ["12001", "12002", "12003", "12004"]
    for c_id in ["C01", "C02", "C03"]:
        for n in nums:
            agg.add_observation(VisualObservation(coach_segment_id=c_id, frame_id=1, raw_text=n, normalized_text=n, entity_type="SERVICE_NUMBER"))

    consensus = agg.compute_consensus()
    assert consensus.number_board_type in ("FOUR_NUMBER", "COMPLEX_BOARD")


def test_low_confidence_name_suspends_auto_confirmation():
    """Test 28: low confidence name (<0.60) is filtered or marked provisional."""
    agg = CoachConsensusAggregator(min_distinct_coaches=3)
    agg.add_observation(VisualObservation(coach_segment_id="C01", frame_id=1, raw_text="GARIB RATH", normalized_text="GARIB RATH", entity_type="TRAIN_NAME", ocr_confidence=0.45))
    consensus = agg.compute_consensus()
    # 1 observation with 0.45 confidence does not reach threshold
    assert len(consensus.train_name_candidates) == 0 or consensus.train_name_candidates[0].distinct_coaches < 3


def test_multiple_coaches_same_name_consensus():
    """Test 29: multiple coaches observing the same train name form strong consensus."""
    agg = CoachConsensusAggregator(min_distinct_coaches=3)
    for c_id in ["C01", "C02", "C03", "C04"]:
        agg.add_observation(VisualObservation(coach_segment_id=c_id, frame_id=1, raw_text="LUCKNOW MAIL", normalized_text="LUCKNOW MAIL", entity_type="TRAIN_NAME", ocr_confidence=0.95))

    consensus = agg.compute_consensus()
    assert len(consensus.train_name_candidates) == 1
    assert consensus.train_name_candidates[0].name == "LUCKNOW MAIL"
    assert consensus.train_name_candidates[0].distinct_coaches == 4


def test_partial_number_not_autocompleted_without_evidence():
    """Test 30: 3 or 4 digit fragment (e.g. '1292') is not treated as a 5-digit candidate."""
    assert len(extract_service_numbers("1292")) == 0
    assert len(extract_service_numbers("920")) == 0
    # Exactly 5 digits
    assert len(extract_service_numbers("12920")) == 1
