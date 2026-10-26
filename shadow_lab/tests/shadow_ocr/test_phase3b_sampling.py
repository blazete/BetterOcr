"""
Unit and regression tests for Phase 3B:
- Two-stage screening vs OCR inference budget separation
- Automatic board detector proposals and geometry checks
- KATRA classified as ENDPOINT, not train name
- BUFFET CAR / AC THREE TIER classified as COACH_DESIGNATION, not service number
- Four-digit coach stencils classified as COACH_SERIAL, not service number
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from shadow_identity.shadow_ocr.contracts import PassageVideoInput
from shadow_identity.shadow_ocr.sources.finalized_mp4 import FinalizedMp4Source
from shadow_identity.shadow_ocr.text_entities import EntityClassifier
from shadow_identity.shadow_ocr.two_stage_sampler import (
    AutomaticBoardDetector,
    TwoStageFrameSampler,
)


def test_automatic_board_detector_synthetic():
    detector = AutomaticBoardDetector(strip_y1=100, strip_y2=300)
    # Synthetic frame with a yellow rectangular board
    frame = np.zeros((576, 768, 3), dtype=np.uint8)
    # Draw yellow board in upper half: B=0, G=220, R=220 (HSV ~ H:30, S:255, V:220)
    frame[150:200, 300:500] = [0, 220, 220]

    props = detector.propose_boards(frame, frame_index=10, timestamp_ms=400)
    assert len(props) >= 1
    p0 = props[0]
    assert p0.bbox_xywh[0] == 300
    assert p0.bbox_xywh[1] == 150
    assert p0.bbox_xywh[2] == 200
    assert p0.bbox_xywh[3] == 50
    assert p0.aspect_ratio == 4.0
    # Check expanded crop margin
    ex1, ey1, ex2, ey2 = p0.expanded_crop_xyxy
    assert ex1 < 300
    assert ey1 < 150
    assert ex2 > 500
    assert ey2 > 200


def test_entity_safeguards_katra_endpoint_not_train_name():
    c = EntityClassifier.classify("KATRA", 0.95)
    assert c.entity_type == "ENDPOINT"
    assert c.entity_type != "TRAIN_NAME"

    c2 = EntityClassifier.classify("SHRI MATA VAISHNO DEVI KATRA", 0.98)
    assert c2.entity_type == "ENDPOINT"
    assert c2.entity_type != "TRAIN_NAME"


def test_entity_safeguards_buffet_car_and_classes():
    c_buffet = EntityClassifier.classify("BUFFET CAR", 0.99)
    assert c_buffet.entity_type == "COACH_DESIGNATION"
    assert c_buffet.entity_type != "SERVICE_NUMBER"

    c_ac3 = EntityClassifier.classify("AC THREE TIER", 0.95)
    assert c_ac3.entity_type == "COACH_DESIGNATION"
    assert c_ac3.entity_type != "SERVICE_NUMBER"


def test_entity_safeguards_four_digit_stencils_not_service_number():
    c_stencil = EntityClassifier.classify("2249", 0.99)
    # Must NOT be classified as a 5-digit service number!
    assert c_stencil.entity_type != "SERVICE_NUMBER"

    c_stencil2 = EntityClassifier.classify("1630", 0.98)
    assert c_stencil2.entity_type != "SERVICE_NUMBER"


def test_two_stage_sampler_real_video():
    vpath = Path("train-identity-lab/videos/coach_id11.mp4")
    if not vpath.is_file():
        pytest.skip("coach_id11.mp4 not present")

    p_in = PassageVideoInput("ndls", "inst1", "p11", str(vpath))
    source = FinalizedMp4Source(p_in)
    source.validate()

    sampler = TwoStageFrameSampler(screening_fps=2.0, max_ocr_budget=10)
    selected_frames, proposals = sampler.screen_passage(source)

    assert len(selected_frames) <= 10
    # Must have proposals found in Video 11
    assert len(proposals) > 0
    # Selected frames must prioritize frames with board proposals
    board_frame_indices = {p.frame_index for p in proposals}
    has_board_frame = any(sf.frame_index in board_frame_indices for sf in selected_frames)
    assert has_board_frame is True


def test_katra_observation_regression():
    """
    Phase 3D regression test: Ensures known KATRA observation on coach_id11.mp4
    frame 2250 under focused board crop is never silently lost.
    """
    import cv2
    from shadow_identity.shadow_ocr.paddle_engine import PaddleEngineWrapper
    from shadow_identity.shadow_ocr.preprocess import PreprocessingEngine

    vpath = Path("train-identity-lab/videos/coach_id11.mp4")
    if not vpath.is_file():
        pytest.skip("coach_id11.mp4 not present")

    cap = cv2.VideoCapture(str(vpath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 2250)
    ret, frame = cap.read()
    cap.release()
    assert ret is True, "Failed to decode frame 2250"

    # Focused board crop around destination board
    crop = frame[140:270, 200:768]
    p1 = PreprocessingEngine.apply_p1_color_2x_bicubic(crop)
    engine = PaddleEngineWrapper.get_shared_instance()
    res = engine.predict(p1)

    texts = [t for poly, t, c in res]
    assert any("KATRA" in t for t in texts), f"KATRA regression failed! Detections were: {texts}"

