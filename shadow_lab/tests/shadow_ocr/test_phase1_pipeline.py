"""
Unit and regression tests for Phase 1 ShadowOCR pipeline.
Tests MP4 ingestion, safety bounds, frame diversity, preprocessors, and OCR contracts.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from shadow_identity.shadow_ocr.board_roi import BoardRoiExtractor
from shadow_identity.shadow_ocr.contracts import (
    BoardRoiConfig,
    PassageVideoInput,
    RawOcrObservation,
)
from shadow_identity.shadow_ocr.frame_sampler import FrameDiversitySampler
from shadow_identity.shadow_ocr.preprocess import PreprocessingEngine
from shadow_identity.shadow_ocr.settings import ShadowOcrSettings
from shadow_identity.shadow_ocr.sources.finalized_mp4 import FinalizedMp4Source


def test_board_roi_bounds_validation():
    # Valid config
    cfg = BoardRoiConfig("cam1", (0, 100, 768, 300), (768, 576), "valid")
    assert cfg.validate_bounds() is True

    # Invalid config (out of bounds x2 > width)
    cfg_invalid = BoardRoiConfig("cam1", (0, 100, 800, 300), (768, 576), "invalid")
    assert cfg_invalid.validate_bounds() is False

    # Invalid config (y1 >= y2)
    cfg_invalid2 = BoardRoiConfig("cam1", (0, 300, 768, 100), (768, 576), "inverted")
    assert cfg_invalid2.validate_bounds() is False


def test_board_roi_extraction():
    cfg = BoardRoiConfig("cam1", (10, 20, 110, 120), (200, 200), "test")
    extractor = BoardRoiExtractor(cfg)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[20:120, 10:110] = 255

    crop, bbox = extractor.extract_crop(frame)
    assert crop.shape == (100, 100, 3)
    assert bbox == (10, 20, 110, 120)
    assert np.all(crop == 255)


def test_preprocess_p1_2x_bicubic():
    img = np.ones((50, 100, 3), dtype=np.uint8) * 128
    out = PreprocessingEngine.apply_p1_color_2x_bicubic(img)
    assert out.shape == (100, 200, 3)


def test_preprocess_p2_historical_gamma_darkening():
    # Verify that gamma=0.7 with inv_gamma=1/0.7 darkens midtones
    gray_val = 128
    img = np.ones((10, 10, 3), dtype=np.uint8) * gray_val
    out = PreprocessingEngine.apply_p2_historical_gamma_lut(img, gamma=0.7)
    # 128/255 = 0.50196 -> (0.50196)^(1/0.7) = 0.373 -> * 255 ~= 95
    assert out.shape == (20, 20, 3)
    assert out[0, 0, 0] < gray_val
    assert abs(int(out[0, 0, 0]) - 95) <= 2


def test_preprocess_p3_clahe_bgr():
    img = np.zeros((40, 80, 3), dtype=np.uint8)
    img[10:30, 20:60] = 180
    out = PreprocessingEngine.apply_p3_clahe_2x_bicubic(img)
    assert out.shape == (80, 160, 3)


def test_finalized_mp4_source_missing_file():
    p_input = PassageVideoInput(
        site_id="test_site",
        installation_id="inst_1",
        passage_id="pass_01",
        video_path="/non/existent/path.mp4",
    )
    source = FinalizedMp4Source(p_input)
    valid, msg = source.validate()
    assert valid is False
    assert "does not exist" in msg


def test_finalized_mp4_source_real_video_smoke():
    # Test on coach_id11.mp4 if present in workspace
    real_video = Path("/Users/shreyanshmalviya/Desktop/BetterOcr/train-identity-lab/videos/coach_id11.mp4")
    if not real_video.is_file():
        pytest.skip("coach_id11.mp4 not present in workspace")

    p_input = PassageVideoInput(
        site_id="ndls_pf12_13",
        installation_id="ndls_01",
        passage_id="coach_id11_test",
        video_path=str(real_video),
    )
    source = FinalizedMp4Source(p_input)
    valid, msg = source.validate()
    assert valid is True
    assert p_input.frame_count == 3717
    assert p_input.fps == 25.0
    assert p_input.duration_sec > 140.0

    # Read frame 100
    with source:
        f = source.read_frame_at(100)
        assert f is not None
        assert f.shape == (576, 768, 3)


def test_frame_diversity_sampler():
    real_video = Path("/Users/shreyanshmalviya/Desktop/BetterOcr/train-identity-lab/videos/coach_id11.mp4")
    if not real_video.is_file():
        pytest.skip("coach_id11.mp4 not present in workspace")

    p_input = PassageVideoInput(
        site_id="ndls_pf12_13",
        installation_id="ndls_01",
        passage_id="coach_id11_test",
        video_path=str(real_video),
    )
    source = FinalizedMp4Source(p_input)
    assert source.validate()[0] is True

    roi_cfg = BoardRoiConfig("side_a", (0, 130, 768, 280), (768, 576))
    roi_ext = BoardRoiExtractor(roi_cfg)

    sampler = FrameDiversitySampler(sampling_fps=0.2, max_frames=5)
    sampled = sampler.sample_passage_frames(source, roi_ext)

    assert len(sampled) > 0
    assert len(sampled) <= 5
    # Ensure frames are separated in time (diverse across video)
    indices = [s.frame_index for s in sampled]
    assert indices == sorted(indices)
    assert indices[-1] > indices[0]
