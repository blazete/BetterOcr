"""
Tests for Package P1.1: Real OCR CALL budget plus temporal diversity.
"""
from dataclasses import dataclass
from typing import List
import pytest

from shadow_identity.shadow_ocr.two_stage_sampler import (
    TwoStageFrameSampler,
    BoardRoiProposal,
)
from shadow_identity.shadow_ocr.contracts import SampledFrame


def test_temporal_diversity_round_robin_selection():
    sampler = TwoStageFrameSampler(screening_fps=5.0, max_ocr_budget=6)

    # Simulate 3 temporal buckets (0ms-2000ms, 2000ms-4000ms, 4000ms-6000ms)
    # Bucket 0 has 5 high-sharpness frames with boards
    # Bucket 1 has 2 frames with boards
    # Bucket 2 has 2 frames with boards
    frames: List[SampledFrame] = []
    proposals: List[BoardRoiProposal] = []

    # Bucket 0: ts 0..1600ms (5 frames)
    for i in range(5):
        f_idx = i * 10
        ts = i * 300
        frames.append(SampledFrame(
            frame_index=f_idx, timestamp_ms=ts,
            coach_id=None, coach_id_source="UNVERIFIED",
            sharpness_laplacian=500.0 - i * 10,
            brightness_mean=120.0, contrast_std=40.0,
            is_valid_quality=True, selection_reason="test"
        ))
        proposals.append(BoardRoiProposal(
            frame_index=f_idx, timestamp_ms=ts, bbox_xywh=(100, 100, 200, 50),
            aspect_ratio=4.0, area=10000, confidence_score=0.9,
            expanded_crop_xyxy=(80, 80, 220, 170)
        ))

    # Bucket 1: ts 2200..2500ms (2 frames)
    for i in range(2):
        f_idx = 100 + i * 10
        ts = 2200 + i * 300
        frames.append(SampledFrame(
            frame_index=f_idx, timestamp_ms=ts,
            coach_id=None, coach_id_source="UNVERIFIED",
            sharpness_laplacian=200.0 - i * 10,
            brightness_mean=120.0, contrast_std=40.0,
            is_valid_quality=True, selection_reason="test"
        ))
        proposals.append(BoardRoiProposal(
            frame_index=f_idx, timestamp_ms=ts, bbox_xywh=(100, 100, 200, 50),
            aspect_ratio=4.0, area=10000, confidence_score=0.85,
            expanded_crop_xyxy=(80, 80, 220, 170)
        ))

    # Bucket 2: ts 4200..4500ms (2 frames)
    for i in range(2):
        f_idx = 200 + i * 10
        ts = 4200 + i * 300
        frames.append(SampledFrame(
            frame_index=f_idx, timestamp_ms=ts,
            coach_id=None, coach_id_source="UNVERIFIED",
            sharpness_laplacian=150.0 - i * 10,
            brightness_mean=120.0, contrast_std=40.0,
            is_valid_quality=True, selection_reason="test"
        ))
        proposals.append(BoardRoiProposal(
            frame_index=f_idx, timestamp_ms=ts, bbox_xywh=(100, 100, 200, 50),
            aspect_ratio=4.0, area=10000, confidence_score=0.8,
            expanded_crop_xyxy=(80, 80, 220, 170)
        ))

    # With the old global-topK, all 5 frames from Bucket 0 would be chosen (because laplacian 500..460 > 200)
    # Under P1.1 round-robin temporal buckets:
    # Round 1: chooses 1 best from Bucket 0, 1 best from Bucket 1, 1 best from Bucket 2 (3 total)
    # Round 2: chooses second best from Bucket 0, second best from Bucket 1, second best from Bucket 2 (3 total, hitting budget 6)
    # All 3 buckets MUST be covered!
    selected = sampler.select_temporal_diverse_frames(frames, proposals, budget=6)

    selected_buckets = {f.timestamp_ms // 2000 for f in selected}
    assert selected_buckets == {0, 1, 2}, (
        f"Temporal diversity must cover all 3 buckets {0, 1, 2}, but got {selected_buckets}"
    )
    # Max per bucket capped at 2
    from collections import Counter
    counts = Counter(f.timestamp_ms // 2000 for f in selected)
    for b_id, count in counts.items():
        assert count <= 2, f"Bucket {b_id} exceeded max_per_bucket cap: {count}"
