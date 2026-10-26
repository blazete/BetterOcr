"""
Tests for Balanced Two-Pool Selection Algorithm:
Pool A: up to 15 OCR crop calls for high-quality board proposals with temporal diversity (max 2 per 2s bucket).
Pool B: up to 10 OCR crop calls for highest-quality unused crops with spatial/temporal deduplication.
Total inference limit: exactly 25 maximum calls, never exceeded.
"""

import pytest
from shadow_identity.shadow_ocr.contracts import SampledFrame
from shadow_identity.shadow_ocr.two_stage_sampler import (
    BoardRoiProposal,
    SelectedOcrCrop,
    TwoStageFrameSampler,
)


def test_balanced_two_pool_budgets_and_deduplication():
    # Create 20 frames across 10 temporal buckets (0 to 18 seconds, 2s each)
    frames = []
    proposals = []

    for b_idx in range(10):
        t_base = b_idx * 2000
        # Frame 1 in bucket
        f1_idx = b_idx * 10
        frames.append(SampledFrame(
            frame_index=f1_idx,
            timestamp_ms=t_base,
            coach_id=None,
            coach_id_source="UNVERIFIED",
            sharpness_laplacian=120.0 + b_idx,
            brightness_mean=120.0,
            contrast_std=40.0,
            is_valid_quality=True,
            selection_reason="screening",
        ))
        # Add 3 proposals on f1 (exceeding bucket cap of 2)
        proposals.append(BoardRoiProposal(
            frame_index=f1_idx,
            timestamp_ms=t_base,
            bbox_xywh=(100, 100, 200, 80),
            aspect_ratio=2.5,
            area=16000,
            confidence_score=0.90,
            expanded_crop_xyxy=(80, 80, 320, 200),
        ))
        # Spatially duplicate proposal on f1
        proposals.append(BoardRoiProposal(
            frame_index=f1_idx,
            timestamp_ms=t_base,
            bbox_xywh=(105, 102, 195, 78),
            aspect_ratio=2.5,
            area=15210,
            confidence_score=0.88,
            expanded_crop_xyxy=(85, 82, 320, 200),
        ))
        # Non-overlapping second proposal on f1
        proposals.append(BoardRoiProposal(
            frame_index=f1_idx,
            timestamp_ms=t_base,
            bbox_xywh=(400, 100, 200, 80),
            aspect_ratio=2.5,
            area=16000,
            confidence_score=0.75,
            expanded_crop_xyxy=(380, 80, 620, 200),
        ))

        # Frame 2 in bucket (no proposals, sharp fallback candidate)
        f2_idx = f1_idx + 5
        frames.append(SampledFrame(
            frame_index=f2_idx,
            timestamp_ms=t_base + 1000,
            coach_id=None,
            coach_id_source="UNVERIFIED",
            sharpness_laplacian=150.0 + b_idx,
            brightness_mean=120.0,
            contrast_std=40.0,
            is_valid_quality=True,
            selection_reason="fallback",
        ))

    default_crop = (0, 130, 768, 280)
    selected_crops = TwoStageFrameSampler.select_balanced_two_pool_crops(
        screened_frames=frames,
        all_proposals=proposals,
        default_crop_xyxy=default_crop,
        pool_a_budget=15,
        pool_b_budget=10,
        max_total_budget=25,
        bucket_duration_ms=2000,
    )

    pool_a = [c for c in selected_crops if c.pool == "POOL_A"]
    pool_b = [c for c in selected_crops if c.pool == "POOL_B"]

    # 1. Total inference limit: exactly <= 25 calls
    assert len(selected_crops) <= 25
    assert len(pool_a) <= 15
    assert len(pool_b) <= 10
    assert len(selected_crops) == len(pool_a) + len(pool_b)

    # 2. Pool A: Max 2 crops per 2-second bucket
    bucket_counts = {}
    for c in pool_a:
        b_id = int(c.timestamp_ms // 2000)
        bucket_counts[b_id] = bucket_counts.get(b_id, 0) + 1
        assert bucket_counts[b_id] <= 2, f"Bucket {b_id} exceeded max 2 crops in Pool A"

    # 3. Pool B: Deduplication verification
    # Spatially duplicate proposal (85, 82, 320, 200) on f1 must NOT be in selected_crops
    frame_0_crops = [c.crop_xyxy for c in selected_crops if c.frame_index == 0]
    for i in range(len(frame_0_crops)):
        for j in range(i + 1, len(frame_0_crops)):
            assert TwoStageFrameSampler._compute_iou(frame_0_crops[i], frame_0_crops[j]) <= 0.6


def test_balanced_two_pool_strictly_caps_small_budgets():
    """Verify that when a smaller budget is requested (e.g. 5 or 10), neither pool nor total exceeds it."""
    frames = []
    proposals = []
    for i in range(20):
        frames.append(SampledFrame(
            frame_index=i,
            timestamp_ms=i * 500,
            coach_id=None,
            coach_id_source="UNVERIFIED",
            sharpness_laplacian=100.0 + i,
            brightness_mean=120.0,
            contrast_std=40.0,
            is_valid_quality=True,
            selection_reason="screening",
        ))
        proposals.append(BoardRoiProposal(
            frame_index=i,
            timestamp_ms=i * 500,
            bbox_xywh=(100, 100, 200, 80),
            aspect_ratio=2.5,
            area=16000,
            confidence_score=0.85,
            expanded_crop_xyxy=(80, 80, 320, 200),
        ))

    default_crop = (0, 130, 768, 280)

    for budget in [1, 3, 5, 10, 15, 20, 25]:
        crops = TwoStageFrameSampler.select_balanced_two_pool_crops(
            screened_frames=frames,
            all_proposals=proposals,
            default_crop_xyxy=default_crop,
            pool_a_budget=15,
            pool_b_budget=10,
            max_total_budget=budget,
        )
        assert len(crops) <= budget
        assert len([c for c in crops if c.pool == "POOL_A"]) <= 15
        assert len([c for c in crops if c.pool == "POOL_B"]) <= 10
        assert len(crops) <= 25
