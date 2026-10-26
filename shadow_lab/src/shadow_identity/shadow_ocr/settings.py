"""
Settings and resource thresholds for ShadowOCR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class ShadowOcrSettings:
    enabled: bool = False
    allow_network: bool = False
    max_frames_per_passage: int = 40
    max_frames_per_coach: int = 3
    sampling_fps: float = 2.0
    min_sharpness_laplacian: float = 80.0
    min_ocr_confidence: float = 0.50
    default_resolution: Tuple[int, int] = (768, 576)
    # Default side camera board ROI (x1, y1, x2, y2)
    default_board_rois: Dict[str, Tuple[int, int, int, int]] = field(
        default_factory=lambda: {
            "side_a": (0, 130, 768, 280),
            "side_b": (0, 130, 768, 280),
        }
    )
    # Resource budgets
    max_passage_duration_sec: float = 300.0
    ocr_timeout_sec: float = 120.0
