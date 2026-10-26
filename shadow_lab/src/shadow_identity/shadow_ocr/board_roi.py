"""
Board ROI extraction and camera boundary verification.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from shadow_identity.shadow_ocr.contracts import BoardRoiConfig


class BoardRoiExtractor:
    """
    Extracts board region-of-interest from a full camera frame.
    Validates boundary coordinates and handles out-of-bound requests safely.
    """

    def __init__(self, config: BoardRoiConfig):
        self.config = config
        if not self.config.validate_bounds():
            raise ValueError(
                f"Invalid Board ROI bounds {config.roi_xyxy} for resolution {config.resolution_wh}"
            )

    def extract_crop(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.config.roi_xyxy

        # Clamp safely to actual frame dimensions
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))

        crop = frame[y1:y2, x1:x2].copy()
        return crop, (x1, y1, x2, y2)
