"""
Exact preprocessing variants for ShadowOCR.
Implements P1 (Color 2x Bicubic primary) with deterministic mathematical fidelity.
"""

from __future__ import annotations

import cv2
import numpy as np


class PreprocessingEngine:
    """
    Applies image transforms for OCR text recognition.
    """

    @staticmethod
    def apply_p1_color_2x_bicubic(bgr_crop: np.ndarray) -> np.ndarray:
        """
        P1 Primary: 2x Bicubic interpolation on original BGR color image.
        Reproduces the primary winning variant from final validation batch.
        """
        h, w = bgr_crop.shape[:2]
        return cv2.resize(bgr_crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def apply_p2_historical_gamma_lut(bgr_crop: np.ndarray, gamma: float = 0.7) -> np.ndarray:
        """
        P2 Reproducibility: Historical gamma LUT with gamma=0.7 and invGamma=1/0.7 (~1.43).
        Note: Per mathematical specification, invGamma > 1 DARKENs midtones.
        Applied to BGR image, followed by 2x bicubic upscaling.
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        darkened = cv2.LUT(bgr_crop, table)
        h, w = darkened.shape[:2]
        return cv2.resize(darkened, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def apply_p3_clahe_2x_bicubic(bgr_crop: np.ndarray) -> np.ndarray:
        """
        P3 Fallback: Grayscale -> CLAHE (clipLimit=2.5, tileGridSize=(8,8)) -> 2x Bicubic -> BGR.
        """
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY) if len(bgr_crop.shape) == 3 else bgr_crop
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        h, w = cl.shape[:2]
        up = cv2.resize(cl, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
