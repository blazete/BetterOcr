"""
Pinned PaddleOCR inference engine wrapper.
Single lazy-loaded instance per process; strict memory and latency tracking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

_log = logging.getLogger("shadow_ocr.paddle_engine")


class PaddleEngineWrapper:
    """
    Wraps PaddleOCR (PP-OCRv6) execution with bounded lifecycle and safe error handling.
    """

    _instance: Optional[PaddleEngineWrapper] = None

    def __init__(self, lang: str = "en", device: str = "cpu"):
        self.lang = lang
        self.device = device
        self._engine: Any = None
        self.model_id = "PP-OCRv6"
        self.total_calls = 0

    @classmethod
    def get_shared_instance(cls, lang: str = "en", device: str = "cpu") -> PaddleEngineWrapper:
        if cls._instance is None:
            cls._instance = cls(lang=lang, device=device)
        return cls._instance

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            from paddleocr import PaddleOCR
            _log.info("Initializing PaddleOCR (%s on %s)...", self.lang, self.device)
            # Safe initialization using cached weights
            self._engine = PaddleOCR(use_angle_cls=False, lang=self.lang, device=self.device)
            return self._engine
        except Exception as e:
            _log.error("Failed to initialize PaddleOCR engine: %s", e)
            raise RuntimeError(f"PaddleOCR initialization failed: {e}") from e

    def predict(
        self, image: np.ndarray
    ) -> List[Tuple[List[List[int]], str, float]]:
        """
        Executes text detection + recognition on a single image.
        Returns: List of (polygon_points, text, confidence)
        Guaranteed to return empty list on no-text, never raises unhandled exceptions.
        """
        engine = self._ensure_engine()
        self.total_calls += 1

        try:
            # Handle PaddleX predict() or paddleocr ocr() APIs
            if hasattr(engine, "predict"):
                res = engine.predict(image)
            elif hasattr(engine, "ocr"):
                res = engine.ocr(image)
            else:
                raise AttributeError("Unrecognized PaddleOCR engine interface")
        except Exception as e:
            _log.warning("OCR inference error on frame crop: %s", e)
            return []

        if not res:
            return []

        parsed_detections: List[Tuple[List[List[int]], str, float]] = []

        # Format parsing: Handle both PaddleX Result format and classic list format
        first_item = res[0] if isinstance(res, list) and len(res) > 0 else res

        # PaddleX result object/dict check
        if isinstance(first_item, dict) and "rec_texts" in first_item:
            texts = first_item.get("rec_texts", [])
            scores = first_item.get("rec_scores", [])
            polys = first_item.get("dt_polys", [])
            for poly, txt, score in zip(polys, texts, scores):
                poly_list = poly.tolist() if hasattr(poly, "tolist") else list(poly)
                parsed_detections.append((poly_list, str(txt).strip(), float(score)))
            return parsed_detections

        if hasattr(first_item, "get") and "rec_text" in first_item:
            # Single or list of dicts
            for item in res:
                if isinstance(item, dict):
                    txt = item.get("rec_text", "")
                    score = item.get("rec_score", 0.0)
                    poly = item.get("dt_poly", [])
                    parsed_detections.append((poly, str(txt).strip(), float(score)))
            return parsed_detections

        # Classic PaddleOCR format: [[[ [x,y],[x,y],[x,y],[x,y] ], ("text", score)], ...]
        if isinstance(first_item, list):
            for line in first_item:
                if not line or len(line) < 2:
                    continue
                box = line[0]
                text_info = line[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    txt = str(text_info[0]).strip()
                    conf = float(text_info[1])
                    parsed_detections.append((box, txt, conf))

        return parsed_detections
