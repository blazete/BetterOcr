"""
OCR Adapter for Shadow Identity Lab
Ingests historical per-frame OCR CSVs from train-identity-lab or runs
inference, converting detections into canonical VisualOcrEvidence models.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

from shadow_identity.coach_consensus import CoachConsensusAggregator
from shadow_identity.models import VisualObservation, VisualOcrEvidence
from shadow_identity.text_normalization import is_negative_class


class OcrAdapter:
    def __init__(self, confirmed_threshold: int = 3):
        self.aggregator = CoachConsensusAggregator(confirmed_threshold=confirmed_threshold)

    def load_from_per_frame_csv(
        self, csv_path: Path, passage_id: str, default_entity_type: str = "SERVICE_NUMBER"
    ) -> VisualOcrEvidence:
        """
        Load historical per-frame OCR CSV (e.g. from results/video_00X_per_frame_ocr.csv)
        into a canonical VisualOcrEvidence object.
        """
        if not csv_path.is_file():
            raise FileNotFoundError(f"Per-frame OCR CSV not found: {csv_path}")

        observations: List[VisualObservation] = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                cid = r.get("coach_id", "C01")
                fid = int(r.get("frame_id", 0))
                ts_sec = float(r.get("timestamp", 0.0))
                raw_txt = r.get("raw_text", "")
                norm_txt = r.get("normalized_text", raw_txt)
                conf = float(r.get("ocr_confidence", 0.0))
                c5d = r.get("candidate_5digit", "")

                # Classify entity type
                if is_negative_class(raw_txt):
                    etype = "NEGATIVE_STENCIL"
                elif c5d:
                    etype = "SERVICE_NUMBER"
                elif any(c.isalpha() for c in raw_txt) and len(raw_txt) > 3:
                    etype = "TRAIN_NAME"
                else:
                    etype = "SERVICE_NUMBER" if any(c.isdigit() for c in raw_txt) else "UNKNOWN"

                obs = VisualObservation(
                    coach_segment_id=cid,
                    frame_id=fid,
                    timestamp_offset_ms=int(ts_sec * 1000),
                    raw_text=raw_txt,
                    normalized_text=norm_txt,
                    entity_type=etype,
                    bbox=[],
                    ocr_confidence=conf,
                )
                observations.append(obs)

        consensus = self.aggregator.aggregate_observations(observations)

        return VisualOcrEvidence(
            passage_id=passage_id,
            ocr_model_version="paddleocr_ppocrv6_frozen",
            preprocessing_profile="2x_bicubic",
            per_observation=observations,
            passage_consensus=consensus,
        )
