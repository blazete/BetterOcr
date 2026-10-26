"""
Phase 1 ShadowOCR Pipeline Orchestrator.
Processes a finalized MP4 passage into raw JSONL OCR observations.
Strictly read-only; zero production writes.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import List, Optional

from shadow_identity.shadow_ocr.board_roi import BoardRoiExtractor
from shadow_identity.shadow_ocr.contracts import (
    BoardRoiConfig,
    PassageVideoInput,
    RawOcrObservation,
)
from shadow_identity.shadow_ocr.frame_sampler import FrameDiversitySampler
from shadow_identity.shadow_ocr.paddle_engine import PaddleEngineWrapper
from shadow_identity.shadow_ocr.preprocess import PreprocessingEngine
from shadow_identity.shadow_ocr.settings import ShadowOcrSettings


class ShadowOcrPhase1Pipeline:
    """
    Orchestrates Phase 1: Completed MP4 -> Diverse Sampled Frames -> P1 Preprocessing -> PaddleOCR -> Raw Observations.
    """

    def __init__(
        self,
        settings: Optional[ShadowOcrSettings] = None,
        engine: Optional[PaddleEngineWrapper] = None,
    ):
        self.settings = settings or ShadowOcrSettings()
        self.engine = engine or PaddleEngineWrapper.get_shared_instance()

    def run_mp4_inference(
        self,
        passage_input: PassageVideoInput,
        output_jsonl_path: Optional[Path] = None,
    ) -> List[RawOcrObservation]:
        """
        Runs Phase 1 inference pipeline on a completed, verified MP4 file.
        """
        from shadow_identity.shadow_ocr.sources.finalized_mp4 import FinalizedMp4Source

        start_time = time.time()
        source = FinalizedMp4Source(passage_input)
        valid, msg = source.validate()
        if not valid:
            raise ValueError(f"MP4 validation failed: {msg}")

        # Compute file hash for provenance
        passage_input.compute_and_verify_hash()

        # Resolve camera ROI
        roi_xyxy = passage_input.board_roi_xyxy or self.settings.default_board_rois.get(
            passage_input.source_camera_id, (0, 130, 768, 280)
        )
        roi_config = BoardRoiConfig(
            camera_id=passage_input.source_camera_id,
            roi_xyxy=roi_xyxy,
            resolution_wh=self.settings.default_resolution,
            description="Phase 1 camera board ROI",
        )
        roi_extractor = BoardRoiExtractor(roi_config)

        # Sample time-diverse frames
        sampler = FrameDiversitySampler(
            sampling_fps=self.settings.sampling_fps,
            max_frames=self.settings.max_frames_per_passage,
            min_laplacian=self.settings.min_sharpness_laplacian,
        )
        sampled_frames = sampler.sample_passage_frames(
            source=source,
            roi_extractor=roi_extractor,
            coach_intervals=passage_input.coach_intervals,
        )

        observations: List[RawOcrObservation] = []

        with source:
            for sf in sampled_frames:
                frame = source.read_frame_at(sf.frame_index)
                if frame is None:
                    continue

                crop, (rx1, ry1, rx2, ry2) = roi_extractor.extract_crop(frame)

                # Primary Preprocessing: P1 Color 2x Bicubic
                preprocessed = PreprocessingEngine.apply_p1_color_2x_bicubic(crop)

                # PaddleOCR Detection + Recognition
                detections = self.engine.predict(preprocessed)

                for poly, text, conf in detections:
                    if conf < self.settings.min_ocr_confidence:
                        continue

                    # Classify raw token
                    clean_text = text.strip()
                    entity_type = "UNKNOWN"
                    if len(clean_text) == 5 and clean_text.isdigit():
                        entity_type = "SERVICE_NUMBER"
                    elif any(c.isalpha() for c in clean_text) and len(clean_text) >= 3:
                        entity_type = "TRAIN_NAME"

                    # Full frame bounding box calculation (mapping 2x preprocessed crop back to frame)
                    full_poly = []
                    if poly:
                        for pt in poly:
                            px, py = pt[0] / 2.0, pt[1] / 2.0
                            full_poly.append([int(px + rx1), int(py + ry1)])

                    obs = RawOcrObservation(
                        schema_version="shadow_ocr.v1",
                        site_id=passage_input.site_id,
                        passage_id=passage_input.passage_id,
                        camera_id=passage_input.source_camera_id,
                        frame_id=sf.frame_index,
                        timestamp_ms=sf.timestamp_ms,
                        board_roi_xyxy=[rx1, ry1, rx2, ry2],
                        coach_id=sf.coach_id,
                        coach_id_source=sf.coach_id_source,
                        ocr_model_id=self.engine.model_id,
                        preprocessing_variant="P1_COLOR_2X_BICUBIC",
                        raw_text=clean_text,
                        normalized_text=clean_text.upper(),
                        entity_type=entity_type,
                        ocr_confidence=conf,
                        bbox_in_crop=[int(p) for pt in poly for p in pt] if poly else [],
                        bbox_full_frame=[int(p) for pt in full_poly for p in pt] if full_poly else [],
                    )
                    observations.append(obs)

        # Write to JSONL if path requested
        if output_jsonl_path:
            output_jsonl_path = Path(output_jsonl_path)
            output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_jsonl_path, "w", encoding="utf-8") as f:
                for obs in observations:
                    f.write(json.dumps(obs.to_dict()) + "\n")

        return observations
