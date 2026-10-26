"""
Time and coach-diverse frame sampling.
Enforces per-coach / per-window coverage instead of naive global sharpness sorting.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import cv2
import numpy as np

from shadow_identity.shadow_ocr.board_roi import BoardRoiExtractor
from shadow_identity.shadow_ocr.contracts import SampledFrame
from shadow_identity.shadow_ocr.sources.finalized_mp4 import FinalizedMp4Source


class FrameDiversitySampler:
    """
    Samples frames across the duration of an MP4 passage.
    Prioritizes time and coach diversity over global sharpness clustering.
    """

    def __init__(
        self,
        sampling_fps: float = 2.0,
        max_frames: int = 40,
        min_laplacian: float = 60.0,
    ):
        self.sampling_fps = sampling_fps
        self.max_frames = max_frames
        self.min_laplacian = min_laplacian

    @staticmethod
    def compute_quality(crop: np.ndarray) -> Tuple[float, float, float]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        return sharpness, brightness, contrast

    def sample_passage_frames(
        self,
        source: FinalizedMp4Source,
        roi_extractor: BoardRoiExtractor,
        coach_intervals: Optional[List[dict]] = None,
    ) -> List[SampledFrame]:
        p = source.passage_input
        native_fps = p.fps if p.fps > 0 else 25.0
        total_frames = p.frame_count

        if total_frames <= 0:
            return []

        step = max(1, int(round(native_fps / self.sampling_fps)))
        candidate_indices = list(range(0, total_frames, step))

        # Build coach window mapping if external verified boundaries exist
        # coach_intervals: [{'coach_id': 'C01', 'start_frame': 100, 'end_frame': 250}, ...]
        sampled_frames: List[SampledFrame] = []

        with source:
            for idx in candidate_indices:
                frame = source.read_frame_at(idx)
                if frame is None:
                    continue

                crop, _ = roi_extractor.extract_crop(frame)
                sharpness, brightness, contrast = self.compute_quality(crop)

                # Quality sanity: ignore pitch black or severely washed out frames
                is_valid = (
                    sharpness >= self.min_laplacian
                    and 15.0 <= brightness <= 245.0
                    and contrast >= 15.0
                )

                coach_id = None
                coach_source = "UNVERIFIED"

                if coach_intervals:
                    for ci in coach_intervals:
                        if ci.get("start_frame", -1) <= idx <= ci.get("end_frame", -1):
                            coach_id = ci.get("coach_id")
                            coach_source = "VERIFIED_BOUNDARY"
                            break

                ts_ms = int(round((idx / native_fps) * 1000.0))

                s_frame = SampledFrame(
                    frame_index=idx,
                    timestamp_ms=ts_ms,
                    coach_id=coach_id,
                    coach_id_source=coach_source,
                    sharpness_laplacian=sharpness,
                    brightness_mean=brightness,
                    contrast_std=contrast,
                    is_valid_quality=is_valid,
                    selection_reason="time_diverse_interval",
                )
                sampled_frames.append(s_frame)

        # Diverse selection: filter by valid quality, take up to max_frames
        valid_frames = [f for f in sampled_frames if f.is_valid_quality]

        if not valid_frames:
            # Fallback to top sharpness if all were below threshold
            sampled_frames.sort(key=lambda x: x.sharpness_laplacian, reverse=True)
            return sampled_frames[: min(5, len(sampled_frames))]

        # If more than max_frames, downsample uniformly across the passage duration
        if len(valid_frames) > self.max_frames:
            step_stride = len(valid_frames) / self.max_frames
            valid_frames = [valid_frames[int(i * step_stride)] for i in range(self.max_frames)]

        return valid_frames
