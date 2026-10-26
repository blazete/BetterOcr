"""
Contracts and data models for ShadowOCR pipeline.
Strictly decoupled from production ARDS and ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PassageVideoInput:
    """
    Validated metadata and handle for a finalized passage MP4 recording.
    """
    site_id: str
    installation_id: str
    passage_id: str
    video_path: str
    source_camera_id: str = "side_a"
    capture_start_utc: str = ""
    capture_end_utc: str = ""
    station_timezone: str = "Asia/Kolkata"
    site_profile_version: str = "1.0"
    allowed_platforms: List[str] = field(default_factory=list)
    observed_platform: Optional[str] = None
    camera_orientation: str = "normal"
    board_roi_xyxy: Optional[Tuple[int, int, int, int]] = None
    coach_intervals: List[Dict[str, Any]] = field(default_factory=list)
    video_sha256: str = ""
    file_size_bytes: int = 0
    duration_sec: float = 0.0
    frame_count: int = 0
    fps: float = 25.0

    def compute_and_verify_hash(self) -> str:
        p = Path(self.video_path)
        if not p.is_file():
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        self.video_sha256 = h.hexdigest()
        self.file_size_bytes = p.stat().st_size
        return self.video_sha256

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BoardRoiConfig:
    """
    Camera-specific board region-of-interest bounding box.
    """
    camera_id: str
    roi_xyxy: Tuple[int, int, int, int]
    resolution_wh: Tuple[int, int]
    description: str = ""

    def validate_bounds(self) -> bool:
        x1, y1, x2, y2 = self.roi_xyxy
        w, h = self.resolution_wh
        return 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h


@dataclass
class SampledFrame:
    """
    A single decoded frame selected for OCR inference.
    """
    frame_index: int
    timestamp_ms: int
    coach_id: Optional[str]
    coach_id_source: str  # "VERIFIED_BOUNDARY" | "ESTIMATED_WINDOW" | "UNVERIFIED"
    sharpness_laplacian: float
    brightness_mean: float
    contrast_std: float
    is_valid_quality: bool
    selection_reason: str


@dataclass
class RawOcrObservation:
    """
    Low-level OCR observation on a specific board crop.
    Retains all raw model metrics and uncertainty.
    """
    schema_version: str = "shadow_ocr.v1"
    site_id: str = ""
    passage_id: str = ""
    camera_id: str = ""
    frame_id: int = 0
    timestamp_ms: int = 0
    board_roi_xyxy: List[int] = field(default_factory=list)
    coach_id: Optional[str] = None
    coach_id_source: str = "UNVERIFIED"
    ocr_model_id: str = "PP-OCRv6"
    preprocessing_variant: str = "P1_COLOR_2X_BICUBIC"
    raw_text: str = ""
    normalized_text: str = ""
    entity_type: str = "UNKNOWN"  # "SERVICE_NUMBER" | "TRAIN_NAME" | "ENDPOINT" | "NEGATIVE_STENCIL" | "UNKNOWN"
    ocr_confidence: float = 0.0
    bbox_in_crop: List[int] = field(default_factory=list)
    bbox_full_frame: List[int] = field(default_factory=list)
    source_video_sha256: str = ""
    inference_timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
