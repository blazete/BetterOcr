"""
Read-only MP4 source decoder and validator.
Operates strictly on completed, immutable local video copies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional, Tuple
import cv2
import numpy as np

from shadow_identity.shadow_ocr.contracts import PassageVideoInput


class FinalizedMp4Source:
    """
    Safely opens and reads frames from a completed passage recording.
    Guarantees read-only semantics; rejects unstable or non-existent files.
    """

    def __init__(self, passage_input: PassageVideoInput):
        self.passage_input = passage_input
        self.video_path = Path(passage_input.video_path).resolve()
        self._cap: Optional[cv2.VideoCapture] = None

    def validate(self) -> Tuple[bool, str]:
        """
        Verify file exists, is regular file, is not 0-bytes, and container opens.
        """
        if not self.video_path.is_file():
            return False, f"File does not exist: {self.video_path}"

        size = self.video_path.stat().st_size
        if size == 0:
            return False, f"File is empty (0 bytes): {self.video_path}"

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            cap.release()
            return False, f"OpenCV failed to open MP4 container: {self.video_path}"

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
            return False, f"Invalid container metadata: frames={frame_count}, fps={fps}, res={width}x{height}"

        self.passage_input.frame_count = frame_count
        self.passage_input.fps = fps
        self.passage_input.duration_sec = frame_count / fps if fps > 0 else 0.0
        self.passage_input.file_size_bytes = size
        return True, "OK"

    def open(self) -> cv2.VideoCapture:
        if self._cap is not None and self._cap.isOpened():
            return self._cap
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise IOError(f"Could not open video file: {self.video_path}")
        return self._cap

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_frame_at(self, frame_index: int) -> Optional[np.ndarray]:
        cap = self.open()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        return frame if ret else None

    def __enter__(self) -> FinalizedMp4Source:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
