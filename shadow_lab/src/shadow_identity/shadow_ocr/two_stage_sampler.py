"""
Automatic board ROI proposals and two-stage sampling architecture for ShadowOCR.
Separates lightweight frame screening (~40 FPS) from expensive PaddleOCR inference (~5 FPS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np

from shadow_identity.shadow_ocr.contracts import SampledFrame
from shadow_identity.shadow_ocr.sources.finalized_mp4 import FinalizedMp4Source


@dataclass
class BoardRoiProposal:
    frame_index: int
    timestamp_ms: int
    bbox_xywh: Tuple[int, int, int, int]  # x, y, w, h
    aspect_ratio: float
    area: int
    confidence_score: float
    expanded_crop_xyxy: Tuple[int, int, int, int]  # x1, y1, x2, y2 with margin


@dataclass
class SelectedOcrCrop:
    frame_index: int
    timestamp_ms: int
    crop_xyxy: Tuple[int, int, int, int]
    is_proposal: bool
    confidence_score: float
    frame_sharpness: float
    coach_id: Optional[str] = None
    coach_id_source: str = "UNVERIFIED"
    pool: str = "POOL_A"  # "POOL_A" or "POOL_B"


class AutomaticBoardDetector:
    """
    Lightweight color & geometry board proposal generator.
    Finds standard Indian Railway yellow-and-black destination boards in <2ms.
    """

    def __init__(
        self,
        strip_y1: int = 120,
        strip_y2: int = 300,
        min_aspect: float = 2.2,
        min_area: int = 3000,
        margin_x: int = 30,
        margin_y: int = 25,
    ):
        self.strip_y1 = strip_y1
        self.strip_y2 = strip_y2
        self.min_aspect = min_aspect
        self.min_area = min_area
        self.margin_x = margin_x
        self.margin_y = margin_y

    def propose_boards(self, frame: np.ndarray, frame_index: int = 0, timestamp_ms: int = 0) -> List[BoardRoiProposal]:
        h, w = frame.shape[:2]
        y1 = max(0, min(self.strip_y1, h - 1))
        y2 = max(y1 + 10, min(self.strip_y2, h))

        roi_strip = frame[y1:y2, :]
        hsv = cv2.cvtColor(roi_strip, cv2.COLOR_BGR2HSV)

        # Indian Railway yellow destination board HSV profile
        lower_yellow = np.array([14, 55, 55])
        upper_yellow = np.array([36, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Morphological closing to join board text and frame
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        proposals: List[BoardRoiProposal] = []

        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh) if bh > 0 else 0
            area = bw * bh

            if aspect >= self.min_aspect and area >= self.min_area:
                full_y = by + y1
                # Expand safety margin to preserve stencil borders
                ex1 = max(0, bx - self.margin_x)
                ey1 = max(0, full_y - self.margin_y)
                ex2 = min(w, bx + bw + self.margin_x)
                ey2 = min(h, full_y + bh + self.margin_y)

                # Heuristic confidence based on aspect ratio proximity to standard IR board (~3.5 to 5.0)
                aspect_score = 1.0 - min(1.0, abs(aspect - 4.0) / 4.0)
                conf = round(0.5 + 0.5 * aspect_score, 3)

                proposals.append(
                    BoardRoiProposal(
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        bbox_xywh=(bx, full_y, bw, bh),
                        aspect_ratio=round(aspect, 2),
                        area=area,
                        confidence_score=conf,
                        expanded_crop_xyxy=(ex1, ey1, ex2, ey2),
                    )
                )

        proposals.sort(key=lambda p: (p.confidence_score, p.area), reverse=True)
        return proposals


class TwoStageFrameSampler:
    """
    Two-stage architecture:
    Stage 1: Screen every Nth frame via fast metrics (Laplacian + Brightness + Board proposals).
    Stage 2: Rank and pass top candidates to OCR up to max_ocr_calls budget.
    """

    def __init__(
        self,
        screening_fps: float = 2.0,
        max_ocr_budget: int = 20,
        min_laplacian: float = 50.0,
    ):
        self.screening_fps = screening_fps
        self.max_ocr_budget = max_ocr_budget
        self.min_laplacian = min_laplacian
        self.board_detector = AutomaticBoardDetector()

    def screen_passage(
        self,
        source: FinalizedMp4Source,
        coach_intervals: Optional[List[dict]] = None,
    ) -> Tuple[List[SampledFrame], List[BoardRoiProposal]]:
        p = source.passage_input
        native_fps = p.fps if p.fps > 0 else 25.0
        total_frames = p.frame_count

        if total_frames <= 0:
            return [], []

        step = max(1, int(round(native_fps / self.screening_fps)))
        candidate_indices = list(range(0, total_frames, step))

        screened_frames: List[SampledFrame] = []
        all_proposals: List[BoardRoiProposal] = []

        with source:
            for idx in candidate_indices:
                frame = source.read_frame_at(idx)
                if frame is None:
                    continue

                ts_ms = int(round((idx / native_fps) * 1000.0))
                h, w = frame.shape[:2]
                y1 = min(120, h - 1) if h > 1 else 0
                y2 = min(300, h)
                strip = frame[y1:y2, :] if y2 > y1 else frame
                if strip.size == 0:
                    continue
                gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
                lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                b_mean = float(np.mean(gray))
                c_std = float(np.std(gray))

                is_valid = (lap >= self.min_laplacian and 15.0 <= b_mean <= 245.0 and c_std >= 15.0)

                # Check coach interval
                coach_id = None
                coach_source = "UNVERIFIED"
                if coach_intervals:
                    for ci in coach_intervals:
                        if ci.get("start_frame", -1) <= idx <= ci.get("end_frame", -1):
                            coach_id = ci.get("coach_id")
                            coach_source = ci.get("coach_id_source", "UNVERIFIED")
                            break

                # Automatic board proposals
                props = self.board_detector.propose_boards(frame, frame_index=idx, timestamp_ms=ts_ms)
                all_proposals.extend(props)

                sf = SampledFrame(
                    frame_index=idx,
                    timestamp_ms=ts_ms,
                    coach_id=coach_id,
                    coach_id_source=coach_source,
                    sharpness_laplacian=lap,
                    brightness_mean=b_mean,
                    contrast_std=c_std,
                    is_valid_quality=is_valid,
                    selection_reason="has_board_proposal" if props else "screened_quality",
                )
                screened_frames.append(sf)

        self.last_screened_frames: List[SampledFrame] = screened_frames
        selected = self.select_temporal_diverse_frames(
            screened_frames=screened_frames,
            all_proposals=all_proposals,
            budget=self.max_ocr_budget,
            max_per_bucket=2,
            bucket_duration_ms=2000,
        )
        return selected, all_proposals

    @staticmethod
    def select_temporal_diverse_frames(
        screened_frames: List[SampledFrame],
        all_proposals: List[BoardRoiProposal],
        budget: int = 25,
        max_per_bucket: int = 2,
        bucket_duration_ms: int = 2000,
    ) -> List[SampledFrame]:
        """
        Selects frames across 2.0-second temporal coverage buckets using round-robin.
        Ranks frames within each bucket by:
        (has_valid_board_proposal, max_board_confidence, frame_sharpness, -frame_index).
        Caps max_per_bucket to guarantee diversity across the entire passage.
        """
        valid_frames = [f for f in screened_frames if f.is_valid_quality]
        if not valid_frames or budget <= 0:
            return []

        # Index proposals by frame_index
        props_by_frame: dict[int, list[BoardRoiProposal]] = {}
        for p in all_proposals:
            props_by_frame.setdefault(p.frame_index, []).append(p)

        # 1. Group valid frames into chronological temporal buckets (2.0s = 2000ms)
        buckets: dict[int, list[SampledFrame]] = {}
        for f in valid_frames:
            bucket_id = int(f.timestamp_ms // bucket_duration_ms)
            buckets.setdefault(bucket_id, []).append(f)

        # 2. Sort candidate frames within each bucket
        def frame_rank_key(f: SampledFrame):
            f_props = props_by_frame.get(f.frame_index, [])
            has_prop = 1 if len(f_props) > 0 else 0
            max_conf = max([p.confidence_score for p in f_props], default=0.0)
            return (has_prop, max_conf, f.sharpness_laplacian, -f.frame_index)

        sorted_bucket_ids = sorted(buckets.keys())
        bucket_queues: dict[int, list[SampledFrame]] = {}
        for b_id in sorted_bucket_ids:
            bucket_queues[b_id] = sorted(buckets[b_id], key=frame_rank_key, reverse=True)

        selected: List[SampledFrame] = []
        selected_set: set[int] = set()
        bucket_take_count: dict[int, int] = {b_id: 0 for b_id in sorted_bucket_ids}

        # 3. Round-robin among chronological nonempty buckets
        current_cap = max_per_bucket
        while len(selected) < budget:
            progress = False
            for b_id in sorted_bucket_ids:
                if len(selected) >= budget:
                    break
                if bucket_take_count[b_id] >= current_cap:
                    continue

                q = bucket_queues[b_id]
                while q and q[0].frame_index in selected_set:
                    q.pop(0)

                if q:
                    candidate = q.pop(0)
                    selected.append(candidate)
                    selected_set.add(candidate.frame_index)
                    bucket_take_count[b_id] += 1
                    progress = True

            if not progress:
                # If budget remains and all buckets reached cap, relax cap to fill remaining budget
                if any(len(bucket_queues[b]) > 0 for b in sorted_bucket_ids):
                    current_cap += 1
                else:
                    break

        # Restore chronological order
        selected.sort(key=lambda x: x.frame_index)
        return selected

    @staticmethod
    def _compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        ix1 = max(x1_1, x1_2)
        iy1 = max(y1_1, y1_2)
        ix2 = min(x2_1, x2_2)
        iy2 = min(y2_1, y2_2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - inter
        return float(inter / union) if union > 0 else 0.0

    @classmethod
    def select_balanced_two_pool_crops(
        cls,
        screened_frames: List[SampledFrame],
        all_proposals: List[BoardRoiProposal],
        default_crop_xyxy: Tuple[int, int, int, int],
        pool_a_budget: int = 15,
        pool_b_budget: int = 10,
        max_total_budget: int = 25,
        bucket_duration_ms: int = 2000,
    ) -> List[SelectedOcrCrop]:
        """
        Balanced Two-Pool Selection Algorithm:
        - Pool A: up to 15 OCR crop calls for high-quality board proposals with temporal
          diversity and maximum 2 crops per 2-second bucket.
        - Pool B: up to 10 OCR crop calls for the highest-quality unused crops, with
          spatial/temporal deduplication.
        - Total inference limit: exactly 25 maximum actual calls, never exceeded.
        """
        valid_frames = [f for f in screened_frames if f.is_valid_quality]
        if not valid_frames or max_total_budget <= 0:
            return []

        frames_by_idx = {f.frame_index: f for f in valid_frames}
        props_by_frame: dict[int, list[BoardRoiProposal]] = {}
        for p in all_proposals:
            if p.frame_index in frames_by_idx:
                props_by_frame.setdefault(p.frame_index, []).append(p)

        proposal_candidates: List[SelectedOcrCrop] = []
        fallback_candidates: List[SelectedOcrCrop] = []

        for f in valid_frames:
            f_props = props_by_frame.get(f.frame_index, [])
            if f_props:
                f_props_sorted = sorted(f_props, key=lambda p: p.confidence_score, reverse=True)
                kept_props: List[BoardRoiProposal] = []
                for p in f_props_sorted:
                    if not any(cls._compute_iou(p.expanded_crop_xyxy, kp.expanded_crop_xyxy) > 0.6 for kp in kept_props):
                        kept_props.append(p)

                for p in kept_props:
                    crop_item = SelectedOcrCrop(
                        frame_index=f.frame_index,
                        timestamp_ms=f.timestamp_ms,
                        crop_xyxy=p.expanded_crop_xyxy,
                        is_proposal=True,
                        confidence_score=p.confidence_score,
                        frame_sharpness=f.sharpness_laplacian,
                        coach_id=f.coach_id,
                        coach_id_source=f.coach_id_source,
                        pool="POOL_A",
                    )
                    proposal_candidates.append(crop_item)
            else:
                fallback_candidates.append(SelectedOcrCrop(
                    frame_index=f.frame_index,
                    timestamp_ms=f.timestamp_ms,
                    crop_xyxy=default_crop_xyxy,
                    is_proposal=False,
                    confidence_score=0.0,
                    frame_sharpness=f.sharpness_laplacian,
                    coach_id=f.coach_id,
                    coach_id_source=f.coach_id_source,
                    pool="POOL_B",
                ))

        # --- POOL A SELECTION ---
        buckets: dict[int, list[SelectedOcrCrop]] = {}
        for c in proposal_candidates:
            b_id = int(c.timestamp_ms // bucket_duration_ms)
            buckets.setdefault(b_id, []).append(c)

        sorted_b_ids = sorted(buckets.keys())
        bucket_queues: dict[int, list[SelectedOcrCrop]] = {
            b: sorted(buckets[b], key=lambda c: (c.confidence_score, c.frame_sharpness, -c.frame_index), reverse=True)
            for b in sorted_b_ids
        }

        effective_pool_a_budget = min(pool_a_budget, max_total_budget)
        pool_a: List[SelectedOcrCrop] = []
        bucket_take_count: dict[int, int] = {b: 0 for b in sorted_b_ids}

        for pass_num in (1, 2):
            if len(pool_a) >= effective_pool_a_budget:
                break
            for b in sorted_b_ids:
                if len(pool_a) >= effective_pool_a_budget:
                    break
                if bucket_take_count[b] < pass_num and bucket_queues[b]:
                    chosen = bucket_queues[b].pop(0)
                    chosen.pool = "POOL_A"
                    pool_a.append(chosen)
                    bucket_take_count[b] += 1

        # --- POOL B SELECTION ---
        remaining_candidates: List[SelectedOcrCrop] = []
        for b in sorted_b_ids:
            remaining_candidates.extend(bucket_queues[b])
        remaining_candidates.extend(fallback_candidates)

        remaining_candidates.sort(
            key=lambda c: (1 if c.is_proposal else 0, c.confidence_score, c.frame_sharpness, -c.frame_index),
            reverse=True,
        )

        effective_pool_b_budget = min(pool_b_budget, max(0, max_total_budget - len(pool_a)))
        pool_b: List[SelectedOcrCrop] = []
        selected_so_far = list(pool_a)

        for cand in remaining_candidates:
            if len(pool_b) >= effective_pool_b_budget:
                break
            if len(pool_a) + len(pool_b) >= max_total_budget:
                break

            # Spatial & Temporal Deduplication
            is_dup = False
            for s in selected_so_far:
                if s.frame_index == cand.frame_index:
                    if cls._compute_iou(s.crop_xyxy, cand.crop_xyxy) > 0.5:
                        is_dup = True
                        break
                elif s.crop_xyxy == cand.crop_xyxy and abs(s.timestamp_ms - cand.timestamp_ms) < 300:
                    is_dup = True
                    break

            if not is_dup:
                cand.pool = "POOL_B"
                pool_b.append(cand)
                selected_so_far.append(cand)

        total_selected = pool_a + pool_b
        assert len(total_selected) <= max_total_budget, f"Total budget exceeded: {len(total_selected)} > {max_total_budget}"
        total_selected.sort(key=lambda c: (c.frame_index, c.crop_xyxy))
        return total_selected
