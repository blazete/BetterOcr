"""
Unified Train Identity ShadowOCR CLI Orchestrator (Phase 5A Hardened).
Processes a finalized ARDS MP4 passage using two-stage screening + PaddleOCR inference,
physical-coach consensus, and the canonical BetterOcr candidate resolver.
Zero production writes; completely offline and safe.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2

from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.models import (
    NtesSnapshot,
    NtesTrainEntry,
    PassageInput,
    ShadowDecision,
    TimetableRow,
)
from shadow_identity.shadow_ocr.adapter import ShadowOcrToBetterOcrAdapter
from shadow_identity.shadow_ocr.consensus import PhysicalCoachConsensusEngine
from shadow_identity.shadow_ocr.contracts import PassageVideoInput, RawOcrObservation
from shadow_identity.shadow_ocr.paddle_engine import PaddleEngineWrapper
from shadow_identity.shadow_ocr.preprocess import PreprocessingEngine
from shadow_identity.shadow_ocr.settings import ShadowOcrSettings
from shadow_identity.shadow_ocr.sources.finalized_mp4 import FinalizedMp4Source
from shadow_identity.shadow_ocr.temp_mp4_lifecycle import TempMp4LifecycleManager
from shadow_identity.shadow_ocr.text_entities import EntityClassifier
from shadow_identity.shadow_ocr.two_stage_sampler import (
    AutomaticBoardDetector,
    TwoStageFrameSampler,
)
from shadow_identity.timetable_importer import TimetableImporter

_log = logging.getLogger("shadow_ocr.cli")


def atomic_write_json(path: Path, data: Any) -> None:
    """Durably and atomically writes JSON to disk using a temporary file and fsync."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".tmp_{int(time.time()*1000)}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        import os
        os.fsync(f.fileno())
    tmp_path.replace(path)


def atomic_write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Durably and atomically writes JSONL rows to disk using a temporary file and fsync."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".tmp_{int(time.time()*1000)}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        import os
        os.fsync(f.fileno())
    tmp_path.replace(path)


def validate_coach_boundaries(
    boundaries_raw: Any,
    expected_passage_id: str,
    expected_camera_id: str,
    total_frames: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Validates coach-boundary provenance, passage/camera alignment, and frame bounds.
    If valid, entries keep coach_id_source="VERIFIED_BOUNDARY".
    If unverified, untrusted, or mismatched, downgraded to "UNVERIFIED".
    """
    report: Dict[str, Any] = {
        "status": "VALID",
        "provenance_source": None,
        "trusted": False,
        "passage_match": True,
        "camera_match": True,
        "alignment_valid": True,
        "intervals_valid": True,
        "rejection_reasons": [],
    }

    if not boundaries_raw:
        report["status"] = "MISSING"
        return [], report

    intervals: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

    if isinstance(boundaries_raw, list):
        intervals = boundaries_raw
    elif isinstance(boundaries_raw, dict):
        metadata = boundaries_raw.get("metadata", {})
        intervals = boundaries_raw.get("coaches", boundaries_raw.get("intervals", []))

    report["provenance_source"] = metadata.get("source")

    # 1. Provenance & Sensor Source Check
    # Do not trust a JSON file solely because metadata contains trusted keywords.
    # Require genuine hardware provenance tied to trusted sensor metadata, passage/camera identity,
    # and explicit hardware synchronization verification.
    source_val = metadata.get("source")
    hardware_sync_verified = metadata.get("hardware_sync_verified", False)
    sync_method = metadata.get("sync_method")

    trusted_sources = {"ARDS_HARDWARE_BEAM", "SYNCHRONIZED_SENSOR"}
    if source_val in trusted_sources and hardware_sync_verified is True and sync_method:
        report["trusted"] = True
    else:
        report["trusted"] = False
        if source_val not in trusted_sources:
            report["rejection_reasons"].append(f"UNTRUSTED_SOURCE_{source_val}")
        if not hardware_sync_verified:
            report["rejection_reasons"].append("HARDWARE_SYNC_NOT_ESTABLISHED")
        if not sync_method:
            report["rejection_reasons"].append("MISSING_SYNC_METHOD")

    # 2. Passage Association Check - Missing passage ID cannot be accepted as verified
    b_pas = metadata.get("passage_id")
    if not b_pas:
        report["passage_match"] = False
        report["rejection_reasons"].append("MISSING_PASSAGE_ID")
    elif b_pas != expected_passage_id:
        report["passage_match"] = False
        report["rejection_reasons"].append(f"PASSAGE_MISMATCH_{b_pas}_VS_{expected_passage_id}")

    # 3. Camera Association Check - Missing camera ID cannot be accepted as verified
    b_cam = metadata.get("camera_id")
    if not b_cam:
        report["camera_match"] = False
        report["rejection_reasons"].append("MISSING_CAMERA_ID")
    elif b_cam != expected_camera_id:
        report["camera_match"] = False
        report["rejection_reasons"].append(f"CAMERA_MISMATCH_{b_cam}_VS_{expected_camera_id}")

    # 4. Interval and Frame Alignment Validation (Strict non-overlapping intervals)
    valid_intervals: List[Dict[str, Any]] = []
    last_end = -1
    for i, c in enumerate(intervals):
        cid = c.get("coach_id", f"C{i+1:02d}")
        sf = c.get("start_frame")
        ef = c.get("end_frame")

        if sf is None or ef is None:
            report["intervals_valid"] = False
            report["rejection_reasons"].append(f"COACH_{cid}_MISSING_FRAMES")
            continue

        if not (0 <= sf <= ef <= total_frames):
            report["intervals_valid"] = False
            report["rejection_reasons"].append(f"COACH_{cid}_OUT_OF_BOUNDS_{sf}_{ef}_MAX_{total_frames}")
            continue

        # Strict non-overlapping check: next coach start must be greater than previous coach end
        if sf <= last_end:
            report["intervals_valid"] = False
            report["rejection_reasons"].append(f"COACH_{cid}_OVERLAPPING_INTERVAL_{sf}_LE_{last_end}")

        last_end = max(last_end, ef)
        valid_intervals.append(c)

    is_fully_verified = (
        report["trusted"]
        and report["passage_match"]
        and report["camera_match"]
        and report["intervals_valid"]
        and len(valid_intervals) > 0
    )

    if is_fully_verified:
        report["status"] = "VERIFIED_BOUNDARY"
        for v in valid_intervals:
            v["coach_id_source"] = "VERIFIED_BOUNDARY"
    else:
        report["status"] = "UNVERIFIED"
        for v in valid_intervals:
            v["coach_id_source"] = "UNVERIFIED"

    return valid_intervals, report


def load_coach_boundaries_file(json_path: Optional[str]) -> Any:
    """Load coach boundaries raw data from JSON file."""
    if not json_path:
        return None
    p = Path(json_path)
    if not p.is_file():
        raise FileNotFoundError(f"Coach boundaries file not found: {json_path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_timetable_rows(csv_path: Optional[str], site_id: str) -> List[TimetableRow]:
    """Load canonical timetable rows from CSV."""
    if not csv_path:
        return []
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"Timetable file not found: {csv_path}")
    return TimetableImporter.load_schedule_csv(
        file_path=p,
        site_id=site_id,
        station_code=site_id.split("_")[0].upper(),
        schedule_version="1.0",
    )


def load_ntes_snapshot(json_path: Optional[str]) -> Optional[NtesSnapshot]:
    """Load an NTES snapshot JSON if provided."""
    if not json_path:
        return None
    p = Path(json_path)
    if not p.is_file():
        raise FileNotFoundError(f"NTES snapshot file not found: {json_path}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    trains = [
        NtesTrainEntry(
            train_number=t["train_number"],
            train_name=t["train_name"],
            platform=str(t.get("platform", "")),
            time_raw=t.get("time_raw", ""),
        )
        for t in data.get("trains", [])
    ]
    return NtesSnapshot(
        site_id=data.get("site_id", ""),
        station_code=data.get("station_code", ""),
        requested_at=data.get("requested_at", ""),
        fetched_at=data.get("fetched_at", ""),
        window_hr=data.get("window_hr", 2),
        available=data.get("available", True),
        from_cache=data.get("from_cache", False),
        stale_fallback=data.get("stale_fallback", False),
        trains=trains,
        raw_hash=data.get("raw_hash", ""),
    )


def execute_pipeline(
    mp4_path: str,
    site_id: str,
    passage_id: str,
    platform: Optional[str] = None,
    passage_start: Optional[str] = None,
    passage_end: Optional[str] = None,
    camera_id: str = "side_a",
    coach_boundaries_path: Optional[str] = None,
    timetable_path: Optional[str] = None,
    ntes_snapshot_path: Optional[str] = None,
    raw_observations_path: Optional[str] = None,
    output_path: Optional[str] = None,
    stage_temp_copy: bool = False,
    temp_dir: Optional[str] = None,
    cleanup_retention_sec: float = 3600.0,
    screening_fps: float = 5.0,
    max_ocr_budget: int = 25,
) -> Dict[str, Any]:
    """
    Hardened Phase 5A execution pipeline.
    """
    t_start = time.time()
    timings: Dict[str, float] = {}

    orig_mp4_p = Path(mp4_path).resolve()
    if not orig_mp4_p.is_file():
        raise FileNotFoundError(f"MP4 file not found: {mp4_path}")

    # 1. TempMp4LifecycleManager Staging Integration
    t0 = time.time()
    lifecycle_mgr: Optional[TempMp4LifecycleManager] = None
    staged_job: Optional[Any] = None
    active_mp4_p = orig_mp4_p

    if stage_temp_copy:
        td = Path(temp_dir or "shadow_lab/data/temp_mp4")
        lifecycle_mgr = TempMp4LifecycleManager(temp_dir=td, retention_completed_sec=cleanup_retention_sec)
        staged_job = lifecycle_mgr.stage_temporary_copy(
            original_mp4_path=orig_mp4_p,
            passage_id=passage_id,
            site_id=site_id,
        )
        active_mp4_p = Path(staged_job.temp_mp4_path).resolve()
        assert active_mp4_p.is_file(), f"Staged temporary MP4 missing: {active_mp4_p}"

    # 2. Metadata Probing & Ingestion
    cap = cv2.VideoCapture(str(active_mp4_p))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration_sec = total_frames / native_fps if native_fps > 0 else 0.0

    # Boundary Validation & Provenance Enforcement
    raw_bounds = load_coach_boundaries_file(coach_boundaries_path)
    coach_intervals, boundary_report = validate_coach_boundaries(
        raw_bounds,
        expected_passage_id=passage_id,
        expected_camera_id=camera_id,
        total_frames=total_frames,
    )

    p_inp = PassageVideoInput(
        site_id=site_id,
        installation_id="SHADOW_INST_01",
        passage_id=passage_id,
        source_camera_id=camera_id,
        video_path=str(active_mp4_p),
        fps=native_fps,
        frame_count=total_frames,
        duration_sec=round(duration_sec, 2),
        coach_intervals=coach_intervals,
    )
    video_sha = p_inp.compute_and_verify_hash()
    source = FinalizedMp4Source(p_inp)
    source.validate()
    timings["ingestion_and_hashing_s"] = round(time.time() - t0, 3)

    # 3. Two-Stage Screening & Proposals
    t0 = time.time()
    detector = AutomaticBoardDetector(
        strip_y1=70,
        strip_y2=320,
        min_aspect=2.0,
        min_area=2500,
        margin_x=120,
        margin_y=35,
    )
    sampler = TwoStageFrameSampler(screening_fps=screening_fps, max_ocr_budget=max_ocr_budget)
    sampler.board_detector = detector

    selected_frames, all_proposals = sampler.screen_passage(source, coach_intervals=coach_intervals)
    screened_frames = getattr(sampler, "last_screened_frames", selected_frames)

    # Configurable Camera-Specific Resolution-Aware Fallback Bounds
    settings = ShadowOcrSettings()
    cfg_roi = settings.default_board_rois.get(camera_id, (0, 130, 768, 280))
    scale_x = width / 768.0 if width > 0 else 1.0
    scale_y = height / 576.0 if height > 0 else 1.0
    fallback_crop_xyxy = (
        int(cfg_roi[0] * scale_x),
        int(cfg_roi[1] * scale_y),
        min(width, int(cfg_roi[2] * scale_x)),
        min(height, int(cfg_roi[3] * scale_y)),
    )

    # Balanced Two-Pool Selection:
    # Pool A: up to 15 OCR crop calls for high-quality board proposals with temporal diversity and max 2 crops / 2s bucket.
    # Pool B: up to 10 OCR crop calls for highest-quality unused crops with spatial/temporal deduplication.
    # Total limit: exactly 25 maximum actual calls, never exceeded.
    selected_crops = TwoStageFrameSampler.select_balanced_two_pool_crops(
        screened_frames=screened_frames,
        all_proposals=all_proposals,
        default_crop_xyxy=fallback_crop_xyxy,
        pool_a_budget=15,
        pool_b_budget=10,
        max_total_budget=min(max_ocr_budget, 25),
        bucket_duration_ms=2000,
    )

    timings["screening_and_proposals_s"] = round(time.time() - t0, 3)

    # 4. OCR Inference with Balanced Two-Pool Selection
    t0 = time.time()
    engine = PaddleEngineWrapper.get_shared_instance()
    raw_observations: List[RawOcrObservation] = []

    actual_ocr_calls = 0
    pool_a_calls = 0
    pool_b_calls = 0
    fallback_calls = 0
    unique_frames_in_ocr = set()
    board_buckets_covered = set()
    crop_coordinates = []

    cached_frame_idx = -1
    cached_frame = None

    with source:
        for crop_item in selected_crops:
            if actual_ocr_calls >= max_ocr_budget or actual_ocr_calls >= 25:
                break

            if crop_item.frame_index == cached_frame_idx and cached_frame is not None:
                frame = cached_frame
            else:
                frame = source.read_frame_at(crop_item.frame_index)
                cached_frame_idx = crop_item.frame_index
                cached_frame = frame

            if frame is None:
                continue

            cx1, cy1, cx2, cy2 = crop_item.crop_xyxy
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            actual_ocr_calls += 1
            if crop_item.pool == "POOL_A":
                pool_a_calls += 1
            else:
                pool_b_calls += 1

            if not crop_item.is_proposal:
                fallback_calls += 1

            unique_frames_in_ocr.add(crop_item.frame_index)
            board_buckets_covered.add(int(crop_item.timestamp_ms // 2000))
            crop_coordinates.append([cx1, cy1, cx2, cy2])

            p1 = PreprocessingEngine.apply_p1_color_2x_bicubic(crop)
            detections = engine.predict(p1)

            for poly, text, conf in detections:
                if conf < 0.45 or not text.strip():
                    continue

                entities = EntityClassifier.classify_all(text, conf)
                full_poly = []
                if poly:
                    for pt in poly:
                        px, py = pt[0] / 2.0, pt[1] / 2.0
                        full_poly.append([int(px + cx1), int(py + cy1)])

                for ent in entities:
                    obs = RawOcrObservation(
                        schema_version="shadow_ocr.v1",
                        site_id=site_id,
                        passage_id=passage_id,
                        camera_id=camera_id,
                        frame_id=crop_item.frame_index,
                        timestamp_ms=crop_item.timestamp_ms,
                        board_roi_xyxy=[cx1, cy1, cx2, cy2],
                        coach_id=crop_item.coach_id,
                        coach_id_source=crop_item.coach_id_source,
                        ocr_model_id=engine.model_id,
                        preprocessing_variant="P1_COLOR_2X_BICUBIC",
                        raw_text=text.strip(),
                        normalized_text=ent.normalized_text,
                        entity_type=ent.entity_type,
                        ocr_confidence=conf,
                        bbox_in_crop=[int(p) for pt in poly for p in pt] if poly else [],
                        bbox_full_frame=[int(p) for pt in full_poly for p in pt] if full_poly else [],
                        source_video_sha256=video_sha,
                    )
                    raw_observations.append(obs)

    timings["ocr_inference_s"] = round(time.time() - t0, 3)

    # Persist Raw OCR Evidence JSONL if requested
    if raw_observations_path:
        raw_p = Path(raw_observations_path)
        raw_p.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_p, "w", encoding="utf-8") as rf:
            for obs in raw_observations:
                rf.write(json.dumps(obs.to_dict()) + "\n")

    # 5. Physical Coach Consensus & Passage Visual Identity
    t0 = time.time()
    consensus_engine = PhysicalCoachConsensusEngine()
    identity_res = consensus_engine.build_passage_identity(
        site_id=site_id,
        passage_id=passage_id,
        camera_id=camera_id,
        video_sha256=video_sha,
        observations=raw_observations,
        processing_latency_ms=timings["ocr_inference_s"] * 1000.0,
    )
    timings["consensus_ms"] = round((time.time() - t0) * 1000.0, 2)

    # 6. Resolver Adaptation & Invocation (Timestamp Hardening & NTES +180s Contract)
    t0 = time.time()
    visual_evidence = ShadowOcrToBetterOcrAdapter.to_visual_ocr_evidence(
        identity_res, raw_observations
    )

    # Verified Source Timestamps: If missing, pass empty strings to disable time corroboration
    has_verified_timestamps = bool(passage_start and passage_end)
    p_start_iso = passage_start or ""
    p_end_iso = passage_end or ""

    passage_canonical = PassageInput(
        site_id=site_id,
        installation_id="SHADOW_INST_01",
        passage_id=passage_id,
        camera_id=camera_id,
        passage_start=p_start_iso,
        passage_end=p_end_iso,
        possible_platforms=[platform] if platform else [],
        observed_platform=platform,
    )

    timetable_rows = load_timetable_rows(timetable_path, site_id)
    ntes_snap = load_ntes_snapshot(ntes_snapshot_path)

    # Evaluate NTES +180s Timing Contract
    # Distinguish: scheduled verification time, actual fetch time, verification delay,
    # snapshot freshness, stale fallback, and eligibility for resolver confirmation.
    ntes_timing_contract: Dict[str, Any] = {
        "verified_timestamps_present": has_verified_timestamps,
        "ntes_available": ntes_snap is not None and ntes_snap.available,
        "acquisition_time": ntes_snap.fetched_at if ntes_snap else None,
        "scheduled_verification_time": None,
        "actual_fetch_time": ntes_snap.fetched_at if ntes_snap else None,
        "verification_delay_seconds": None,
        "relative_to_passage_end_sec": None,
        "freshness_valid": False,
        "stale_fallback": ntes_snap.stale_fallback if ntes_snap else False,
        "eligible_for_t2_confirmation": False,
        "rejection_reasons": [],
    }

    ntes_for_resolver = ntes_snap
    if ntes_snap and ntes_snap.available:
        if not passage_end:
            ntes_timing_contract["rejection_reasons"].append("MISSING_PASSAGE_END_TIMESTAMP")
            ntes_for_resolver = None
        else:
            try:
                p_end_dt = datetime.fromisoformat(passage_end)
                sched_dt = p_end_dt + timedelta(seconds=180.0)
                ntes_timing_contract["scheduled_verification_time"] = sched_dt.isoformat()

                n_fetched_dt = datetime.fromisoformat(ntes_snap.fetched_at)
                delta_s = (n_fetched_dt - p_end_dt).total_seconds()
                delay_s = (n_fetched_dt - sched_dt).total_seconds()

                ntes_timing_contract["relative_to_passage_end_sec"] = round(delta_s, 2)
                ntes_timing_contract["verification_delay_seconds"] = round(delay_s, 2)

                is_fresh = not ntes_snap.stale_fallback
                ntes_timing_contract["freshness_valid"] = is_fresh

                # Strict timing policy: A snapshot fetched < 180s after passage end (e.g. +20s)
                # is NOT a +180-second verification snapshot.
                if delta_s < 180.0:
                    ntes_timing_contract["rejection_reasons"].append(
                        f"FETCHED_TOO_EARLY_FOR_T2_CONFIRMATION_{round(delta_s, 1)}S_LT_180S"
                    )
                    ntes_for_resolver = None
                elif not is_fresh:
                    ntes_timing_contract["rejection_reasons"].append("SNAPSHOT_IS_STALE_FALLBACK")
                    ntes_for_resolver = None
                else:
                    ntes_timing_contract["eligible_for_t2_confirmation"] = True
            except Exception as e:
                _log.warning("Could not calculate NTES timing contract: %s", e)
                ntes_timing_contract["rejection_reasons"].append(f"CALCULATION_ERROR_{e}")
                ntes_for_resolver = None

    decision: ShadowDecision = resolve_shadow(
        passage=passage_canonical,
        visual_evidence=visual_evidence,
        site_schedule_rows=timetable_rows,
        ntes_before=ntes_for_resolver,
    )
    timings["resolver_arbitration_ms"] = round((time.time() - t0) * 1000.0, 2)
    timings["total_wall_clock_s"] = round(time.time() - t_start, 3)

    # 7. Output Result Assembly (C4 Schema + Backward Compatibility)
    from shadow_identity.evidence_fusion import format_v1_identity_output

    rec_names = [decision.recognized_train_name] if getattr(decision, "recognized_train_name", None) else [
        c.name for c in identity_res.train_name_candidates
    ]
    vis_nums = [c.number for c in identity_res.service_number_candidates]
    fam_cands = getattr(decision, "service_family_candidates", [])
    active_num = getattr(decision, "active_service_number", None)
    dir_ver = getattr(decision, "active_direction_verified", False)

    if decision.shadow_status == "CONFLICT":
        id_status = "CONFLICT"
    elif dir_ver and active_num is not None:
        id_status = "VERIFIED_ACTIVE"
    elif fam_cands:
        id_status = "FAMILY_CANDIDATE"
    else:
        id_status = "UNKNOWN"

    output_dict = format_v1_identity_output(
        site_id=site_id,
        passage_id=passage_id,
        input_video_sha256=video_sha,
        recognized_train_names=rec_names,
        visual_service_numbers=vis_nums,
        recognized_endpoints=identity_res.endpoint_candidates,
        service_family_candidates=fam_cands,
        active_service_number=active_num,
        active_direction_verified=dir_ver,
        identity_status=id_status,
        reason_codes=decision.reason_codes,
        evidence_provenance=[
            {
                "evidence_type": "VISUAL_OBSERVATION",
                "observations_count": len(raw_observations),
                "distinct_coaches": identity_res.distinct_verified_coaches_total,
                "tracking_quality": identity_res.coach_tracking_quality,
            }
        ],
        contradictions=[rc for rc in decision.reason_codes if "CONFLICT" in rc],
        processing_metrics={
            "ocr_calls_executed": actual_ocr_calls,
            "pool_a_calls": pool_a_calls,
            "pool_b_calls": pool_b_calls,
            "total_wall_clock_s": timings.get("total_wall_clock_s", 0.0),
        },
        passage_metadata={
            "passage_id": passage_id,
            "site_id": site_id,
            "camera_id": camera_id,
            "platform": platform,
            "passage_start": p_start_iso if p_start_iso else None,
            "passage_end": p_end_iso if p_end_iso else None,
            "time_corroboration_enabled": has_verified_timestamps,
            "video_path": str(active_mp4_p),
            "original_video_path": str(orig_mp4_p),
            "video_sha256": video_sha,
            "video_properties": {
                "width": width,
                "height": height,
                "native_fps": native_fps,
                "total_frames": total_frames,
                "duration_sec": round(duration_sec, 2),
            },
            "boundary_provenance": boundary_report,
            "staged_temporary_job": staged_job.to_dict() if staged_job else None,
        },
        visual_evidence={
            "tracking_quality": identity_res.coach_tracking_quality,
            "distinct_verified_coaches": identity_res.distinct_verified_coaches_total,
            "visual_pair_status": identity_res.visual_pair_status,
            "stable_pair": identity_res.stable_pair,
            "dominant_service_number": identity_res.dominant_service_number,
            "service_number_candidates": [
                {
                    "number": c.number,
                    "distinct_coaches": c.distinct_coaches,
                    "total_observations": c.total_observations,
                    "mean_confidence": c.mean_confidence,
                    "supporting_coach_ids": c.supporting_coach_ids,
                }
                for c in identity_res.service_number_candidates
            ],
            "train_name_candidates": [
                {
                    "name": c.name,
                    "distinct_coaches": c.distinct_coaches,
                    "total_observations": c.total_observations,
                    "mean_confidence": c.mean_confidence,
                }
                for c in identity_res.train_name_candidates
            ],
            "endpoint_candidates": identity_res.endpoint_candidates,
            "total_raw_observations": len(raw_observations),
            "evaluated_frames_count": len(unique_frames_in_ocr),
            "ocr_calls_executed": actual_ocr_calls,
            "pool_a_calls": pool_a_calls,
            "pool_b_calls": pool_b_calls,
            "board_buckets_covered": sorted(list(board_buckets_covered)),
            "unique_frame_count": len(unique_frames_in_ocr),
            "fallback_calls": fallback_calls,
            "crop_coordinates": crop_coordinates,
            "selection_explanation": "balanced_two_pool_selection",
        },
        resolver_decision={
            "shadow_final_number": decision.shadow_final_number,
            "shadow_status": decision.shadow_status,
            "reason_codes": decision.reason_codes,
            "recognized_train_name": getattr(decision, "recognized_train_name", None),
            "service_family_candidates": getattr(decision, "service_family_candidates", []),
            "active_service_number": getattr(decision, "active_service_number", None),
            "active_direction_verified": getattr(decision, "active_direction_verified", False),
            "ranked_candidates": [
                {
                    "train_number": c.train_number,
                    "train_name": c.train_name,
                    "score": c.score,
                    "evidence_list": c.evidence_list,
                }
                for c in decision.candidate_services
            ],
            "timetable_available": len(timetable_rows) > 0,
            "ntes_available": ntes_snap is not None and ntes_snap.available,
            "ntes_timing_contract": ntes_timing_contract,
        },
        performance_latency=timings,
    )

    # 8. Durable Atomic Persistence Prior to Lifecycle Completion
    # Make final decision JSON and raw evidence JSONL durable BEFORE marking staging job completed
    if raw_observations_path:
        atomic_write_jsonl(Path(raw_observations_path), [obs.to_dict() for obs in raw_observations])

    if output_path:
        atomic_write_json(Path(output_path), output_dict)

    # 9. Lifecycle Completion & Cleanup
    # 60-minute cleanup starts ONLY AFTER successful durable persistence
    if staged_job and lifecycle_mgr:
        lifecycle_mgr.mark_completed(staged_job.job_id)
        lifecycle_mgr.run_cleanup_due()
        output_dict["passage_metadata"]["staged_temporary_job"] = lifecycle_mgr._jobs[staged_job.job_id].to_dict()

    return output_dict


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ShadowOCR Train Identity CLI (Phase 5A Hardened) — Autonomous offline MP4 visual inference and arbitration."
    )
    parser.add_argument("--mp4", required=True, help="Path to finalized passage MP4 recording")
    parser.add_argument("--site-id", required=True, help="Site identifier (e.g. ndls_pf12_13)")
    parser.add_argument("--platform", default=None, help="Observed platform number (e.g. 12)")
    parser.add_argument("--passage-id", required=True, help="Unique passage identifier")
    parser.add_argument("--passage-start", default=None, help="Verified passage start ISO 8601 timestamp")
    parser.add_argument("--passage-end", default=None, help="Verified passage end ISO 8601 timestamp")
    parser.add_argument("--camera-id", default="side_a", help="Camera identifier (default: side_a)")
    parser.add_argument("--coach-boundaries", default=None, help="Path to verified coach boundary JSON")
    parser.add_argument("--timetable", default=None, help="Path to timetable CSV schedule")
    parser.add_argument("--ntes-snapshot", default=None, help="Path to NTES snapshot JSON")
    parser.add_argument("--raw-observations", default=None, help="Path to output raw OCR observations JSONL")
    parser.add_argument("--stage-temp-copy", action="store_true", help="Stage MP4 copy in shadow temp dir before processing")
    parser.add_argument("--temp-dir", default=None, help="Shadow temporary MP4 storage directory")
    parser.add_argument("--cleanup-retention-sec", type=float, default=3600.0, help="Retention period in seconds before deletion (default: 3600s)")
    parser.add_argument("--output", default=None, help="Destination JSON file for output results")

    args = parser.parse_args()

    try:
        res = execute_pipeline(
            mp4_path=args.mp4,
            site_id=args.site_id,
            passage_id=args.passage_id,
            platform=args.platform,
            passage_start=args.passage_start,
            passage_end=args.passage_end,
            camera_id=args.camera_id,
            coach_boundaries_path=args.coach_boundaries,
            timetable_path=args.timetable,
            ntes_snapshot_path=args.ntes_snapshot,
            raw_observations_path=args.raw_observations,
            output_path=args.output,
            stage_temp_copy=args.stage_temp_copy,
            temp_dir=args.temp_dir,
            cleanup_retention_sec=args.cleanup_retention_sec,
        )

        output_json = json.dumps(res, indent=2)
        if args.output:
            print(f"[OK] Results durably written to {args.output}")

        print(output_json)
        return 0

    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
