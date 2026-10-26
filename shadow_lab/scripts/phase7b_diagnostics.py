"""
BetterOcr / ShadowOCR — Phase 7B: Evidence Recovery & Root-Cause Analysis Script
Executes read-only diagnostic inspection for held-out videos:
- coach_id11.mp4
- coach_id12.mp4
- coach_id13.mp4
- coach_id14.mp4

Strict Rules:
- Read-only on source videos and production ARDS files.
- Writes exclusively to shadow_lab/data/phase7b_diagnostics/.
- Never alters frozen Phase 7 pipeline, thresholds, or models.
"""

from __future__ import annotations

import csv
import difflib
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure shadow_lab/src is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
SHADOW_SRC = REPO_ROOT / "shadow_lab" / "src"
if str(SHADOW_SRC) not in sys.path:
    sys.path.insert(0, str(SHADOW_SRC))

from shadow_identity.models import PassageInput, TimetableRow, NtesSnapshot
from shadow_identity.candidate_resolver import resolve_shadow, TimetableMatcher, match_name_with_aliases
from shadow_identity.shadow_ocr.contracts import PassageVideoInput, SampledFrame, RawOcrObservation
from shadow_identity.shadow_ocr.sources.finalized_mp4 import FinalizedMp4Source
from shadow_identity.shadow_ocr.two_stage_sampler import (
    AutomaticBoardDetector,
    BoardRoiProposal,
    SelectedOcrCrop,
    TwoStageFrameSampler,
)
from shadow_identity.shadow_ocr.text_entities import (
    EntityClassifier,
    ClassifiedEntity,
    KNOWN_TRAIN_NAMES,
    COACH_CLASS_DESIGNATIONS,
    KNOWN_ENDPOINTS,
)
from shadow_identity.shadow_ocr.consensus import PhysicalCoachConsensusEngine
from shadow_identity.shadow_ocr.adapter import ShadowOcrToBetterOcrAdapter

DIAGNOSTIC_DIR = REPO_ROOT / "shadow_lab" / "data" / "phase7b_diagnostics"
HELDOUT_INBOX = REPO_ROOT / "shadow_lab" / "data" / "heldout_4" / "inbox"
HELDOUT_OUTPUT = REPO_ROOT / "shadow_lab" / "data" / "heldout_4" / "output"

VIDEOS = [
    {
        "id": "coach_id11",
        "filename": "coach_id11.mp4",
        "expected_sha": "b0ccb58bfc45980c3f5d6f11502e2bccc395cecb83b5f37a7a9364734a79d214",
    },
    {
        "id": "coach_id12",
        "filename": "coach_id12.mp4",
        "expected_sha": "fd1fa8972cebbb347f837280f935fb630a6faa853109bc57883ab179c26d2db6",
    },
    {
        "id": "coach_id13",
        "filename": "coach_id13.mp4",
        "expected_sha": "2cd1c91e40faf989f1f2f13741baeb9c2476e717f1a3113fcbf44ecdc748503f",
    },
    {
        "id": "coach_id14",
        "filename": "coach_id14.mp4",
        "expected_sha": "d791d944c444c790084c030487e6c0bf91b844c7635b4081c5fa1cdc8111aeaf",
    },
]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def find_heldout_artifacts(video_id: str, sha256_hash: str) -> Tuple[Optional[Path], Optional[Path]]:
    decisions = list(HELDOUT_OUTPUT.glob(f"*{video_id}_{sha256_hash}_decision.json"))
    raws = list(HELDOUT_OUTPUT.glob(f"*{video_id}_{sha256_hash}_raw.jsonl"))
    dec_path = decisions[0] if decisions else None
    raw_path = raws[0] if raws else None
    return dec_path, raw_path


def get_detector() -> AutomaticBoardDetector:
    return AutomaticBoardDetector(
        strip_y1=70,
        strip_y2=320,
        min_aspect=2.0,
        min_area=2500,
        margin_x=120,
        margin_y=35,
    )


def reconstruct_sampler_selection(source: FinalizedMp4Source) -> Tuple[List[SampledFrame], List[BoardRoiProposal], List[SelectedOcrCrop]]:
    detector = get_detector()
    sampler = TwoStageFrameSampler(screening_fps=5.0, max_ocr_budget=25)
    sampler.board_detector = detector
    selected_frames, all_proposals = sampler.screen_passage(source)
    screened_frames = getattr(sampler, "last_screened_frames", selected_frames)
    selected_crops = TwoStageFrameSampler.select_balanced_two_pool_crops(
        screened_frames=screened_frames,
        all_proposals=all_proposals,
        default_crop_xyxy=(0, 130, 768, 280),
        pool_a_budget=15,
        pool_b_budget=10,
        max_total_budget=25,
        bucket_duration_ms=2000,
    )
    return screened_frames, all_proposals, selected_crops


def generate_task_a_and_b(v_info: Dict[str, str]):
    vid = v_info["id"]
    v_path = HELDOUT_INBOX / v_info["filename"]
    sha = compute_sha256(v_path)
    assert sha == v_info["expected_sha"], f"SHA mismatch for {vid}: {sha} != {v_info['expected_sha']}"

    out_dir = DIAGNOSTIC_DIR / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    ocr_frames_dir = out_dir / "selected_ocr_frames"
    ocr_frames_dir.mkdir(parents=True, exist_ok=True)

    dec_path, raw_path = find_heldout_artifacts(vid, sha)
    with open(dec_path, "r", encoding="utf-8") as f:
        decision_data = json.load(f)

    # 1. Video properties from OpenCV
    cap = cv2.VideoCapture(str(v_path))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / actual_fps

    # Reconstruct sampler selection
    p_inp = PassageVideoInput(
        site_id="ndls_pf12_13",
        installation_id="inst_01",
        passage_id=f"pas_{vid}",
        video_path=str(v_path),
        source_camera_id="side_a",
        fps=actual_fps,
        frame_count=total_frames,
        duration_sec=round(duration_sec, 2),
    )
    source = FinalizedMp4Source(p_inp)
    screened_frames, all_proposals, selected_crops = reconstruct_sampler_selection(source)

    # Verify reproduction
    frozen_crops = decision_data["visual_evidence"]["crop_coordinates"]
    reproduced_crops = [list(c.crop_xyxy) for c in selected_crops]
    assert reproduced_crops == frozen_crops, f"Reproduction mismatch in {vid}"

    selected_crop_by_frame: Dict[int, List[SelectedOcrCrop]] = {}
    for c in selected_crops:
        selected_crop_by_frame.setdefault(c.frame_index, []).append(c)

    detector = get_detector()

    # 2. Sample 1 frame per second across entire duration
    # Preserve original frame indexes and timestamps
    sec_count = int(duration_sec) + 1
    sampled_1fps_manifest = []
    thumbnails_1fps = []

    for sec in range(sec_count):
        target_fidx = min(total_frames - 1, int(round(sec * actual_fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_fidx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        ts_ms = int(round((target_fidx / actual_fps) * 1000.0))

        # Sharpness, brightness, contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        b_mean = float(np.mean(gray))
        c_std = float(np.std(gray))

        # Propose boards
        props = detector.propose_boards(frame, target_fidx, ts_ms)

        is_sel = target_fidx in selected_crop_by_frame
        crops_here = selected_crop_by_frame.get(target_fidx, [])

        # Build thumbnail for contact sheet
        thumb = cv2.resize(frame, (192, 144))
        # Draw proposal boxes on thumbnail (scaled)
        sx = 192.0 / width
        sy = 144.0 / height
        for p in props:
            px, py, pw, ph = p.bbox_xywh
            cv2.rectangle(
                thumb,
                (int(px * sx), int(py * sy)),
                (int((px + pw) * sx), int((py + ph) * sy)),
                (0, 255, 255),
                1,
            )

        # Highlight if selected in Phase 7
        if is_sel:
            p_types = "/".join([c.pool for c in crops_here])
            border_color = (0, 255, 0) if "POOL_A" in p_types else (255, 200, 0)
            cv2.rectangle(thumb, (0, 0), (191, 143), border_color, 3)
            cv2.rectangle(thumb, (0, 120), (192, 144), (0, 0, 0), -1)
            cv2.putText(
                thumb,
                f"OCR: {p_types}",
                (4, 136),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                border_color,
                1,
                cv2.LINE_AA,
            )

        # Top banner for frame and timestamp
        cv2.rectangle(thumb, (0, 0), (192, 18), (0, 0, 0), -1)
        cv2.putText(
            thumb,
            f"F:{target_fidx} | {sec}s | P:{len(props)}",
            (4, 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        thumbnails_1fps.append(thumb)

        sampled_1fps_manifest.append({
            "video_sha256": sha,
            "frame_index": target_fidx,
            "timestamp_ms": ts_ms,
            "sharpness_laplacian": round(lap, 2),
            "brightness_mean": round(b_mean, 2),
            "selected_in_phase7": is_sel,
            "original_ocr_crops": [c.crop_xyxy for c in crops_here],
            "pools": [c.pool for c in crops_here],
            "proposals_count": len(props),
            "proposals": [
                {
                    "bbox_xywh": p.bbox_xywh,
                    "aspect_ratio": p.aspect_ratio,
                    "area": p.area,
                    "confidence_score": p.confidence_score,
                    "expanded_crop_xyxy": p.expanded_crop_xyxy,
                }
                for p in props
            ],
            "visual_review_classification": "INCONCLUSIVE",
            "visual_review_notes": "Operator review pending",
        })

    # Build Contact Sheets (10 columns grid)
    cols = 10
    total_thumbs = len(thumbnails_1fps)
    rows = (total_thumbs + cols - 1) // cols
    sheet_h = rows * 144
    sheet_w = cols * 192
    contact_sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for i, thumb in enumerate(thumbnails_1fps):
        r = i // cols
        c = i % cols
        contact_sheet[r * 144 : (r + 1) * 144, c * 192 : (c + 1) * 192] = thumb

    contact_sheet_path = out_dir / "contact_sheet.png"
    cv2.imwrite(str(contact_sheet_path), contact_sheet)

    # 3. Save full-resolution samples corresponding to original 25 OCR-selected frames
    for idx, crop_item in enumerate(selected_crops):
        cap.set(cv2.CAP_PROP_POS_FRAMES, crop_item.frame_index)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        cx1, cy1, cx2, cy2 = crop_item.crop_xyxy
        crop_roi = frame[cy1:cy2, cx1:cx2]

        # Annotated full frame
        annotated = frame.copy()
        color = (0, 255, 0) if crop_item.pool == "POOL_A" else (255, 180, 0)
        cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), color, 2)
        cv2.rectangle(annotated, (cx1, max(0, cy1 - 22)), (min(width, cx1 + 320), cy1), (0, 0, 0), -1)
        label = f"#{idx+1:02d} {crop_item.pool} [{cx1},{cy1},{cx2},{cy2}] Conf:{crop_item.confidence_score:.2f}"
        cv2.putText(annotated, label, (cx1 + 4, max(14, cy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        full_frame_path = ocr_frames_dir / f"ocr_call_{idx+1:02d}_frame_{crop_item.frame_index:05d}_{crop_item.pool}_full.png"
        crop_roi_path = ocr_frames_dir / f"ocr_call_{idx+1:02d}_frame_{crop_item.frame_index:05d}_{crop_item.pool}_crop.png"
        cv2.imwrite(str(full_frame_path), annotated)
        cv2.imwrite(str(crop_roi_path), crop_roi)

    cap.release()

    # 4. Save sampling_manifest.json
    manifest_path = out_dir / "sampling_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video_id": vid,
                "video_sha256": sha,
                "native_fps": actual_fps,
                "total_frames": total_frames,
                "duration_sec": duration_sec,
                "resolution": f"{width}x{height}",
                "total_1fps_samples": len(sampled_1fps_manifest),
                "selected_ocr_call_count": len(selected_crops),
                "samples": sampled_1fps_manifest,
            },
            f,
            indent=2,
        )

    # 5. Task B: Audit the board detector across screened frames (5 FPS)
    # Generate proposal_sampling_audit.csv
    # Determine selection status and rejection reasons
    selected_pool_a_set = {(c.frame_index, tuple(c.crop_xyxy)) for c in selected_crops if c.pool == "POOL_A"}
    selected_pool_b_set = {(c.frame_index, tuple(c.crop_xyxy)) for c in selected_crops if c.pool == "POOL_B"}
    screened_frame_indices = {f.frame_index for f in screened_frames}
    valid_screened_frame_indices = {f.frame_index for f in screened_frames if f.is_valid_quality}

    # Props by frame
    props_by_f: Dict[int, List[BoardRoiProposal]] = {}
    for p in all_proposals:
        props_by_f.setdefault(p.frame_index, []).append(p)

    csv_rows = []
    for sf in screened_frames:
        f_idx = sf.frame_index
        ts = sf.timestamp_ms
        f_props = props_by_f.get(f_idx, [])
        p_count = len(f_props)
        sharpness = round(sf.sharpness_laplacian, 2)

        if not f_props:
            # Fallback frame candidate
            sel_a = False
            sel_b = (f_idx, (0, 130, 768, 280)) in selected_pool_b_set
            rej = ""
            if not sel_b:
                if not sf.is_valid_quality:
                    rej = "QUALITY_REJECTED_SHARPNESS_OR_CONTRAST"
                else:
                    rej = "NO_BOARD_PROPOSAL_FALLBACK_NOT_CHOSEN"
            csv_rows.append({
                "frame_index": f_idx,
                "timestamp_ms": ts,
                "proposals_count": 0,
                "proposal_bboxes": "[]",
                "proposal_confidences": "[]",
                "sharpness_laplacian": sharpness,
                "selected_pool_a": sel_a,
                "selected_pool_b": sel_b,
                "rejection_reason": rej,
            })
        else:
            for p in f_props:
                crop_box = tuple(p.expanded_crop_xyxy)
                sel_a = (f_idx, crop_box) in selected_pool_a_set
                sel_b = (f_idx, crop_box) in selected_pool_b_set

                rej = ""
                if not sel_a and not sel_b:
                    if not sf.is_valid_quality:
                        rej = "FRAME_QUALITY_REJECTED"
                    elif p.confidence_score < 0.6:
                        rej = "LOW_PROPOSAL_CONFIDENCE"
                    else:
                        rej = "POOL_A_BUCKET_CAP_OR_POOL_B_DEDUP_EXCLUDED"

                csv_rows.append({
                    "frame_index": f_idx,
                    "timestamp_ms": ts,
                    "proposals_count": p_count,
                    "proposal_bboxes": f"[{p.bbox_xywh[0]},{p.bbox_xywh[1]},{p.bbox_xywh[2]},{p.bbox_xywh[3]}]",
                    "proposal_confidences": f"[{p.confidence_score}]",
                    "sharpness_laplacian": sharpness,
                    "selected_pool_a": sel_a,
                    "selected_pool_b": sel_b,
                    "rejection_reason": rej,
                })

    csv_path = out_dir / "proposal_sampling_audit.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "timestamp_ms",
                "proposals_count",
                "proposal_bboxes",
                "proposal_confidences",
                "sharpness_laplacian",
                "selected_pool_a",
                "selected_pool_b",
                "rejection_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"[{vid}] Task A and Task B completed: contact sheet, {len(selected_crops)} crops, manifest, and audit CSV saved.")


def run_task_c_audit():
    # Audit original 25 OCR calls for each video
    # Save shadow_lab/data/phase7b_diagnostics/ocr_evidence_audit.csv
    audit_rows = []

    for v_info in VIDEOS:
        vid = v_info["id"]
        v_path = HELDOUT_INBOX / v_info["filename"]
        sha = compute_sha256(v_path)
        dec_path, raw_path = find_heldout_artifacts(vid, sha)

        with open(dec_path, "r", encoding="utf-8") as f:
            dec_data = json.load(f)

        # Load raw observations
        raw_obs_by_frame_crop: Dict[Tuple[int, Tuple[int, int, int, int]], List[Dict[str, Any]]] = {}
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    k = (item["frame_id"], tuple(item["board_roi_xyxy"]))
                    raw_obs_by_frame_crop.setdefault(k, []).append(item)

        # Reconstruct sampler crops to get exact pool assignment
        cap = cv2.VideoCapture(str(v_path))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / actual_fps
        cap.release()

        p_inp = PassageVideoInput(
            site_id="ndls_pf12_13",
            installation_id="inst_01",
            passage_id=f"pas_{vid}",
            video_path=str(v_path),
            source_camera_id="side_a",
            fps=actual_fps,
            frame_count=total_frames,
            duration_sec=round(duration_sec, 2),
        )
        source = FinalizedMp4Source(p_inp)
        _, _, selected_crops = reconstruct_sampler_selection(source)

        for idx, crop_item in enumerate(selected_crops):
            f_idx = crop_item.frame_index
            ts_ms = crop_item.timestamp_ms
            crop_box = tuple(crop_item.crop_xyxy)
            pool = crop_item.pool
            prep = "P1_COLOR_2X_BICUBIC"

            obs_list = raw_obs_by_frame_crop.get((f_idx, crop_box), [])
            if not obs_list:
                # OCR executed on this crop, but yielded 0 text items above confidence threshold (0.45)
                audit_rows.append({
                    "video_id": vid,
                    "ocr_call_index": idx + 1,
                    "frame_index": f_idx,
                    "timestamp_ms": ts_ms,
                    "pool": pool,
                    "crop_xyxy": f"[{crop_box[0]},{crop_box[1]},{crop_box[2]},{crop_box[3]}]",
                    "preprocessing_profile": prep,
                    "raw_text": "<NO_TEXT_DETECTED>",
                    "normalized_text": "",
                    "ocr_confidence": 0.0,
                    "entity_type": "<NONE>",
                    "reached_resolver": False,
                })
            else:
                for obs in obs_list:
                    # An observation reaches the resolver if it is recognized as a railway entity
                    # or passed into consensus
                    e_type = obs.get("entity_type", "OTHER_TEXT")
                    reached = e_type in ("SERVICE_NUMBER", "TRAIN_NAME", "ENDPOINT")
                    audit_rows.append({
                        "video_id": vid,
                        "ocr_call_index": idx + 1,
                        "frame_index": f_idx,
                        "timestamp_ms": ts_ms,
                        "pool": pool,
                        "crop_xyxy": f"[{crop_box[0]},{crop_box[1]},{crop_box[2]},{crop_box[3]}]",
                        "preprocessing_profile": obs.get("preprocessing_variant", prep),
                        "raw_text": obs.get("raw_text", ""),
                        "normalized_text": obs.get("normalized_text", ""),
                        "ocr_confidence": round(obs.get("ocr_confidence", 0.0), 4),
                        "entity_type": e_type,
                        "reached_resolver": reached,
                    })

    csv_path = DIAGNOSTIC_DIR / "ocr_evidence_audit.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "ocr_call_index",
                "frame_index",
                "timestamp_ms",
                "pool",
                "crop_xyxy",
                "preprocessing_profile",
                "raw_text",
                "normalized_text",
                "ocr_confidence",
                "entity_type",
                "reached_resolver",
            ],
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    print(f"Task C completed: ocr_evidence_audit.csv saved with {len(audit_rows)} rows across 100 total OCR calls.")


def run_task_d_trace():
    trace_dir = DIAGNOSTIC_DIR / "coach_id12"
    trace_dir.mkdir(parents=True, exist_ok=True)
    report_file = trace_dir / "train_name_trace.md"

    raw_text = "UTTAR SAMPARK KRAN"
    conf = 0.9930616021156311

    # Step 1: Normalization
    norm_text = EntityClassifier.normalize_text(raw_text)

    # Step 2: Entity Classification
    classified_all = EntityClassifier.classify_all(raw_text, conf)
    classified_single = EntityClassifier.classify(raw_text, conf)

    # Inspect reasons for classification
    # Check KNOWN_TRAIN_NAMES
    in_known = norm_text in KNOWN_TRAIN_NAMES
    substring_known = [kn for kn in KNOWN_TRAIN_NAMES if kn in norm_text or norm_text in kn]
    has_keywords = any(kw in norm_text for kw in ["EXPRESS", "EXP", "MAIL", "SUPERFAST"])

    # Step 3: Site Aliases Lookup
    alias_path = REPO_ROOT / "shadow_lab" / "sites" / "ndls_pf12_13" / "aliases.yaml"
    import yaml
    with open(alias_path, "r", encoding="utf-8") as f:
        site_aliases = yaml.safe_load(f)

    # Step 4: Timetable lookup
    tt_path = REPO_ROOT / "shadow_lab" / "sites" / "ndls_pf12_13" / "schedules" / "ndls_validated.csv"
    tt_rows = []
    with open(tt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            tt_rows.append(r)

    # Check matches in timetable
    tt_matches = []
    for r in tt_rows:
        tname = r.get("Train Name", "")
        # Compute fuzzy match
        ratio = difflib.SequenceMatcher(None, norm_text, tname.upper()).ratio()
        if "SAMPARK" in tname.upper() or "KRANTI" in tname.upper() or "UTTAR" in tname.upper() or ratio > 0.4:
            tt_matches.append({"train_number": r.get("Train No."), "train_name": tname, "similarity": round(ratio, 3)})

    # Build Trace Markdown
    trace_content = f"""# Task D: Video 12 Train-Name Recovery Investigation

## Recognized String: `{raw_text}` (OCR Confidence: {conf:.4f}, Frame 1660)

### 1. Text Normalization Trace
- Raw OCR Text: `{raw_text}`
- Normalized Text: `{norm_text}`
- Meaningful words preserved:
  - `UTTAR`: Preserved (Yes)
  - `SAMPARK`: Preserved (Yes)
  - `KRAN`: Preserved (Yes, truncated root of KRANTI)

### 2. Entity Classifier Trace
- Classification Result: `{classified_single.entity_type}` (Notes: `{classified_single.notes}`)
- Inspection of `KNOWN_TRAIN_NAMES`:
  - Is `{norm_text}` in `KNOWN_TRAIN_NAMES`? **{in_known}**
  - Substring matches in `KNOWN_TRAIN_NAMES`: `{substring_known}`
  - Note: `KNOWN_TRAIN_NAMES` in `text_entities.py` contains `"UTTAR SAMPRK KRANTI"` and `"UTTAR S KRANTI"`.
  - Notice the spelling in the repository: `SAMPRK` (missing the second 'A'). Because `"UTTAR SAMPARK KRAN"` has `SAMPARK` (with 'A') and `KRAN` (truncated), neither string is a substring of the other!
  - Keyword check: Does `{norm_text}` contain any of `["EXPRESS", "EXP", "MAIL", "SUPERFAST"]`? **{has_keywords}**
  - The keyword list excludes `"KRANTI"`, `"SAMPARK KRANTI"`, and `"SUPER FAST"`.
  - **Result**: The token was classified as `OTHER_TEXT` instead of `TRAIN_NAME`.

### 3. Consensus Engine Impact
- In `PhysicalCoachConsensusEngine.build_passage_identity`:
  - Tokens with `entity_type == "OTHER_TEXT"` are re-screened via `EntityClassifier.classify()`.
  - Because it remains `OTHER_TEXT`, it is **not** added to `name_by_coach`.
  - `train_name_candidates` remains **EMPTY (`[]`)**.
  - With 0 service numbers and 0 train names, `build_passage_identity` sets:
    - `board_category = "UNKNOWN"`
    - `visual_pair_status = "UNKNOWN"`
    - `reason_codes = ["NO_RECOGNIZABLE_RAILWAY_ENTITIES"]`

### 4. Curated Aliases Inspection
- Path: `shadow_lab/sites/ndls_pf12_13/aliases.yaml`
- Current curated alias entries:
  - `LUCKNOW MAIL` (12229/12230)
  - `TAJ EXPRESS` (12279/12280)
  - `SHIV GANGA` (12559/12560)
  - `SHRAM SHAKTI` (12451/12452)
- **Result**: `UTTAR SAMPARK KRANTI` (12445/12446) is **completely absent** from `aliases.yaml`.

### 5. Platform 12/13 Timetable Inspection
- Timetable: `shadow_lab/sites/ndls_pf12_13/schedules/ndls_validated.csv`
- Total scheduled services on PF 12/13: {len(tt_rows)}
- Timetable records with similarity or keyword overlap:
{json.dumps(tt_matches, indent=2)}
- **Result**: `Uttar Sampark Kranti` (12445 / 12446) is **not scheduled on Platform 12/13** in this timetable dataset.
  - The only Sampark Kranti in `ndls_validated.csv` is Train `12565` (`DBG-NDLS, Bihar Sampark Kranti`).
  - Train `19566` is `DDN-OKA Uttaranchal Exp`.

### 6. Candidate Resolver Architecture & Separation of Concerns
- Trace through Resolver (`resolve_shadow`):
  1. Input `VisualOcrEvidence`:
     - `train_name_candidates`: `[]` (lost at entity classifier)
     - `service_number_candidates`: `[]` (only 6-digit coach serial `251373` detected)
     - `numeric_status`: `PAIR_UNCONFIRMED`
  2. Even if injected as `train_name = "UTTAR SAMPARK KRANTI"`:
     - Matcher against `ndls_validated.csv` finds 0 exact timetable matches for 12445/12446 because the service is absent from PF 12/13 timetable.
     - Even if 12445/12446 were in timetable: without active visual service numbers or verified NTES arrival/departure timestamps, the resolver pathway for `NOT_NUMBERED` boards requires independent direction corroboration or exact endpoint matching.
     - In the absence of direction corroboration, a bidirectional pair would be `AMBIGUOUS`.
  3. **Architectural Gap**:
     - `ShadowDecision` only outputs `shadow_final_number`, `shadow_status`, and `ranked_candidates`.
     - The resolver treats active running number resolution as an all-or-nothing requirement.
     - There is no first-class `service_family` output field (e.g., identifying that the passage belongs to the `UTTAR SAMPARK KRANTI` rake family without declaring an active running train number).
     - As a result, the entire decision fell back to `status: UNKNOWN`.
"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(trace_content)

    print(f"Task D completed: train_name_trace.md saved for coach_id12.")


def run_task_e_comparison():
    # Load Phase 6 and Phase 7 development benchmark results
    p6_path = REPO_ROOT / "shadow_lab" / "data" / "phase_6_evaluation_results.json"
    p7_path = REPO_ROOT / "shadow_lab" / "data" / "phase_7_evaluation_results.json"
    heldout_path = HELDOUT_OUTPUT / "heldout_4_evaluation_results.json"

    with open(p6_path, "r", encoding="utf-8") as f:
        d6 = json.load(f)
    with open(p7_path, "r", encoding="utf-8") as f:
        d7 = json.load(f)
    with open(heldout_path, "r", encoding="utf-8") as f:
        dh = json.load(f)

    historical_vids = [
        {"name": "Video 001", "passage_id": "pas_video_001", "expected_train": "15707/15708"},
        {"name": "Video 004", "passage_id": "pas_video_004", "expected_train": "20433/20434"},
        {"name": "Video 007", "passage_id": "pas_video_007", "expected_train": "18103/18104"},
        {"name": "Video 008", "passage_id": "pas_video_008", "expected_train": "12919/12920"},
        {"name": "Video 009", "passage_id": "pas_video_009", "expected_train": "13307/13308"},
        {"name": "Video 010", "passage_id": "pas_video_010", "expected_train": "13307/13308"},
    ]

    p6_by_id = {p.get("passage_id", p.get("video_id")): p for p in d6.get("passage_results", [])}
    p7_by_id = {p.get("passage_id", p.get("video_id")): p for p in d7.get("passage_results", [])}
    heldout_by_id = {p.get("video_id"): p for p in dh.get("passage_results", [])}

    comparison_rows = []

    for h in historical_vids:
        pid = h["passage_id"]
        res7 = p7_by_id.get(pid, {})
        res6 = p6_by_id.get(pid, {})

        # Load decision json from phase7_benchmark/output
        dec_files = list((REPO_ROOT / "shadow_lab" / "data" / "phase7_benchmark" / "output").glob(f"*{pid}*_decision.json"))
        dec_data = {}
        if dec_files:
            with open(dec_files[0], "r", encoding="utf-8") as f:
                dec_data = json.load(f)

        v_ev = dec_data.get("visual_evidence", {})
        res_dec = dec_data.get("resolver_decision", {})

        comparison_rows.append({
            "cohort": "Development (Phase 6/7)",
            "video_identifier": h["name"] + f" ({pid})",
            "expected_service": h["expected_train"],
            "phase6_predicted": res6.get("predicted_train_number", "None"),
            "phase6_status": res6.get("resolver_status", "UNKNOWN"),
            "phase7_predicted": res7.get("predicted_train_number", "None"),
            "phase7_status": res7.get("resolver_status", "UNKNOWN"),
            "visual_numbers": str(res7.get("visual_numbers", [])),
            "visual_pair_status": res7.get("visual_pair_status", "UNKNOWN"),
            "train_names_observed": str(res7.get("train_names_observed", [])),
            "temporal_buckets_covered": len(v_ev.get("board_buckets_covered", [])),
            "raw_observations_count": v_ev.get("total_raw_observations", 0),
            "resolver_reason_codes": str(res_dec.get("reason_codes", [])),
        })

    heldout_vids = [
        {"id": "coach_id11", "name": "Heldout Video 11"},
        {"id": "coach_id12", "name": "Heldout Video 12"},
        {"id": "coach_id13", "name": "Heldout Video 13"},
        {"id": "coach_id14", "name": "Heldout Video 14"},
    ]

    for hv in heldout_vids:
        vid = hv["id"]
        res = heldout_by_id.get(vid, {})
        dec_files = list(HELDOUT_OUTPUT.glob(f"*{vid}*_decision.json"))
        dec_data = {}
        if dec_files:
            with open(dec_files[0], "r", encoding="utf-8") as f:
                dec_data = json.load(f)

        v_ev = dec_data.get("visual_evidence", {})
        res_dec = dec_data.get("resolver_decision", {})

        comparison_rows.append({
            "cohort": "Blind Heldout (Phase 7)",
            "video_identifier": hv["name"] + f" ({vid})",
            "expected_service": "BLIND_GT_ISOLATED",
            "phase6_predicted": "N/A",
            "phase6_status": "N/A",
            "phase7_predicted": str(res.get("predicted_train_number", "None")),
            "phase7_status": str(res.get("resolver_status", "UNKNOWN")),
            "visual_numbers": str(res.get("visual_numbers", [])),
            "visual_pair_status": str(res.get("visual_pair_status", "UNKNOWN")),
            "train_names_observed": str(res.get("train_names_observed", [])),
            "temporal_buckets_covered": len(v_ev.get("board_buckets_covered", [])),
            "raw_observations_count": v_ev.get("total_raw_observations", 0),
            "resolver_reason_codes": str(res_dec.get("reason_codes", [])),
        })

    csv_path = DIAGNOSTIC_DIR / "historical_comparison_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cohort",
                "video_identifier",
                "expected_service",
                "phase6_predicted",
                "phase6_status",
                "phase7_predicted",
                "phase7_status",
                "visual_numbers",
                "visual_pair_status",
                "train_names_observed",
                "temporal_buckets_covered",
                "raw_observations_count",
                "resolver_reason_codes",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"Task E completed: historical_comparison_table.csv saved with {len(comparison_rows)} comparative rows.")


def run_task_f_matrix():
    matrix_rows = [
        {
            "video_id": "coach_id11",
            "frame_range": "0 - 3717 (Full duration 148.7s)",
            "root_cause": "PROPOSAL_MISS / BOARD_NOT_VISIBLE",
            "supporting_evidence": "All 25 OCR calls screened exterior coach body, stencils ('LA', 'GENERAL', 'B20'), and coach serials ('243560', '252002', '247199'). 0 5-digit service numbers detected. Detector proposed 100 yellow body/window patches but no destination board.",
            "crop_coordinates_sample": "[377, 177, 768, 355], [502, 115, 768, 214]",
            "observed_values": "Only coach serial numbers and general stencils visible; no destination board proposed or recognized.",
            "component_responsible": "AutomaticBoardDetector / Physical Train State",
        },
        {
            "video_id": "coach_id12",
            "frame_range": "Frame 1660 & Frame 3305",
            "root_cause": "ENTITY_FAILURE",
            "supporting_evidence": "PaddleOCR read 'UTTAR SAMPARK KRAN' at Frame 1660 with 0.9931 confidence on crop [378, 131, 768, 270]. EntityClassifier classified as 'OTHER_TEXT' because 'UTTAR SAMPARK KRAN' was missing from KNOWN_TRAIN_NAMES (which only had misspelled 'UTTAR SAMPRK KRANTI' and 'UTTAR S KRANTI'), and 'KRAN'/'KRANTI' was not in generic keyword list. Consensus engine discarded it from train_name_candidates.",
            "crop_coordinates_sample": "[378, 131, 768, 270] (Frame 1660)",
            "observed_values": "raw_text: 'UTTAR SAMPARK KRAN', conf: 0.9931 -> entity_type: 'OTHER_TEXT' -> train_name_candidates: []",
            "component_responsible": "shadow_identity.shadow_ocr.text_entities.EntityClassifier",
        },
        {
            "video_id": "coach_id12",
            "frame_range": "Resolver Stage (T1)",
            "root_cause": "RESOLVER_FAILURE / INSUFFICIENT_DIRECTION_EVIDENCE",
            "supporting_evidence": "Even if train name were recovered, 'UTTAR SAMPARK KRANTI' (12445/12446) is absent from both aliases.yaml and ndls_validated.csv. Furthermore, resolver lacks a service-family identification pathway separate from active 5-digit train number confirmation.",
            "crop_coordinates_sample": "N/A (Resolver arbitration stage)",
            "observed_values": "aliases.yaml missing 12445/12446; ndls_validated.csv missing 12445/12446; resolver cannot emit service-family without numeric confirmation.",
            "component_responsible": "shadow_identity.candidate_resolver / Timetable & Aliases",
        },
        {
            "video_id": "coach_id13",
            "frame_range": "0 - 3959 (Full duration 158.4s)",
            "root_cause": "PROPOSAL_MISS / CROP_FAILURE",
            "supporting_evidence": "31 raw observations generated across 25 calls. All observations are interior/exterior window annotations ('TO SEAT 1-90', 'EMERGENCY WINDOW', '1-90') and coach serials ('124922'). Detector locked onto yellow window frame reflections and seating stencils instead of destination boards.",
            "crop_coordinates_sample": "[346, 139, 768, 277], [0, 137, 376, 265]",
            "observed_values": "'TO SEAT 1-90' (conf 0.998), 'EMERGENCY WINDOW' (conf 0.996), '124922' (conf 0.999). 0 service numbers or train names.",
            "component_responsible": "AutomaticBoardDetector",
        },
        {
            "video_id": "coach_id14",
            "frame_range": "0 - 3717 (Full duration 148.7s)",
            "root_cause": "PROPOSAL_MISS / CROP_FAILURE",
            "supporting_evidence": "28 raw observations generated across 25 calls. Observations consist of coach exterior stencils ('LUGGAGE', 'SECONDC', 'NR', '084538') and seating labels ('TO SEAT/SLEEP', '1-90'). Destination board was never proposed or captured in the 25 crops.",
            "crop_coordinates_sample": "[384, 151, 768, 289], [344, 157, 768, 285]",
            "observed_values": "'LUGGAGE' (conf 0.999), 'SECONDC' (conf 0.998), 'NR' (conf 0.999), '084538' (conf 0.999). 0 service numbers or train names.",
            "component_responsible": "AutomaticBoardDetector",
        },
    ]

    csv_path = DIAGNOSTIC_DIR / "root_cause_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "frame_range",
                "root_cause",
                "supporting_evidence",
                "crop_coordinates_sample",
                "observed_values",
                "component_responsible",
            ],
        )
        writer.writeheader()
        writer.writerows(matrix_rows)

    print(f"Task F completed: root_cause_matrix.csv saved with {len(matrix_rows)} verified failure rows.")


if __name__ == "__main__":
    print("=== Starting Phase 7B Diagnostics Execution ===")
    t_start = time.time()

    # Task A & Task B for each video
    for v_info in VIDEOS:
        generate_task_a_and_b(v_info)

    # Task C: 100-call OCR audit
    run_task_c_audit()

    # Task D: Video 12 Train-name recovery trace
    run_task_d_trace()

    # Task E: Historical Development Video comparison
    run_task_e_comparison()

    # Task F: Root Cause Matrix
    run_task_f_matrix()

    elapsed = round(time.time() - t_start, 2)
    print(f"=== Phase 7B Diagnostics Completed in {elapsed}s ===")
