"""
Phase 7 — Single-Shot Blind Held-Out Evaluation Script (heldout_4).

Executes the frozen ShadowOCR implementation on 4 new independent held-out passage recordings
(coach_id11.mp4 through coach_id14.mp4):
1. Strict Blind Protocol:
   - Zero pre-inspection or parameter tuning on heldout videos.
   - Zero ground-truth leakage into OCR, frame selection, candidate scoring, or resolver.
2. Frozen Algorithm Configuration:
   - Screening FPS: 5.0
   - Sampling: Balanced Two-Pool Selection:
       * Pool A: up to 15 OCR crop calls for high-quality board proposals with temporal diversity (max 2 per 2s bucket)
       * Pool B: up to 10 OCR crop calls for highest-quality unused crops with spatial/temporal deduplication
       * Total actual OCR call budget: exactly 25 maximum calls, never exceeded.
   - Engine: PaddleOCR PP-OCRv6_medium (textline orientation)
   - Preprocessing: P1_COLOR_2X_BICUBIC
   - Timetable: ndls_validated.csv (Platform 12/13 canonical)
   - NTES Evidence: HONEST_ABSENT (no synthetic substitution)
   - Authoritative finalization required (.done / sidecar) for all 4 videos before running.
3. Objective Metric Separation:
   - Exact active-service match ONLY where independently verified single-number label exists.
   - Bidirectional pairs evaluate pair_membership_match and service_family_match.
   - Separate accounting of abstentions, incorrect provisionals, and false confirmations.
   - 4 held-out results compared separately from 10 development-video baseline.
4. Production Safety:
   - Pre- and post-run ARDS integrity check.
   - Zero production writes. Deployment stays OFF.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import psutil

repo_root = Path(__file__).resolve().parents[2]
src_dir = repo_root / "shadow_lab" / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from shadow_identity.shadow_ocr.worker import ShadowWorker, compute_file_sha256


def verify_production_files(
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Authoritative structured integrity verification of production ARDS files."""
    scripts_dir = repo_root / "shadow_lab" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from verify_originals_unchanged import verify_originals_structured
        manifest = manifest_path or str(repo_root / "shadow_lab" / "workspace" / "fixtures" / "original_hashes.json")
        return verify_originals_structured(manifest)
    except Exception as e:
        return {
            "passed": False,
            "total_expected": 0,
            "verified_count": 0,
            "missing_count": 0,
            "mismatch_count": 0,
            "error": str(e),
        }


def exact_pair_membership(
    pred_num: Optional[str],
    pair_members: Optional[List],
    verified: bool,
) -> Optional[bool]:
    """Deterministic service-pair membership check.

    Returns:
        True  — pred_num is an exact member of verified pair.
        False — pred_num is absent or not in verified pair.
        None  — pair not verified or missing, result is indeterminate.
    """
    if not verified or not pair_members:
        return None

    members = {str(n).strip() for n in pair_members if n is not None}

    if not members:
        return None

    if pred_num is None:
        return False

    return str(pred_num).strip() in members


def ensure_fresh_benchmark(output_dir: Path, storage_dir: Path) -> None:
    """Fail closed: never reuse or overwrite any earlier blind-run evidence/state."""
    if any(output_dir.iterdir()) or any(storage_dir.iterdir()):
        raise RuntimeError("EXISTING_BLIND_BENCHMARK_EVIDENCE: refuse second run")


def get_video_info(mp4_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return {"error": "unreadable"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = round(frame_count / fps, 2) if fps > 0 else 0.0
    cap.release()
    return {
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration,
        "size_bytes": mp4_path.stat().st_size,
    }


def main():
    print("=" * 80)
    print("PHASE 7: SINGLE-SHOT BLIND HELDOUT EVALUATION (heldout_4)")
    print("=" * 80)

    # 1. Pre-Run Production Integrity Check
    print("\n[STEP 1] Pre-Run Production ARDS Integrity Check...")
    pre_integrity = verify_production_files()
    if not pre_integrity.get("passed", False):
        print("[FATAL] Original ARDS files modified! Aborting evaluation.")
        sys.exit(1)
    print(f"[PASS] {pre_integrity.get('verified_count', 0)} / {pre_integrity.get('total_expected', 0)} ARDS files untouched.")

    # 2. Setup Held-Out Directories
    heldout_dir = repo_root / "shadow_lab" / "data" / "heldout_4"
    inbox_dir = heldout_dir / "inbox"
    manifest_file = heldout_dir / "manifest" / "ground_truth_manifest.json"
    output_dir = heldout_dir / "output"
    storage_dir = heldout_dir / "storage"

    inbox_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "heldout_4_evaluation_results.json"
    try:
        ensure_fresh_benchmark(output_dir, storage_dir)
    except RuntimeError:
        print("[FATAL] Existing benchmark outputs or worker state detected; refusing to overwrite first-run evidence.")
        print("To protect initial blind evaluation provenance, outputs cannot be silently overwritten.")
        sys.exit(1)

    # 3. Validation: Exactly 4 recordings required under approved offline-replay mode
    print("\n[STEP 2] Inspecting Held-Out Inbox for Required 4 Historical Recordings...")
    expected_filenames = [
        "coach_id11.mp4",
        "coach_id12.mp4",
        "coach_id13.mp4",
        "coach_id14.mp4",
    ]

    missing_recordings = []
    unreadable_recordings = []
    valid_recordings = []

    for fname in expected_filenames:
        mp4_p = inbox_dir / fname
        if not mp4_p.is_file():
            missing_recordings.append(fname)
            continue

        if mp4_p.stat().st_size == 0:
            unreadable_recordings.append(f"{fname} (empty file)")
            continue

        cap = cv2.VideoCapture(str(mp4_p))
        if not cap.isOpened():
            unreadable_recordings.append(f"{fname} (cv2 decode failed)")
            continue
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            unreadable_recordings.append(f"{fname} (first frame decode failed)")
            continue

        valid_recordings.append(mp4_p)

    if missing_recordings or unreadable_recordings or len(valid_recordings) != 4:
        print("  - [BLOCKED_DATA] Held-out evaluation requires all 4 finalized historical recordings:")
        print(f"    Expected: {expected_filenames}")
        if missing_recordings:
            print(f"    Missing MP4 files ({len(missing_recordings)}): {missing_recordings}")
        if unreadable_recordings:
            print(f"    Unreadable / corrupt MP4 files ({len(unreadable_recordings)}): {unreadable_recordings}")
        print(f"    Currently valid and verified: {len(valid_recordings)} / 4")
        print("  - Benchmark will NOT execute a partial set. Aborting inference until all 4 recordings are verified.")
        return 0

    print(f"  + All 4 required held-out recordings verified under offline-replay ingestion mode:")
    for v in valid_recordings:
        print(f"    * {v.name} ({v.stat().st_size / (1024*1024):.1f} MB) [HISTORICAL_FINALIZED]")

    # 4. Blind Execution with Frozen ShadowWorker (GT ISOLATED)
    print("\n[STEP 3] Executing Frozen Pipeline (Strict Blind Mode)...")
    print("  * Ground truth manifest is STRICTLY ISOLATED from execution.")
    print("  * Ingestion Mode: Approved offline-replay on shadow-owned inbox (no fabricated ARDS markers).")
    print("  * Frozen Parameters: 5.0 FPS screening, balanced two-pool budget (Pool A: max 15, Pool B: max 10, total <= 25).")

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    worker = ShadowWorker(
        inbox_dir=inbox_dir,
        storage_dir=storage_dir,
        output_dir=output_dir,
        site_id="ndls_pf12_13",
        camera_id="side_a",
        timetable_path=str(repo_root / "shadow_lab" / "sites" / "ndls_pf12_13" / "schedules" / "ndls_validated.csv"),
        ground_truth_dir=None,  # STRICT BLIND EXECUTION: NO GT PROVIDED TO WORKER
        max_concurrency=1,
        require_finalization_signal=True,
        allow_unmarked_offline_replay=True,
        operator_replay_workspace_root=heldout_dir,
        size_stability_interval_sec=0.0,
        screening_fps=5.0,
        max_ocr_budget=25,
    )

    t_start = time.time()
    cycle_stats = worker.run_cycle()
    elapsed = time.time() - t_start
    mem_peak = process.memory_info().rss / (1024 * 1024)

    print(f"\n[STEP 4] Benchmark Completed in {elapsed:.2f}s")
    print(f"  - Discovered: {cycle_stats['discovered']}")
    print(f"  - Processed: {cycle_stats['processed']}")
    print(f"  - Failed: {cycle_stats['failed']}")

    # A single-shot benchmark must not claim success on missing/failed jobs.
    if cycle_stats.get("processed") != 4 or cycle_stats.get("failed") or cycle_stats.get("discovered") != 4:
        print("[FATAL] Worker did not complete exactly four held-out passages. Results remain unevaluated.")
        raise RuntimeError("INCOMPLETE_HELDOUT_WORKER_CYCLE")

    # 5. Post-Run Truth-Isolated Evaluation Against Manifest
    print("\n[STEP 5] Post-Run Evaluation Against Read-Only Manifest...")
    manifest_data: Dict[str, Any] = {}
    if manifest_file.is_file():
        with open(manifest_file, "r", encoding="utf-8") as mf:
            manifest_data = json.load(mf)

    recordings_manifest = {
        r.get("expected_filename", ""): r for r in manifest_data.get("recordings", [])
    }

    decision_files = list(output_dir.glob("*_decision.json"))
    raw_files = list(output_dir.glob("*_raw.jsonl"))

    passage_reports: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for v_path in valid_recordings:
        v_stem = v_path.stem
        # Locate corresponding decision and raw files
        # Exact passage token boundary prevents substring/mismatched-file selection.
        matched_decisions = [f for f in decision_files if f.name.endswith(f"_{v_stem}_" + compute_file_sha256(v_path) + "_decision.json")]
        matched_raw = [f for f in raw_files if f.name.endswith(f"_{v_stem}_" + compute_file_sha256(v_path) + "_raw.jsonl")]
        # Legacy worker IDs may contain passage prefixes; require unique exact stem and SHA.
        if not matched_decisions:
            matched_decisions = [f for f in decision_files if f"pas_{v_stem}_" in f.name and compute_file_sha256(v_path) in f.name]
        if not matched_raw:
            matched_raw = [f for f in raw_files if f"pas_{v_stem}_" in f.name and compute_file_sha256(v_path) in f.name]
        if len(matched_decisions) != 1 or len(matched_raw) != 1:
            raise RuntimeError(f"Missing or ambiguous durable evidence for {v_path.name}: decisions={len(matched_decisions)}, raw={len(matched_raw)}")
        dec_file, raw_file = matched_decisions[0], matched_raw[0]
        v_info = get_video_info(v_path)

        dec_data: Dict[str, Any] = {}
        if dec_file and dec_file.is_file():
            with open(dec_file, "r", encoding="utf-8") as df:
                dec_data = json.load(df)

        raw_obs_count = 0
        if raw_file and raw_file.is_file():
            with open(raw_file, "r", encoding="utf-8") as rf:
                raw_obs_count = sum(1 for line in rf if line.strip())

        v_ev = dec_data.get("visual_evidence", {})
        r_dec = dec_data.get("resolver_decision", {})
        perf = dec_data.get("performance_latency", {})
        lat = perf.get("total_wall_clock_s", 0.0)
        if lat > 0:
            latencies.append(lat)

        pred_num = r_dec.get("shadow_final_number")
        res_status = r_dec.get("shadow_status", "UNKNOWN")

        # Cross-reference with manifest record
        mf_rec = recordings_manifest.get(v_path.name, {})
        gt_num = mf_rec.get("ground_truth_number")
        gt_name = mf_rec.get("ground_truth_name")
        is_single = mf_rec.get("is_single_active_service", False)
        direction_quality = mf_rec.get("direction_label_quality", "PENDING_VERIFICATION")
        service_pair = mf_rec.get("service_pair", [])
        service_family = mf_rec.get("service_family")

        # P0.3 Metrics
        exact_active_match = None
        pair_membership_match = None
        service_family_match = None

        pair_verified = bool(
            mf_rec.get("verified", False)
            or (direction_quality not in ("PENDING_VERIFICATION", "UNLABELED") and direction_quality is not None)
        )

        if gt_num:
            if is_single and pair_verified:
                exact_active_match = (str(pred_num).strip() == str(gt_num).strip()) if pred_num else False
            else:
                exact_active_match = None

        pair_members_to_check = service_pair if service_pair else ([gt_num] if gt_num else [])
        pair_membership_match = exact_pair_membership(pred_num, pair_members_to_check, verified=pair_verified)
        # Pair membership is NOT equivalent to independently verified service-family mapping.
        # No explicit curated family-membership mapping is part of this manifest schema.
        service_family_match = None

        abstained = (pred_num is None or res_status in ("UNKNOWN", "AMBIGUOUS"))
        incorrect_provisional = (
            res_status == "PROVISIONAL"
            and pred_num is not None
            and (pair_membership_match is False or (exact_active_match is False and pair_membership_match is None))
        )
        false_confirmed = (
            res_status == "CONFIRMED"
            and pred_num is not None
            and (pair_membership_match is False or (exact_active_match is False and pair_membership_match is None))
        )

        passage_reports.append({
            "video_id": v_stem,
            "filename": v_path.name,
            "video_duration_sec": v_info.get("duration_sec"),
            "frame_count": v_info.get("frame_count"),
            "resolution": f"{v_info.get('width')}x{v_info.get('height')}",
            "sha256": compute_file_sha256(v_path),
            "ground_truth_train_number": gt_num,
            "ground_truth_train_name": gt_name,
            "direction_label_quality": direction_quality,
            "predicted_train_number": pred_num,
            "visual_pair_status": v_ev.get("visual_pair_status"),
            "visual_numbers": v_ev.get("stable_pair", []),
            "visual_endpoints": v_ev.get("endpoint_candidates", []),
            "train_names_observed": [c.get("name") for c in v_ev.get("train_name_candidates", [])],
            "resolver_status": res_status,
            "reason_codes": r_dec.get("reason_codes", []),
            "candidate_regions_screened": v_ev.get("evaluated_frames_count", 0),
            "ocr_calls_executed": v_ev.get("ocr_calls_executed", 0),
            "board_buckets_covered": v_ev.get("board_buckets_covered", []),
            "unique_frame_count": v_ev.get("unique_frame_count", 0),
            "fallback_calls": v_ev.get("fallback_calls", 0),
            "raw_observations_persisted": raw_obs_count,
            "pipeline_latency_sec": lat,
            "exact_active_service_match": exact_active_match,
            "pair_membership_match": pair_membership_match,
            "service_family_match": service_family_match,
            "abstained": abstained,
            "incorrect_provisional": incorrect_provisional,
            "false_confirmed": false_confirmed,
        })

    # 6. Aggregate Output Assembly
    total_eval = len(passage_reports)
    exact_active_evaluated = [r for r in passage_reports if r.get("exact_active_service_match") is not None]
    exact_active_matches = sum(1 for r in exact_active_evaluated if r.get("exact_active_service_match") is True)
    pair_member_evaluated = [r for r in passage_reports if r.get("pair_membership_match") is not None]
    pair_member_matches = sum(1 for r in pair_member_evaluated if r.get("pair_membership_match") is True)
    family_matches = sum(1 for r in passage_reports if r.get("service_family_match") is True)
    abstain_count = sum(1 for r in passage_reports if r.get("abstained") is True)
    incorrect_provisional_count = sum(1 for r in passage_reports if r.get("incorrect_provisional") is True)
    false_confirmed_count = sum(1 for r in passage_reports if r.get("false_confirmed") is True)

    latencies_sorted = sorted(latencies) if latencies else [0.0]
    p50 = float(np.percentile(latencies_sorted, 50))
    p90 = float(np.percentile(latencies_sorted, 90))
    p95 = float(np.percentile(latencies_sorted, 95))

    results_data: Dict[str, Any] = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "evaluation_version": "phase_7_heldout_4_initial",
        "dataset": "heldout_4",
        "total_passages_evaluated": total_eval,
        "exact_active_service_match_count": exact_active_matches,
        "exact_active_service_evaluated_count": len(exact_active_evaluated),
        "exact_active_service_accuracy_pct": round(exact_active_matches / len(exact_active_evaluated) * 100.0, 1) if exact_active_evaluated else None,
        "pair_membership_match_count": pair_member_matches,
        "pair_membership_evaluated_count": len(pair_member_evaluated),
        "pair_membership_accuracy_pct": round(pair_member_matches / len(pair_member_evaluated) * 100.0, 1) if pair_member_evaluated else None,
        "service_family_match_count": family_matches,
        "service_family_accuracy_pct": None,
        "accuracy_status": "INSUFFICIENT_DATA" if not exact_active_evaluated else "LABELED_SUBSET_ONLY",
        "abstain_count": abstain_count,
        "abstain_pct": round(abstain_count / total_eval * 100.0, 1) if total_eval > 0 else 0.0,
        "incorrect_provisional_count": incorrect_provisional_count,
        "false_confirmation_count": false_confirmed_count,
        "latency_percentiles_sec": {
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "p95": round(p95, 2),
            "mean": round(float(np.mean(latencies_sorted)), 2),
            "min": round(min(latencies_sorted), 2),
            "max": round(max(latencies_sorted), 2),
        },
        "resource_consumption": {
            "peak_rss_mb": round(mem_peak, 1),
            "total_benchmark_elapsed_sec": round(elapsed, 2),
        },
        "settings": {
            "screening_fps": 5.0,
            "max_ocr_budget": 25,
            "pool_a_budget": 15,
            "pool_b_budget": 10,
            "ocr_model": "PaddleOCR PP-OCRv6_medium",
            "preprocessing_profile": "P1_COLOR_2X_BICUBIC",
            "sampling_method": "balanced_two_pool_selection",
            "timetable": "ndls_validated.csv",
            "ntes_evidence": "HONEST_ABSENT",
            "finalization_signal_required": True,
        },
        "passage_results": passage_reports,
    }

    results_path = output_dir / "heldout_4_evaluation_results.json"
    # Exclusive-create prevents replacement even if another process raced the preflight.
    with open(results_path, "x", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"\n[STEP 6] Results durably written to {results_path}")

    # 7. Post-Run Production Integrity Check
    print("\n[STEP 7] Post-Run Production ARDS Integrity Check...")
    post_integrity = verify_production_files()
    if not post_integrity.get("passed", False):
        print("[FATAL] Original ARDS files modified post-benchmark!")
        sys.exit(1)
    print(f"[PASS] {post_integrity.get('verified_count', 0)} / {post_integrity.get('total_expected', 0)} ARDS files untouched.")
    print("\nHeld-out benchmark completed successfully.")


if __name__ == "__main__":
    main()
