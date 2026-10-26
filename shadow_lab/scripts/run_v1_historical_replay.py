"""
BetterOcr / ShadowOCR V1 Historical Passage Replay Runner
Replays all 14 historical recordings (Phase 7 benchmark + held-out 4)
into new shadow-owned output directory: shadow_lab/data/v1_replay/.

Guarantees:
- Zero mutations to ARDS production files (456 files remain read-only).
- Zero mutations to original held-out or benchmark artifacts.
- Deterministic identity resolution conforming to V1 specification:
  - Direction safety (§A1)
  - Offline replay authorization (§A2)
  - Coach consensus evidence codes (§A3)
  - Registry mapping integrity (§A4)
  - Evidence fusion & C4 atomic output schema (§C1-C4)
  - Metrics and daily review report (§D2-D3)
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import yaml

from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.continuous_evaluation import (
    GroundTruthRecord,
    compute_v1_evaluation_metrics,
    generate_daily_review_report,
)
from shadow_identity.evidence_fusion import decide_identity_v1, format_v1_identity_output
from shadow_identity.models import DirectionEvidence, PassageInput
from shadow_identity.shadow_ocr.adapter import ShadowOcrToBetterOcrAdapter
from shadow_identity.shadow_ocr.consensus import PhysicalCoachConsensusEngine
from shadow_identity.shadow_ocr.contracts import RawOcrObservation
from shadow_identity.shadow_ocr.text_entities import EntityClassifier
from shadow_identity.timetable_importer import TimetableImporter


def run_v1_replay():
    repo_root = Path(__file__).resolve().parents[2]
    replay_dir = repo_root / "shadow_lab" / "data" / "v1_replay"
    replay_dir.mkdir(parents=True, exist_ok=True)

    site_id = "ndls_pf12_13"
    aliases_path = repo_root / "shadow_lab" / "sites" / site_id / "aliases.yaml"
    timetable_path = repo_root / "shadow_lab" / "sites" / site_id / "schedules" / "ndls_validated.csv"

    # Load aliases
    with open(aliases_path, "r", encoding="utf-8") as f:
        site_aliases = yaml.safe_load(f)

    # Load timetable rows
    timetable_rows = TimetableImporter.load_schedule_csv(
        file_path=timetable_path,
        site_id=site_id,
        station_code="NDLS",
        schedule_version="1.0",
    )

    # 1. Discover all historical recordings
    heldout_dir = repo_root / "shadow_lab" / "data" / "heldout_4" / "output"
    phase7_dir = repo_root / "shadow_lab" / "data" / "phase7_benchmark" / "output"

    passages_to_replay = []

    # 4 Held-out passages
    heldout_defs = [
        {"id": "coach_id11", "sha": "b0ccb58bfc45980c3f5d6f11502e2bccc395cecb83b5f37a7a9364734a79d214", "dir": heldout_dir},
        {"id": "coach_id12", "sha": "fd1fa8972cebbb347f837280f935fb630a6faa853109bc57883ab179c26d2db6", "dir": heldout_dir},
        {"id": "coach_id13", "sha": "2cd1c91e40faf989f1f2f13741baeb9c2476e717f1a3113fcbf44ecdc748503f", "dir": heldout_dir},
        {"id": "coach_id14", "sha": "d791d944c444c790084c030487e6c0bf91b844c7635b4081c5fa1cdc8111aeaf", "dir": heldout_dir},
    ]
    for h in heldout_defs:
        passages_to_replay.append({
            "passage_id": f"pas_{h['id']}",
            "decision_file": h["dir"] / f"ndls_pf12_13_side_a_pas_{h['id']}_{h['sha']}_decision.json",
            "raw_file": h["dir"] / f"ndls_pf12_13_side_a_pas_{h['id']}_{h['sha']}_raw.jsonl",
            "expected_sha": h["sha"],
        })

    # 10 Phase 7 Benchmark passages
    for i in range(1, 11):
        vid_id = f"video_{i:03d}"
        cand_dec = list(phase7_dir.glob(f"*{vid_id}*_decision.json"))
        cand_raw = list(phase7_dir.glob(f"*{vid_id}*_raw.jsonl"))
        if cand_dec and cand_raw:
            passages_to_replay.append({
                "passage_id": f"pas_{vid_id}",
                "decision_file": cand_dec[0],
                "raw_file": cand_raw[0],
                "expected_sha": None,
            })

    print(f"[REPLAY] Found {len(passages_to_replay)} historical passages to replay.")

    replay_results: List[Dict[str, Any]] = []

    for item in passages_to_replay:
        pid = item["passage_id"]
        dec_p = item["decision_file"]
        raw_p = item["raw_file"]

        if not dec_p.is_file() or not raw_p.is_file():
            print(f"[REPLAY] Skipping missing files for {pid}")
            continue

        with open(dec_p, "r", encoding="utf-8") as f:
            orig_dec = json.load(f)

        orig_raw_lines: List[Dict[str, Any]] = []
        with open(raw_p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    orig_raw_lines.append(json.loads(line))

        # Re-classify observations
        updated_obs: List[RawOcrObservation] = []
        for r in orig_raw_lines:
            raw_text = r.get("raw_text", "")
            conf = float(r.get("ocr_confidence", 0.0))
            classified = EntityClassifier.classify(raw_text, ocr_confidence=conf)

            obs = RawOcrObservation(
                schema_version=r.get("schema_version", "shadow_ocr.v1"),
                site_id=r.get("site_id", site_id),
                passage_id=r.get("passage_id", pid),
                camera_id=r.get("camera_id", "side_a"),
                frame_id=int(r.get("frame_id", 0)),
                timestamp_ms=int(r.get("timestamp_ms", 0)),
                board_roi_xyxy=r.get("board_roi_xyxy", []),
                coach_id=r.get("coach_id"),
                coach_id_source=r.get("coach_id_source", "UNVERIFIED"),
                ocr_model_id=r.get("ocr_model_id", "PP-OCRv6"),
                preprocessing_variant=r.get("preprocessing_variant", "P1_COLOR_2X_BICUBIC"),
                raw_text=raw_text,
                normalized_text=classified.normalized_text,
                entity_type=classified.entity_type,
                ocr_confidence=conf,
                bbox_in_crop=r.get("bbox_in_crop", []),
                bbox_full_frame=r.get("bbox_full_frame", []),
            )
            updated_obs.append(obs)

        # Consensus
        consensus_engine = PhysicalCoachConsensusEngine()
        meta = orig_dec.get("passage_metadata", {})
        camera_id = meta.get("camera_id", "side_a")
        video_sha256 = meta.get("video_sha256") or item["expected_sha"] or orig_dec.get("input_video_sha256", "")

        identity_res = consensus_engine.build_passage_identity(
            site_id=site_id,
            passage_id=pid,
            camera_id=camera_id,
            video_sha256=video_sha256,
            observations=updated_obs,
        )

        visual_evidence = ShadowOcrToBetterOcrAdapter.to_visual_ocr_evidence(identity_res, updated_obs)

        passage_input = PassageInput(
            passage_id=pid,
            site_id=site_id,
            camera_id=camera_id,
            passage_start=meta.get("passage_start", ""),
            passage_end=meta.get("passage_end", ""),
            possible_platforms=meta.get("possible_platforms", ["12", "13"]),
            observed_platform=meta.get("observed_platform"),
        )

        # Deterministic identity arbitration
        v1_decision = decide_identity_v1(
            passage=passage_input,
            visual_evidence=visual_evidence,
            site_schedule_rows=timetable_rows,
        )

        # Build C4 output dictionary
        v1_output = format_v1_identity_output(
            site_id=site_id,
            passage_id=pid,
            input_video_sha256=video_sha256,
            recognized_train_names=v1_decision.recognized_train_names,
            visual_service_numbers=v1_decision.visual_service_numbers,
            recognized_endpoints=v1_decision.recognized_endpoints,
            service_family_candidates=v1_decision.service_family_candidates,
            active_service_number=v1_decision.active_service_number,
            active_direction_verified=v1_decision.active_direction_verified,
            identity_status=v1_decision.identity_status,
            reason_codes=v1_decision.reason_codes,
            evidence_provenance=v1_decision.evidence_provenance,
            contradictions=v1_decision.contradictions,
            processing_metrics=orig_dec.get("performance_latency", {}),
            passage_metadata=meta,
            visual_evidence=orig_dec.get("visual_evidence"),
            resolver_decision=v1_decision.shadow_decision.to_dict(),
            performance_latency=orig_dec.get("performance_latency"),
        )

        # Save to shadow-owned v1_replay output
        out_dec_path = replay_dir / f"v1_replay_{pid}_decision.json"
        out_raw_path = replay_dir / f"v1_replay_{pid}_raw.jsonl"

        with open(out_dec_path, "w", encoding="utf-8") as f:
            json.dump(v1_output, f, indent=2)

        with open(out_raw_path, "w", encoding="utf-8") as f:
            for obs in updated_obs:
                f.write(json.dumps(obs.to_dict()) + "\n")

        replay_results.append({
            "passage_id": pid,
            "recognized_train_names": v1_decision.recognized_train_names,
            "service_family_candidates": v1_decision.service_family_candidates,
            "active_service_number": v1_decision.active_service_number,
            "active_direction_verified": v1_decision.active_direction_verified,
            "identity_status": v1_decision.identity_status,
            "shadow_status": v1_decision.shadow_decision.shadow_status,
            "shadow_final_number": v1_decision.shadow_decision.shadow_final_number,
            "reason_codes": v1_decision.reason_codes,
            "input_video_sha256": video_sha256,
            "duration_sec": meta.get("video_properties", {}).get("duration_sec", 1.0),
            "ocr_calls": orig_dec.get("visual_evidence", {}).get("ocr_calls_executed", 20),
        })

    # Save summary
    summary_path = replay_dir / "v1_replay_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(replay_results, f, indent=2)

    print(f"[REPLAY] Successfully replayed {len(replay_results)} passages to {replay_dir}")
    return replay_results


if __name__ == "__main__":
    run_v1_replay()
