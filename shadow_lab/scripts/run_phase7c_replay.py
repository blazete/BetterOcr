"""
Phase 7C Isolated Replay & Comparison Runner
Replays persisted Phase 7 OCR observations using the revised Phase 7C logic.
Does NOT modify any Phase 7 benchmark output or production ARDS files.
Saves all replay artifacts to shadow_lab/data/phase7c_replay/.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.models import PassageInput
from shadow_identity.shadow_ocr.adapter import ShadowOcrToBetterOcrAdapter
from shadow_identity.shadow_ocr.consensus import PhysicalCoachConsensusEngine
from shadow_identity.shadow_ocr.contracts import RawOcrObservation
from shadow_identity.shadow_ocr.text_entities import EntityClassifier
from shadow_identity.timetable_importer import TimetableImporter


def run_phase7c_replay():
    repo_root = Path(__file__).resolve().parents[2]
    heldout_dir = repo_root / "shadow_lab" / "data" / "heldout_4" / "output"
    replay_dir = repo_root / "shadow_lab" / "data" / "phase7c_replay"
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

    videos = [
        {
            "id": "11",
            "name": "coach_id11",
            "decision_file": "ndls_pf12_13_side_a_pas_coach_id11_b0ccb58bfc45980c3f5d6f11502e2bccc395cecb83b5f37a7a9364734a79d214_decision.json",
            "raw_file": "ndls_pf12_13_side_a_pas_coach_id11_b0ccb58bfc45980c3f5d6f11502e2bccc395cecb83b5f37a7a9364734a79d214_raw.jsonl",
            "passage_id": "pas_coach_id11",
        },
        {
            "id": "12",
            "name": "coach_id12",
            "decision_file": "ndls_pf12_13_side_a_pas_coach_id12_fd1fa8972cebbb347f837280f935fb630a6faa853109bc57883ab179c26d2db6_decision.json",
            "raw_file": "ndls_pf12_13_side_a_pas_coach_id12_fd1fa8972cebbb347f837280f935fb630a6faa853109bc57883ab179c26d2db6_raw.jsonl",
            "passage_id": "pas_coach_id12",
        },
        {
            "id": "13",
            "name": "coach_id13",
            "decision_file": "ndls_pf12_13_side_a_pas_coach_id13_2cd1c91e40faf989f1f2f13741baeb9c2476e717f1a3113fcbf44ecdc748503f_decision.json",
            "raw_file": "ndls_pf12_13_side_a_pas_coach_id13_2cd1c91e40faf989f1f2f13741baeb9c2476e717f1a3113fcbf44ecdc748503f_raw.jsonl",
            "passage_id": "pas_coach_id13",
        },
        {
            "id": "14",
            "name": "coach_id14",
            "decision_file": "ndls_pf12_13_side_a_pas_coach_id14_d791d944c444c790084c030487e6c0bf91b844c7635b4081c5fa1cdc8111aeaf_decision.json",
            "raw_file": "ndls_pf12_13_side_a_pas_coach_id14_d791d944c444c790084c030487e6c0bf91b844c7635b4081c5fa1cdc8111aeaf_raw.jsonl",
            "passage_id": "pas_coach_id14",
        },
    ]

    replay_results: List[Dict[str, Any]] = []

    for v in videos:
        decision_path = heldout_dir / v["decision_file"]
        raw_path = heldout_dir / v["raw_file"]

        # 1. Load original Phase 7 decision
        with open(decision_path, "r", encoding="utf-8") as f:
            orig_dec = json.load(f)
        import copy
        orig_dec_saved = copy.deepcopy(orig_dec)

        # 2. Load original raw observations
        orig_raw_lines: List[Dict[str, Any]] = []
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    orig_raw_lines.append(json.loads(line))

        # 3. Re-classify raw observations using revised EntityClassifier
        updated_obs: List[RawOcrObservation] = []
        updated_raw_dicts: List[Dict[str, Any]] = []

        for r in orig_raw_lines:
            raw_text = r.get("raw_text", "")
            conf = float(r.get("ocr_confidence", 0.0))
            classified = EntityClassifier.classify(raw_text, ocr_confidence=conf)

            # Update observation
            obs = RawOcrObservation(
                schema_version=r.get("schema_version", "shadow_ocr.v1"),
                site_id=r.get("site_id", site_id),
                passage_id=r.get("passage_id", v["passage_id"]),
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

            r_updated = dict(r)
            r_updated["entity_type"] = classified.entity_type
            r_updated["canonical_name"] = classified.canonical_name
            r_updated["rejection_reason"] = classified.rejection_reason
            updated_raw_dicts.append(r_updated)

        # 4. Run updated consensus engine
        consensus_engine = PhysicalCoachConsensusEngine()
        passage_id = orig_dec["passage_metadata"]["passage_id"]
        camera_id = orig_dec["passage_metadata"].get("camera_id", "side_a")
        video_sha256 = orig_dec["passage_metadata"].get("video_sha256", "")

        identity_res = consensus_engine.build_passage_identity(
            site_id=site_id,
            passage_id=passage_id,
            camera_id=camera_id,
            video_sha256=video_sha256,
            observations=updated_obs,
        )

        # 5. Convert to BetterOcr visual evidence
        evidence = ShadowOcrToBetterOcrAdapter.to_visual_ocr_evidence(identity_res, updated_obs)

        # 6. Resolve shadow decision
        passage_input = PassageInput(
            passage_id=passage_id,
            site_id=site_id,
            camera_id=camera_id,
            passage_start=orig_dec["passage_metadata"].get("passage_start", ""),
            passage_end=orig_dec["passage_metadata"].get("passage_end", ""),
            possible_platforms=orig_dec["passage_metadata"].get("possible_platforms", ["12", "13"]),
            observed_platform=orig_dec["passage_metadata"].get("observed_platform"),
        )

        decision = resolve_shadow(
            passage=passage_input,
            visual_evidence=evidence,
            site_schedule_rows=timetable_rows,
            site_aliases=site_aliases,
            stage="T0",
        )

        # 7. Construct replay decision dict
        replay_decision_dict = dict(orig_dec)
        replay_decision_dict["pipeline_stage"] = "PHASE_7C_POST_BENCHMARK_DEVELOPMENT_REPLAY"
        if "visual_evidence" in replay_decision_dict:
            replay_decision_dict["visual_evidence"]["board_category"] = identity_res.board_category
            replay_decision_dict["visual_evidence"]["visual_pair_status"] = identity_res.visual_pair_status
            replay_decision_dict["visual_evidence"]["train_name_candidates"] = [
                {
                    "name": c.name,
                    "distinct_coaches": c.distinct_coaches,
                    "total_observations": c.total_observations,
                    "mean_confidence": c.mean_confidence,
                }
                for c in identity_res.train_name_candidates
            ]
        replay_decision_dict["resolver_decision"]["shadow_final_number"] = decision.shadow_final_number
        replay_decision_dict["resolver_decision"]["shadow_status"] = decision.shadow_status
        replay_decision_dict["resolver_decision"]["reason_codes"] = decision.reason_codes
        replay_decision_dict["resolver_decision"]["recognized_train_name"] = decision.recognized_train_name
        replay_decision_dict["resolver_decision"]["service_family_candidates"] = decision.service_family_candidates
        replay_decision_dict["resolver_decision"]["active_service_number"] = decision.active_service_number
        replay_decision_dict["resolver_decision"]["active_direction_verified"] = decision.active_direction_verified
        replay_decision_dict["resolver_decision"]["ranked_candidates"] = [
            {
                "train_number": c.train_number,
                "train_name": c.train_name,
                "score": c.score,
                "evidence_list": c.evidence_list,
            }
            for c in decision.candidate_services
        ]

        # Save replay artifacts to shadow_lab/data/phase7c_replay/
        replay_raw_path = replay_dir / f"phase7c_replay_{v['name']}_raw.jsonl"
        replay_dec_path = replay_dir / f"phase7c_replay_{v['name']}_decision.json"

        with open(replay_raw_path, "w", encoding="utf-8") as f:
            for item in updated_raw_dicts:
                f.write(json.dumps(item) + "\n")

        with open(replay_dec_path, "w", encoding="utf-8") as f:
            json.dump(replay_decision_dict, f, indent=2)

        # Record comparison summary
        orig_res = orig_dec_saved.get("resolver_decision", {})
        orig_vis = orig_dec_saved.get("visual_evidence", {}) or orig_dec_saved.get("visual_identity_consensus", {})

        summary = {
            "video_id": v["name"],
            "raw_observations_count": len(updated_obs),
            "before": {
                "entity_types": [o.get("entity_type") for o in orig_raw_lines if o.get("raw_text")],
                "train_name_candidates": orig_vis.get("train_name_candidates", []),
                "recognized_train_name": orig_res.get("recognized_train_name", None),
                "service_family_candidates": orig_res.get("service_family_candidates", []),
                "shadow_final_number": orig_res.get("shadow_final_number", None),
                "active_service_number": orig_res.get("active_service_number", None),
                "active_direction_verified": orig_res.get("active_direction_verified", False),
                "shadow_status": orig_res.get("shadow_status", "UNKNOWN"),
                "reason_codes": orig_res.get("reason_codes", []),
            },
            "after": {
                "entity_types": [o.entity_type for o in updated_obs if o.raw_text],
                "train_name_candidates": [
                    {"name": c.name, "distinct_coaches": c.distinct_coaches, "confidence": c.mean_confidence}
                    for c in identity_res.train_name_candidates
                ],
                "recognized_train_name": decision.recognized_train_name,
                "service_family_candidates": decision.service_family_candidates,
                "shadow_final_number": decision.shadow_final_number,
                "active_service_number": decision.active_service_number,
                "active_direction_verified": decision.active_direction_verified,
                "shadow_status": decision.shadow_status,
                "reason_codes": decision.reason_codes,
            },
        }
        replay_results.append(summary)

    # Save summary report
    summary_path = replay_dir / "phase7c_replay_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(replay_results, f, indent=2)

    print("Phase 7C Replay completed successfully.")
    for res in replay_results:
        print(f"\n--- Video: {res['video_id']} ---")
        print(f"  Before: status={res['before']['shadow_status']}, final_number={res['before']['shadow_final_number']}, name_cands={res['before']['train_name_candidates']}")
        print(f"  After:  status={res['after']['shadow_status']}, final_number={res['after']['shadow_final_number']}, recognized_name={res['after']['recognized_train_name']}, family_cands={res['after']['service_family_candidates']}, active_num={res['after']['active_service_number']}, active_dir_verified={res['after']['active_direction_verified']}")
        print(f"  Reason codes: {res['after']['reason_codes']}")


if __name__ == "__main__":
    run_phase7c_replay()
