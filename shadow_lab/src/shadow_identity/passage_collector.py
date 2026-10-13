"""
Real-Passage Shadow Collector Engine
Consumes real ARDS passage evidence, coordinates parallel OCR entity aggregation,
site-specific timetable matching, time-ordered NTES verification (T0, T1, T2),
and generates auditable comparison bundles and daily/weekly reports.
Strictly read-only and additive — zero mutations to production ARDS files.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.coach_consensus import CoachConsensusAggregator
from shadow_identity.comparison import ComparisonEvaluator, categorize_comparison
from shadow_identity.config_loader import ConfigError, SiteRegistry, compute_file_sha256
from shadow_identity.models import (
    NameCandidate,
    NtesSnapshot,
    NtesTrainEntry,
    NumberCandidate,
    PassageBundle,
    PassageInput,
    PassageVisualConsensus,
    ShadowDecision,
    TimetableRow,
    VisualObservation,
    VisualOcrEvidence,
)
from shadow_identity.reason_codes import (
    FT_BOARD_NOT_NUMBERED,
    FT_MULTIPLE_LIVE_CANDIDATES,
    FT_NTES_PLATFORM_CONFLICT,
    FT_NTES_STALE,
    FT_NTES_UNAVAILABLE,
    FT_OCR_NAME_FAILURE,
    FT_OCR_NUMBER_FAILURE,
    FT_SITE_CONFIG_ERROR,
    FT_TIMETABLE_AMBIGUITY,
    FT_TIMETABLE_NTES_CONFLICT,
    FT_TIMETABLE_STALE,
    FT_UNKNOWN,
    FT_VISUAL_NTES_CONFLICT,
    FT_VISUAL_TIMETABLE_CONFLICT,
    RC_AMBIGUOUS_MULTI_CANDIDATE,
    RC_NTES_CORROBORATED_STALE,
    RC_NTES_UNAVAILABLE,
    RC_NTES_WRONG_STATION_REJECTED,
    RC_SCHEDULE_OUTSIDE_WINDOW,
    RC_SCHEDULE_UNAVAILABLE,
    RC_TIMETABLE_NTES_CONFLICT,
    RC_VISUAL_BOARD_NOT_NUMBERED,
    RC_VISUAL_NTES_CONFLICT,
    RC_VISUAL_NUMBERS_UNAVAILABLE,
    RC_VISUAL_TIMETABLE_CONFLICT,
)
from shadow_identity.text_normalization import is_endpoint_text, is_negative_class
from shadow_identity.timetable_importer import TimetableImporter


class ShadowCollector:
    def __init__(self, shadow_lab_root: Optional[Path] = None):
        if shadow_lab_root is None:
            self.root = Path(__file__).resolve().parents[2]
        else:
            self.root = Path(shadow_lab_root)

        self.registry = SiteRegistry(shadow_lab_root=self.root)
        self.data_dir = self.root / "data"
        self.passages_dir = self.data_dir / "passages"
        self.ntes_dir = self.data_dir / "ntes_snapshots"
        self.production_exports_dir = self.data_dir / "production_exports"
        self.ground_truth_dir = self.data_dir / "ground_truth"

        self.results_dir = self.root / "results"
        self.passage_results_dir = self.results_dir / "passage_results"
        self.daily_dir = self.results_dir / "daily"
        self.weekly_dir = self.results_dir / "weekly"

        for d in (
            self.passages_dir,
            self.ntes_dir,
            self.production_exports_dir,
            self.ground_truth_dir,
            self.passage_results_dir,
            self.daily_dir,
            self.weekly_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        from shadow_identity.delayed_ntes_worker import DelayedNtesVerificationWorker
        self.worker = DelayedNtesVerificationWorker(shadow_lab_root=self.root)

    def import_production_passage(
        self,
        source: Union[Path, Dict[str, Any]],
        site_id: str,
        custom_ground_truth: Optional[Dict[str, Any]] = None,
    ) -> PassageBundle:
        """
        Safely import a real ARDS passage.
        Operates strictly on copies/read-only access; guarantees ZERO writes to production.
        """
        if isinstance(source, (str, Path)):
            src_path = Path(source)
            if not src_path.is_file():
                raise FileNotFoundError(f"Production passage file not found: {src_path}")
            with open(src_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        else:
            raw_data = json.loads(json.dumps(source))  # deep copy

        # Validate site_id against site registry
        site_cfg = self.registry.get_site(site_id)
        aliases = self.registry.get_aliases(site_id)

        # 1. Parse Passage Identity Metadata
        p_raw = raw_data.get("passage", raw_data)
        passage_id = p_raw.get("passage_id")
        if not passage_id:
            ts = p_raw.get("start_time") or p_raw.get("passage_start") or datetime.now().isoformat()
            ts_clean = ts.replace(":", "-").replace(".", "-")
            passage_id = f"{ts_clean}_{site_id}_P001"

        passage_start = p_raw.get("start_time") or p_raw.get("passage_start") or ""
        passage_end = p_raw.get("end_time") or p_raw.get("passage_end") or passage_start
        obs_platform = p_raw.get("observed_platform")
        possible_platforms = p_raw.get("possible_platforms") or site_cfg.get("platforms", {}).get("configured", [])
        direction = p_raw.get("direction", "unknown")
        source_files = p_raw.get("source_files", [])

        passage_dict = {
            "start_time": passage_start,
            "end_time": passage_end,
            "observed_platform": obs_platform,
            "possible_platforms": possible_platforms,
            "direction": direction,
            "source_files": source_files,
        }

        # 2. Parse Production ARDS Metadata (Read-Only)
        prod_raw = raw_data.get("production", raw_data.get("production_readonly_result", {}))
        prod_dict = {
            "inspection_id": prod_raw.get("inspection_id") or p_raw.get("production_inspection_id"),
            "train_number": prod_raw.get("train_number"),
            "confidence": prod_raw.get("confidence", "LOW"),
            "resolver_source": prod_raw.get("resolver_source", prod_raw.get("source", "existing_ards")),
            "recorded_at": prod_raw.get("recorded_at", passage_end),
        }

        # 3. Parse Visual Evidence in Parallel (Numbers AND Train Names/Endpoints)
        vis_raw = raw_data.get("visual", raw_data.get("visual_evidence", {}))
        observations: List[VisualObservation] = []
        raw_obs_list = vis_raw.get("coach_evidence") or vis_raw.get("per_observation") or []

        for o in raw_obs_list:
            cid = o.get("coach_segment_id") or o.get("coach_id", "C01")
            fid = int(o.get("frame_id", 0))
            raw_text = o.get("raw_text", "")
            norm_text = o.get("normalized_text", raw_text)
            conf = float(o.get("ocr_confidence", o.get("confidence", 0.0)))
            etype = o.get("entity_type", "OTHER_TEXT")

            if etype == "OTHER_TEXT" or not etype:
                if is_negative_class(raw_text):
                    etype = "NEGATIVE_STENCIL"
                elif any(c.isdigit() for c in raw_text) and len(raw_text.strip()) == 5:
                    etype = "SERVICE_NUMBER"
                elif is_endpoint_text(raw_text):
                    etype = "ENDPOINT"
                elif any(c.isalpha() for c in raw_text) and len(raw_text.strip()) > 3:
                    etype = "TRAIN_NAME"
                else:
                    etype = "OTHER_TEXT"

            observations.append(
                VisualObservation(
                    coach_segment_id=cid,
                    frame_id=fid,
                    raw_text=raw_text,
                    normalized_text=norm_text,
                    entity_type=etype,
                    timestamp_offset_ms=int(o.get("timestamp_offset_ms", 0)),
                    bbox=o.get("bbox", []),
                    ocr_confidence=conf,
                )
            )

        aggregator = CoachConsensusAggregator(confirmed_threshold=3)
        if observations:
            consensus = aggregator.aggregate_observations(observations)
        else:
            # Fallback to pre-aggregated candidates if raw frame observations were omitted
            cons_raw = vis_raw.get("passage_consensus", vis_raw)
            num_cands = [NumberCandidate(**c) for c in cons_raw.get("service_number_candidates", [])]
            name_cands = [NameCandidate(**c) for c in cons_raw.get("train_name_candidates", [])]
            raw_eps = cons_raw.get("endpoint_candidates", [])
            ep_cands = [e if isinstance(e, str) else e.get("endpoint", "") for e in raw_eps]
            consensus = PassageVisualConsensus(
                service_number_candidates=num_cands,
                stable_pair=cons_raw.get("stable_pair", []),
                train_name_candidates=name_cands,
                endpoint_candidates=ep_cands,
                number_board_type=cons_raw.get("number_board_type", "UNKNOWN"),
                numeric_status=cons_raw.get("numeric_status", "UNKNOWN"),
            )

        visual_dict = {
            "service_numbers": [asdict(c) for c in consensus.service_number_candidates],
            "stable_pair": list(consensus.stable_pair),
            "train_name_candidates": [asdict(c) for c in consensus.train_name_candidates],
            "endpoint_candidates": list(consensus.endpoint_candidates),
            "coach_evidence": [asdict(o) for o in observations],
            "numeric_status": consensus.numeric_status,
            "number_board_type": consensus.number_board_type,
        }

        # 4. Load Site-Specific Timetable
        sched_cfg = site_cfg.get("schedule", {})
        sched_path_str = sched_cfg.get("path") or site_cfg.get("timetable_file")
        sched_path = Path(sched_path_str)
        if not sched_path.is_absolute():
            sched_path = self.root.parent / sched_path
            if not sched_path.exists():
                sched_path = self.root / sched_path_str

        timetable_snapshot = TimetableImporter.import_schedule(
            site_id=site_id,
            station_code=site_cfg.get("station", {}).get("code", site_cfg.get("station_code", "")),
            csv_path=str(sched_path),
            version=sched_cfg.get("version", "v1"),
            effective_from=sched_cfg.get("effective_from"),
        )

        # 5. Extract NTES Evidence (Pre-passage and Post-passage +180s)
        ntes_raw = raw_data.get("ntes") or {}
        pre_snap_raw = ntes_raw.get("pre_or_near_passage_snapshot", raw_data.get("ntes_before"))
        post_snap_raw = ntes_raw.get("post_passage_snapshot", raw_data.get("ntes_after"))

        def parse_ntes_obj(d: Optional[Dict[str, Any]]) -> Optional[NtesSnapshot]:
            if not d:
                return None
            train_objs = [NtesTrainEntry(**t) for t in d.get("trains", [])]
            return NtesSnapshot(
                site_id=d.get("site_id", site_id),
                station_code=d.get("station_code", site_cfg.get("station", {}).get("code", "")),
                requested_at=d.get("requested_at", ""),
                fetched_at=d.get("fetched_at", ""),
                window_hr=d.get("window_hr", 2),
                available=d.get("available", True),
                from_cache=d.get("from_cache", False),
                stale_fallback=d.get("stale_fallback", False),
                snapshot_age_seconds_at_decision=d.get("snapshot_age_seconds_at_decision", 0.0),
                source=d.get("source", "replay"),
                trains=train_objs,
                raw_hash=d.get("raw_hash", ""),
            )

        ntes_before = parse_ntes_obj(pre_snap_raw)
        ntes_after = parse_ntes_obj(post_snap_raw)

        # Build Canonical Internal Models for Resolver
        passage_input = PassageInput(
            site_id=site_id,
            installation_id=site_cfg.get("installation_id", "inst_01"),
            passage_id=passage_id,
            camera_id=site_cfg.get("cameras", [{}])[0].get("camera_id", "side_a"),
            passage_start=passage_start,
            passage_end=passage_end,
            possible_platforms=possible_platforms,
            observed_platform=obs_platform,
            direction=direction,
            source_files=source_files,
            production_inspection_id=prod_dict["inspection_id"],
            production_readonly_result=prod_dict,
        )

        visual_evidence = VisualOcrEvidence(
            passage_id=passage_id,
            ocr_model_version="paddleocr_ppocrv6",
            preprocessing_profile="2x_bicubic",
            per_observation=observations,
            passage_consensus=consensus,
        )

        # 6. Execute Multi-Stage Decisions (T0, T1, T2)
        dec_t0 = resolve_shadow(
            passage=passage_input,
            visual_evidence=visual_evidence,
            site_schedule_rows=timetable_snapshot.rows,
            ntes_before=None,
            ntes_after=None,
            site_aliases=aliases,
            config=site_cfg,
            stage="T0",
        )

        dec_t1 = resolve_shadow(
            passage=passage_input,
            visual_evidence=visual_evidence,
            site_schedule_rows=timetable_snapshot.rows,
            ntes_before=ntes_before,
            ntes_after=None,
            site_aliases=aliases,
            config=site_cfg,
            stage="T1",
        )

        dec_t2 = resolve_shadow(
            passage=passage_input,
            visual_evidence=visual_evidence,
            site_schedule_rows=timetable_snapshot.rows,
            ntes_before=ntes_before,
            ntes_after=ntes_after,
            site_aliases=aliases,
            config=site_cfg,
            stage="T2",
        )

        # 7. Ground Truth Handling
        gt_raw = custom_ground_truth or raw_data.get("ground_truth", {})
        gt_num = gt_raw.get("train_number") or gt_raw.get("actual_train_number")
        gt_name = gt_raw.get("train_name") or gt_raw.get("actual_train_name")
        gt_rev = gt_raw.get("reviewer")
        gt_src = gt_raw.get("source") or gt_raw.get("review_source")
        gt_notes = gt_raw.get("notes")
        is_reviewed = bool(gt_raw.get("reviewed", bool(gt_num and gt_rev)))

        ground_truth_dict = {
            "reviewed": is_reviewed,
            "train_number": gt_num,
            "train_name": gt_name,
            "reviewer": gt_rev,
            "source": gt_src,
            "notes": gt_notes,
        }

        # 8. Comparison Classification (§18)
        prod_num = prod_dict["train_number"]
        prod_conf = prod_dict["confidence"]
        shadow_num = dec_t2.shadow_final_number
        shadow_status = dec_t2.shadow_status

        if prod_num and shadow_num:
            prod_vs_shadow = "MATCH" if prod_num == shadow_num else "CONFLICT"
        elif shadow_num and not prod_num:
            prod_vs_shadow = "SHADOW_ONLY"
        elif prod_num and not shadow_num:
            prod_vs_shadow = "PRODUCTION_ONLY"
        else:
            prod_vs_shadow = "BOTH_UNKNOWN"

        eval_cat, rev_req = categorize_comparison(
            prod_num=prod_num,
            prod_conf=prod_conf,
            shadow_num=shadow_num,
            shadow_status=shadow_status,
            ground_truth=gt_num,
        )

        # Failure Taxonomy Assignment (§23)
        failure_reason = None
        if shadow_status in ("UNKNOWN", "CONFLICT", "AMBIGUOUS") or prod_vs_shadow == "CONFLICT":
            if RC_VISUAL_NTES_CONFLICT in dec_t2.reason_codes:
                failure_reason = FT_VISUAL_NTES_CONFLICT
            elif RC_VISUAL_TIMETABLE_CONFLICT in dec_t2.reason_codes:
                failure_reason = FT_VISUAL_TIMETABLE_CONFLICT
            elif RC_TIMETABLE_NTES_CONFLICT in dec_t2.reason_codes:
                failure_reason = FT_TIMETABLE_NTES_CONFLICT
            elif RC_VISUAL_BOARD_NOT_NUMBERED in dec_t2.reason_codes and not consensus.train_name_candidates:
                failure_reason = FT_BOARD_NOT_NUMBERED
            elif RC_AMBIGUOUS_MULTI_CANDIDATE in dec_t2.reason_codes:
                failure_reason = FT_MULTIPLE_LIVE_CANDIDATES
            elif RC_NTES_UNAVAILABLE in dec_t2.reason_codes:
                failure_reason = FT_NTES_UNAVAILABLE
            elif RC_NTES_CORROBORATED_STALE in dec_t2.reason_codes:
                failure_reason = FT_NTES_STALE
            elif RC_SCHEDULE_OUTSIDE_WINDOW in dec_t2.reason_codes:
                failure_reason = FT_TIMETABLE_AMBIGUITY
            elif RC_SCHEDULE_UNAVAILABLE in dec_t2.reason_codes:
                failure_reason = FT_TIMETABLE_STALE
            elif not consensus.service_number_candidates:
                failure_reason = FT_OCR_NUMBER_FAILURE
            else:
                failure_reason = FT_UNKNOWN

        comparison_dict = {
            "category": eval_cat,
            "production_vs_shadow": prod_vs_shadow,
            "review_required": rev_req,
            "failure_reason": failure_reason,
        }

        # 9. Build Canonical Read-Only Passage Evidence Bundle (§3)
        bundle = PassageBundle(
            site_id=site_id,
            passage_id=passage_id,
            passage=passage_dict,
            production=prod_dict,
            visual=visual_dict,
            timetable={
                "version": timetable_snapshot.version,
                "candidate_services": [asdict(c) for c in dec_t0.candidate_services],
            },
            ntes={
                "pre_or_near_passage_snapshot": asdict(ntes_before) if ntes_before else None,
                "post_passage_snapshot": asdict(ntes_after) if ntes_after else None,
            },
            shadow_result={
                "T0": {
                    "train_number": dec_t0.shadow_final_number,
                    "status": dec_t0.shadow_status,
                    "reason_codes": dec_t0.reason_codes,
                },
                "T1": {
                    "train_number": dec_t1.shadow_final_number,
                    "status": dec_t1.shadow_status,
                    "reason_codes": dec_t1.reason_codes,
                },
                "T2": {
                    "train_number": dec_t2.shadow_final_number,
                    "status": dec_t2.shadow_status,
                    "reason_codes": dec_t2.reason_codes,
                    "post_fetch_kind": dec_t2.post_fetch_kind,
                },
            },
            comparison=comparison_dict,
            ground_truth=ground_truth_dict,
        )

        # 10. Persist Bundle Additively in Shadow Lab Directories (NEVER into production)
        out_json_path = self.passages_dir / f"{passage_id}.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(bundle.to_dict(), f, indent=2)

        res_json_path = self.passage_results_dir / f"{passage_id}.json"
        with open(res_json_path, "w", encoding="utf-8") as f:
            json.dump(bundle.to_dict(), f, indent=2)

        # 11. Enqueue for delayed NTES verification if post-passage snapshot is not yet present (§4)
        if ntes_after is None:
            self.worker.enqueue_passage(
                passage_id=passage_id,
                site_id=site_id,
                passage_end_time=passage_end,
            )

        return bundle

    def run_delayed_verification(self, passage_id: Optional[str] = None) -> List[PassageBundle]:
        """Process delayed NTES verification queue."""
        if passage_id:
            res = self.worker.process_passage(passage_id)
            return [res] if res else []
        return self.worker.process_all_pending()


    def record_ground_truth(
        self,
        passage_id: str,
        train_number: str,
        train_name: Optional[str] = None,
        reviewer: str = "human_expert",
        source: str = "manual_station_confirmation",
        notes: Optional[str] = None,
    ) -> PassageBundle:
        """
        Record verified manual ground truth for a passage and recompute comparison category.
        """
        bundle_file = self.passages_dir / f"{passage_id}.json"
        if not bundle_file.is_file():
            # Check tests/fixtures
            fix_file = self.root / "tests" / "fixtures" / f"{passage_id}.json"
            if fix_file.is_file():
                bundle_file = fix_file
            else:
                raise FileNotFoundError(f"Passage bundle not found for ID: {passage_id}")

        with open(bundle_file, "r", encoding="utf-8") as f:
            b_dict = json.load(f)

        gt_entry = {
            "reviewed": True,
            "train_number": str(train_number).strip(),
            "train_name": str(train_name).strip() if train_name else None,
            "reviewer": str(reviewer).strip(),
            "source": str(source).strip(),
            "notes": str(notes).strip() if notes else None,
            "recorded_at": datetime.now().isoformat(),
        }

        # Re-import through safe importer
        bundle = self.import_production_passage(
            source=b_dict,
            site_id=b_dict.get("site_id", "ndls_pf12_13"),
            custom_ground_truth=gt_entry,
        )

        # Store explicit record in data/ground_truth
        gt_path = self.ground_truth_dir / f"{passage_id}_gt.json"
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_entry, f, indent=2)

        return bundle

    def generate_daily_report(self, report_date: str, site_id: Optional[str] = None) -> Tuple[Path, Path]:
        """
        Generate Daily Comparison Report (§29) for a given date YYYY-MM-DD.
        Outputs CSV and Markdown report in shadow_lab/results/daily/
        """
        candidate_files = sorted(list(self.passages_dir.glob("*.json")))
        if not candidate_files:
            candidate_files = sorted(list((self.root / "tests" / "fixtures").glob("*.json")))

        matched_records: List[Dict[str, Any]] = []

        for p_file in candidate_files:
            with open(p_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            s_id = d.get("site_id")
            if site_id and s_id != site_id:
                continue

            # Check if passage timestamp matches report_date
            p_time = d.get("passage", {}).get("start_time") or d.get("passage", {}).get("passage_start") or ""
            if report_date and report_date not in p_time:
                continue

            matched_records.append(d)

        # Compute metrics
        total = len(matched_records)
        matches = sum(1 for r in matched_records if r.get("comparison", {}).get("production_vs_shadow") == "MATCH")
        conflicts = sum(1 for r in matched_records if r.get("comparison", {}).get("production_vs_shadow") == "CONFLICT")
        unknowns = sum(
            1
            for r in matched_records
            if r.get("shadow_result", {}).get("T2", {}).get("status") in ("UNKNOWN", "DATA_UNAVAILABLE")
        )
        gt_pending = sum(1 for r in matched_records if not r.get("ground_truth", {}).get("reviewed"))

        # Write CSV
        csv_path = self.daily_dir / f"daily_{report_date}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "passage_id", "site_id", "start_time", "observed_platform",
                "prod_number", "prod_conf", "shadow_T0", "shadow_T1", "shadow_T2",
                "shadow_status", "prod_vs_shadow", "comparison_category",
                "ground_truth_number", "gt_reviewed", "failure_reason"
            ])
            for r in matched_records:
                p_info = r.get("passage", {})
                prod = r.get("production", {})
                sh = r.get("shadow_result", {})
                cmp = r.get("comparison", {})
                gt = r.get("ground_truth", {})
                writer.writerow([
                    r.get("passage_id"),
                    r.get("site_id"),
                    p_info.get("start_time"),
                    p_info.get("observed_platform") or f"UNKNOWN {p_info.get('possible_platforms')}",
                    prod.get("train_number"),
                    prod.get("confidence"),
                    sh.get("T0", {}).get("train_number"),
                    sh.get("T1", {}).get("train_number"),
                    sh.get("T2", {}).get("train_number"),
                    sh.get("T2", {}).get("status"),
                    cmp.get("production_vs_shadow"),
                    cmp.get("category"),
                    gt.get("train_number"),
                    gt.get("reviewed"),
                    cmp.get("failure_reason") or "NONE",
                ])

        # Write Markdown
        md_path = self.daily_dir / f"daily_{report_date}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Daily Shadow Comparison Report — {report_date}\n\n")
            f.write(f"**Site Filter**: `{site_id or 'ALL SITES'}` &nbsp;|&nbsp; **Generated At**: `{datetime.now().isoformat()}`\n\n")
            f.write("## 1. Summary Metrics\n\n")
            f.write(f"| Metric | Count | Percentage |\n")
            f.write(f"| :--- | :--- | :--- |\n")
            f.write(f"| Total Passages Observed | **{total}** | 100.0% |\n")
            f.write(f"| Production == Shadow Matches | **{matches}** | {matches/total*100 if total else 0:.1f}% |\n")
            f.write(f"| Disagreements / Conflicts | **{conflicts}** | {conflicts/total*100 if total else 0:.1f}% |\n")
            f.write(f"| Shadow Abstentions (Unknown) | **{unknowns}** | {unknowns/total*100 if total else 0:.1f}% |\n")
            f.write(f"| Manual Ground-Truth Reviews Pending | **{gt_pending}** | {gt_pending/total*100 if total else 0:.1f}% |\n\n")

            f.write("## 2. Passage Details\n\n")
            f.write("| Passage ID | Platform | Production | Shadow T0 | Shadow T1 | Shadow T2 | Status | Comparison | Ground Truth |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for r in matched_records:
                p_info = r.get("passage", {})
                prod = r.get("production", {})
                sh = r.get("shadow_result", {})
                cmp = r.get("comparison", {})
                gt = r.get("ground_truth", {})
                pf_disp = p_info.get("observed_platform") or ",".join(p_info.get("possible_platforms", []))
                f.write(
                    f"| `{r.get('passage_id')}` | PF {pf_disp} | "
                    f"`{prod.get('train_number') or '<none>'}` ({prod.get('confidence')}) | "
                    f"`{sh.get('T0', {}).get('train_number') or '-'}` | "
                    f"`{sh.get('T1', {}).get('train_number') or '-'}` | "
                    f"`{sh.get('T2', {}).get('train_number') or '-'}` | "
                    f"**{sh.get('T2', {}).get('status')}** | "
                    f"`{cmp.get('production_vs_shadow')}` | "
                    f"`{gt.get('train_number') or 'PENDING'}` |\n"
                )

        return csv_path, md_path

    def generate_weekly_report(self, week_start: str, site_id: Optional[str] = None) -> Tuple[Path, Path]:
        """
        Generate Weekly Comparison Report (§20-§23, §30).
        Calculates overall, per-site, evidence contribution, and failure taxonomy metrics.
        Outputs CSV and comprehensive Markdown report in shadow_lab/results/weekly/
        """
        w_start_dt = datetime.fromisoformat(week_start)
        w_end_dt = w_start_dt + timedelta(days=7)

        candidate_files = sorted(list(self.passages_dir.glob("*.json")))
        if not candidate_files:
            candidate_files = sorted(list((self.root / "tests" / "fixtures").glob("*.json")))

        week_records: List[Dict[str, Any]] = []

        for p_file in candidate_files:
            with open(p_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            s_id = d.get("site_id")
            if site_id and s_id != site_id:
                continue

            p_time = d.get("passage", {}).get("start_time") or d.get("passage", {}).get("passage_start")
            if p_time:
                try:
                    p_dt = datetime.fromisoformat(p_time)
                    if p_dt.tzinfo and not w_start_dt.tzinfo:
                        ws_cmp, we_cmp = w_start_dt.replace(tzinfo=p_dt.tzinfo), w_end_dt.replace(tzinfo=p_dt.tzinfo)
                    elif not p_dt.tzinfo and w_start_dt.tzinfo:
                        ws_cmp, we_cmp = w_start_dt.replace(tzinfo=None), w_end_dt.replace(tzinfo=None)
                    else:
                        ws_cmp, we_cmp = w_start_dt, w_end_dt

                    if not (ws_cmp <= p_dt <= we_cmp):
                        continue
                except Exception:
                    pass

            week_records.append(d)

        total_passages = len(week_records)
        gt_reviewed_count = sum(1 for r in week_records if r.get("ground_truth", {}).get("reviewed"))

        # Breakdown metrics
        categories: Dict[str, int] = {}
        per_site_counts: Dict[str, Dict[str, int]] = {}
        failure_counts: Dict[str, int] = {}
        evidence_counts = {
            "numeric_pair": 0,
            "train_name": 0,
            "endpoint_evidence": 0,
            "timetable_match": 0,
            "near_passage_ntes": 0,
            "post_passage_ntes": 0,
        }

        prod_correct = 0
        shadow_t0_correct = 0
        shadow_t1_correct = 0
        shadow_t2_correct = 0

        for r in week_records:
            s_id = r.get("site_id", "unknown")
            cat = r.get("comparison", {}).get("category", "UNKNOWN")
            categories[cat] = categories.get(cat, 0) + 1

            per_site_counts.setdefault(s_id, {"total": 0, "match": 0, "conflict": 0})
            per_site_counts[s_id]["total"] += 1
            if r.get("comparison", {}).get("production_vs_shadow") == "MATCH":
                per_site_counts[s_id]["match"] += 1
            elif r.get("comparison", {}).get("production_vs_shadow") == "CONFLICT":
                per_site_counts[s_id]["conflict"] += 1

            fail_reason = r.get("comparison", {}).get("failure_reason")
            if fail_reason:
                failure_counts[fail_reason] = failure_counts.get(fail_reason, 0) + 1

            # Evidence source tracking (§21)
            vis = r.get("visual", {})
            if vis.get("stable_pair"): evidence_counts["numeric_pair"] += 1
            if vis.get("train_name_candidates"): evidence_counts["train_name"] += 1
            if vis.get("endpoint_candidates"): evidence_counts["endpoint_evidence"] += 1
            if r.get("timetable", {}).get("candidate_services"): evidence_counts["timetable_match"] += 1
            if r.get("ntes", {}).get("pre_or_near_passage_snapshot"): evidence_counts["near_passage_ntes"] += 1
            if r.get("ntes", {}).get("post_passage_snapshot"): evidence_counts["post_passage_ntes"] += 1

            # Ground truth accuracy tracking (§20)
            gt_num = r.get("ground_truth", {}).get("train_number")
            if gt_num:
                if r.get("production", {}).get("train_number") == gt_num: prod_correct += 1
                if r.get("shadow_result", {}).get("T0", {}).get("train_number") == gt_num: shadow_t0_correct += 1
                if r.get("shadow_result", {}).get("T1", {}).get("train_number") == gt_num: shadow_t1_correct += 1
                if r.get("shadow_result", {}).get("T2", {}).get("train_number") == gt_num: shadow_t2_correct += 1

        # Write Weekly CSV
        csv_path = self.weekly_dir / f"weekly_comparison_{week_start}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "passage_id", "site_id", "start_time", "observed_platform",
                "prod_train", "prod_conf", "shadow_T0", "shadow_T1", "shadow_T2",
                "shadow_status", "prod_vs_shadow", "comparison_category",
                "ground_truth", "gt_reviewer", "failure_reason"
            ])
            for r in week_records:
                p_info = r.get("passage", {})
                prod = r.get("production", {})
                sh = r.get("shadow_result", {})
                cmp = r.get("comparison", {})
                gt = r.get("ground_truth", {})
                writer.writerow([
                    r.get("passage_id"),
                    r.get("site_id"),
                    p_info.get("start_time"),
                    p_info.get("observed_platform") or f"UNKNOWN {p_info.get('possible_platforms')}",
                    prod.get("train_number"),
                    prod.get("confidence"),
                    sh.get("T0", {}).get("train_number"),
                    sh.get("T1", {}).get("train_number"),
                    sh.get("T2", {}).get("train_number"),
                    sh.get("T2", {}).get("status"),
                    cmp.get("production_vs_shadow"),
                    cmp.get("category"),
                    gt.get("train_number"),
                    gt.get("reviewer"),
                    cmp.get("failure_reason") or "NONE",
                ])

        # Write Comprehensive Weekly Markdown (§30)
        md_path = self.weekly_dir / f"weekly_comparison_{week_start}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Multi-Site Train Identity Shadow Lab — Weekly Comparison Report\n\n")
            f.write(f"**Evaluation Week**: `{week_start}` to `{w_end_dt.date().isoformat()}` &nbsp;|&nbsp; ")
            f.write(f"**Total Passages**: `{total_passages}` &nbsp;|&nbsp; ")
            f.write(f"**Verified Ground Truth**: `{gt_reviewed_count}`\n\n")
            f.write("---\n\n")

            f.write("## 1. Overall Head-to-Head Benchmark Metrics\n\n")
            f.write("| Evaluation Metric | Count | Percentage |\n")
            f.write("| :--- | :--- | :--- |\n")
            f.write(f"| Total Candidate Passages | **{total_passages}** | 100.0% |\n")
            f.write(f"| Ground-Truth Reviewed Passages | **{gt_reviewed_count}** | {gt_reviewed_count/total_passages*100 if total_passages else 0:.1f}% |\n")
            f.write(f"| Production Correct (vs Ground Truth) | **{prod_correct}** | {prod_correct/gt_reviewed_count*100 if gt_reviewed_count else 0:.1f}% |\n")
            f.write(f"| Shadow T0 Correct (Visual + Schedule) | **{shadow_t0_correct}** | {shadow_t0_correct/gt_reviewed_count*100 if gt_reviewed_count else 0:.1f}% |\n")
            f.write(f"| Shadow T1 Correct (+ Near NTES) | **{shadow_t1_correct}** | {shadow_t1_correct/gt_reviewed_count*100 if gt_reviewed_count else 0:.1f}% |\n")
            f.write(f"| Shadow T2 Correct (+ Post NTES) | **{shadow_t2_correct}** | {shadow_t2_correct/gt_reviewed_count*100 if gt_reviewed_count else 0:.1f}% |\n\n")

            f.write("## 2. Comparison Category Taxonomy Breakdown\n\n")
            f.write("| Taxonomy Category | Count | Meaning |\n")
            f.write("| :--- | :--- | :--- |\n")
            for cat_name, cnt in sorted(categories.items()):
                f.write(f"| `{cat_name}` | **{cnt}** | {cnt/total_passages*100 if total_passages else 0:.1f}% of total |\n")
            f.write("\n")

            f.write("## 3. Per-Site Isolation & Performance Breakdown (§22)\n\n")
            f.write("| Site ID | Total Passages | Match Rate | Conflict Rate |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for s_id, s_metrics in sorted(per_site_counts.items()):
                t_site = s_metrics["total"]
                m_rate = s_metrics["match"] / t_site * 100 if t_site else 0
                c_rate = s_metrics["conflict"] / t_site * 100 if t_site else 0
                f.write(f"| `{s_id}` | **{t_site}** | {m_rate:.1f}% | {c_rate:.1f}% |\n")
            f.write("\n")

            f.write("## 4. Evidence Source Contribution Analysis (§21)\n\n")
            f.write("| Evidence Source | Passages Utilizing | Contribution Role |\n")
            f.write("| :--- | :--- | :--- |\n")
            f.write(f"| Numeric OCR (Stable Pair) | **{evidence_counts['numeric_pair']}** | Primary 5-digit number candidate extraction |\n")
            f.write(f"| Train Name OCR | **{evidence_counts['train_name']}** | Non-numeric fallback & 4/6-number rake disambiguation |\n")
            f.write(f"| Route Endpoints | **{evidence_counts['endpoint_evidence']}** | Corroborating directional terminal evidence |\n")
            f.write(f"| Site Timetable Match | **{evidence_counts['timetable_match']}** | Station schedule window and operating day validation |\n")
            f.write(f"| Near-Passage NTES (T1) | **{evidence_counts['near_passage_ntes']}** | Live platform and movement corroboration |\n")
            f.write(f"| Post-Passage NTES (T2) | **{evidence_counts['post_passage_ntes']}** | +180s departure and arrival persistence check |\n\n")

            f.write("## 5. Failure Reason Taxonomy (§23)\n\n")
            if failure_counts:
                f.write("| Failure Reason | Occurrences | Root Cause |\n")
                f.write("| :--- | :--- | :--- |\n")
                for f_code, f_cnt in sorted(failure_counts.items()):
                    f.write(f"| `{f_code}` | **{f_cnt}** | Audited failure category |\n")
            else:
                f.write("No unresolved failures recorded during this evaluation week.\n")
            f.write("\n")

            f.write("## 6. Representative Passage Walkthrough (§30)\n\n")
            for idx, r in enumerate(week_records[:8], 1):
                p_id = r.get("passage_id")
                p_info = r.get("passage", {})
                prod = r.get("production", {})
                sh = r.get("shadow_result", {})
                cmp = r.get("comparison", {})
                gt = r.get("ground_truth", {})
                vis = r.get("visual", {})

                f.write(f"### Passage {idx}: `{p_id}` ({r.get('site_id')})\n")
                f.write(f"- **Time & Platform**: `{p_info.get('start_time')}` | Platform `{p_info.get('observed_platform') or p_info.get('possible_platforms')}`\n")
                f.write(f"- **Visual Evidence**: Numbers `{vis.get('stable_pair')}` | Name `{vis.get('train_name_candidates')[:1]}`\n")
                f.write(f"- **Decision Pipeline**: T0=`{sh.get('T0', {}).get('train_number')}` → T1=`{sh.get('T1', {}).get('train_number')}` → T2=`{sh.get('T2', {}).get('train_number')}` (**{sh.get('T2', {}).get('status')}**)\n")
                f.write(f"- **Production vs Shadow**: Production=`{prod.get('train_number')}` ({prod.get('confidence')}) | Shadow=`{sh.get('T2', {}).get('train_number')}` | Result=`{cmp.get('production_vs_shadow')}`\n")
                f.write(f"- **Ground Truth**: `{gt.get('train_number') or 'Unreviewed'}` ({gt.get('reviewer') or 'Pending'})\n\n")

        return csv_path, md_path
