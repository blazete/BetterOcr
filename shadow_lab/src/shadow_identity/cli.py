"""Command Line Interface for Multi-Site Train Identity Shadow Lab.
Supports preflight checks, schedule validation, fixture replays,
side-by-side comparison generation, and zero-write integrity verification.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.coach_consensus import CoachConsensusAggregator
from shadow_identity.comparison import ComparisonEvaluator, categorize_comparison
from shadow_identity.config_loader import ConfigError, SiteRegistry, compute_file_sha256
from shadow_identity.models import (
    CandidateService,
    ComparisonRecord,
    NameCandidate,
    NtesSnapshot,
    NtesTrainEntry,
    NumberCandidate,
    PassageInput,
    PassageVisualConsensus,
    ShadowDecision,
    TimetableRow,
    VisualObservation,
    VisualOcrEvidence,
)
from shadow_identity.timetable_importer import TimetableImporter


def get_repo_root() -> Path:
    # CLI file is in shadow_lab/src/shadow_identity/cli.py -> parent.parent.parent is BetterOcr repo root
    return Path(__file__).resolve().parents[3]


def cmd_preflight(args: argparse.Namespace) -> int:
    from shadow_lab.scripts.preflight_check import run_preflight
    success = run_preflight(args.site_id)
    return 0 if success else 1


def cmd_validate_schedule(args: argparse.Namespace) -> int:
    if not args.site_id:
        print("[FAIL] --site-id is mandatory for validate-schedule!")
        return 1

    repo_root = get_repo_root()
    registry = SiteRegistry(shadow_lab_root=repo_root / "shadow_lab")
    try:
        cfg = registry.get_site(args.site_id)
    except ConfigError as e:
        print(f"[FAIL] {e}")
        return 1

    sched_cfg = cfg.get("schedule", {})
    sched_path = repo_root / sched_cfg.get("path", "")
    if not sched_path.exists():
        print(f"[FAIL] Schedule file does not exist: {sched_path}")
        return 1

    st_cfg = cfg.get("station", {})
    snapshot = TimetableImporter.import_schedule(
        site_id=args.site_id,
        station_code=st_cfg.get("code", ""),
        csv_path=str(sched_path),
        version=sched_cfg.get("version", "v1"),
        effective_from=sched_cfg.get("effective_from"),
        effective_until=sched_cfg.get("effective_until"),
    )

    print(f"Site: {args.site_id} ({st_cfg.get('name')})")
    print(f"Schedule Version: {snapshot.version}")
    print(f"Source: {sched_path}")
    print(f"SHA256: {snapshot.checksum_sha256}")
    print(f"Total Rows Parsed: {len(snapshot.rows)}")
    
    # Check leading zeros preservation
    leading_zero_cands = [r.train_number for r in snapshot.rows if r.train_number.startswith("0")]
    print(f"Preserved leading zero trains: {len(leading_zero_cands)} (e.g. {leading_zero_cands[:3]})")

    # Check times
    missing_times = [r.train_number for r in snapshot.rows if not r.arrival_local and not r.departure_local]
    if missing_times:
        print(f"[WARN] Rows missing both arrival and departure: {len(missing_times)} ({missing_times[:3]})")
    else:
        print("[PASS] All rows have valid arrival or departure times parsed.")

    return 0


def cmd_verify_no_writes(args: argparse.Namespace) -> int:
    from shadow_lab.scripts.verify_originals_unchanged import verify_originals
    manifest = args.baseline_manifest or "shadow_lab/workspace/fixtures/original_hashes.json"
    ok = verify_originals(manifest)
    return 0 if ok else 1


def load_fixture_data(fixture_path: Path) -> Dict[str, Any]:
    with open(fixture_path, "r") as f:
        return json.load(f)


def dict_to_passage(p_dict: Dict[str, Any]) -> PassageInput:
    return PassageInput(
        schema_version=p_dict.get("schema_version", "1.0"),
        site_id=p_dict["site_id"],
        installation_id=p_dict.get("installation_id", "inst_01"),
        passage_id=p_dict["passage_id"],
        camera_id=p_dict.get("camera_id", "side_a"),
        passage_start=p_dict.get("passage_start"),
        passage_end=p_dict.get("passage_end"),
        possible_platforms=p_dict.get("possible_platforms", []),
        observed_platform=p_dict.get("observed_platform"),
        direction=p_dict.get("direction", "unknown"),
        source_files=p_dict.get("source_files", []),
        production_inspection_id=p_dict.get("production_inspection_id"),
        production_readonly_result=p_dict.get("production_readonly_result", {}),
    )


def dict_to_visual(v_dict: Dict[str, Any]) -> VisualOcrEvidence:
    obs_list = []
    for o in v_dict.get("per_observation", []):
        obs_list.append(VisualObservation(**o))

    if obs_list:
        # Independently recompute consensus from raw coach observations
        aggregator = CoachConsensusAggregator(confirmed_threshold=3)
        consensus = aggregator.aggregate_observations(obs_list)
    else:
        cons_d = v_dict.get("passage_consensus", {})
        num_cands = [NumberCandidate(**c) for c in cons_d.get("service_number_candidates", [])]
        name_cands = [NameCandidate(**c) for c in cons_d.get("train_name_candidates", [])]
        raw_ep = cons_d.get("endpoint_candidates", [])
        ep_cands = [e if isinstance(e, str) else e.get("endpoint", "") for e in raw_ep]

        consensus = PassageVisualConsensus(
            service_number_candidates=num_cands,
            stable_pair=cons_d.get("stable_pair", []),
            train_name_candidates=name_cands,
            endpoint_candidates=ep_cands,
            number_board_type=cons_d.get("number_board_type", "UNKNOWN"),
            numeric_status=cons_d.get("numeric_status", "UNKNOWN"),
        )

    return VisualOcrEvidence(
        passage_id=v_dict.get("passage_id", ""),
        ocr_model_version=v_dict.get("ocr_model_version", "v1"),
        preprocessing_profile=v_dict.get("preprocessing_profile", "default"),
        per_observation=obs_list,
        passage_consensus=consensus,
    )


def dict_to_ntes(n_dict: Optional[Dict[str, Any]]) -> Optional[NtesSnapshot]:
    if not n_dict:
        return None
    train_objs = [NtesTrainEntry(**t) for t in n_dict.get("trains", [])]
    return NtesSnapshot(
        site_id=n_dict.get("site_id", ""),
        station_code=n_dict.get("station_code", ""),
        requested_at=n_dict.get("requested_at", ""),
        fetched_at=n_dict.get("fetched_at", ""),
        window_hr=n_dict.get("window_hr", 2),
        available=n_dict.get("available", True),
        from_cache=n_dict.get("from_cache", False),
        stale_fallback=n_dict.get("stale_fallback", False),
        snapshot_age_seconds_at_decision=n_dict.get("snapshot_age_seconds_at_decision", 0.0),
        source=n_dict.get("source", "archive"),
        trains=train_objs,
        raw_hash=n_dict.get("raw_hash", ""),
    )


def run_single_replay(
    site_id: str,
    fixture_data: Dict[str, Any],
    registry: SiteRegistry,
    repo_root: Path,
    silent: bool = False,
) -> Tuple[ShadowDecision, ShadowDecision, ShadowDecision, ComparisonRecord]:
    cfg = registry.get_site(site_id)
    aliases = registry.get_aliases(site_id)

    passage = dict_to_passage(fixture_data["passage"])
    if passage.site_id != site_id:
        raise ConfigError(
            f"Cross-site violation! Fixture site '{passage.site_id}' does not match replay site '{site_id}'."
        )

    visual_evidence = dict_to_visual(fixture_data["visual_evidence"])
    ntes_before = dict_to_ntes(fixture_data.get("ntes_before"))
    ntes_after = dict_to_ntes(fixture_data.get("ntes_after"))

    sched_path = repo_root / cfg["schedule"]["path"]
    snapshot = TimetableImporter.import_schedule(
        site_id=site_id,
        station_code=cfg["station"]["code"],
        csv_path=str(sched_path),
        version=cfg["schedule"]["version"],
        effective_from=cfg["schedule"]["effective_from"],
    )

    # Run T0 (Visual + Schedule)
    dec_t0 = resolve_shadow(
        passage=passage,
        visual_evidence=visual_evidence,
        site_schedule_rows=snapshot.rows,
        ntes_before=None,
        ntes_after=None,
        site_aliases=aliases,
        config=cfg,
        stage="T0",
    )

    # Run T1 (Pre-NTES)
    dec_t1 = resolve_shadow(
        passage=passage,
        visual_evidence=visual_evidence,
        site_schedule_rows=snapshot.rows,
        ntes_before=ntes_before,
        ntes_after=None,
        site_aliases=aliases,
        config=cfg,
        stage="T1",
    )

    # Run T2 (Post-NTES +180s)
    dec_t2 = resolve_shadow(
        passage=passage,
        visual_evidence=visual_evidence,
        site_schedule_rows=snapshot.rows,
        ntes_before=ntes_before,
        ntes_after=ntes_after,
        site_aliases=aliases,
        config=cfg,
        stage="T2",
    )

    # Extract ground truth and production info
    gt_d = fixture_data.get("ground_truth", {})
    gt_num = gt_d.get("actual_train_number")
    gt_rev = gt_d.get("reviewer")
    gt_qual = gt_d.get("quality")

    prod_res = passage.production_readonly_result
    prod_num = prod_res.get("train_number")
    prod_conf = prod_res.get("confidence")
    prod_src = prod_res.get("source")

    # Categorize final T2 decision
    cat, rev_req = categorize_comparison(
        prod_num=prod_num,
        prod_conf=prod_conf,
        shadow_num=dec_t2.shadow_final_number,
        shadow_status=dec_t2.shadow_status,
        ground_truth=gt_num,
    )

    # Compute site config hash
    site_cfg_hash = compute_file_sha256(repo_root / "shadow_lab" / "sites" / site_id / "site.yaml")[:16]

    v_cons = visual_evidence.passage_consensus
    vis_nums_str = str(v_cons.stable_pair) if v_cons.stable_pair else "<none>"
    vis_name_str = v_cons.train_name_candidates[0].name if v_cons.train_name_candidates else "<not observed>"
    vis_ep_str = v_cons.endpoint_candidates[0] if v_cons.endpoint_candidates else "<none>"

    sched_cands_str = ",".join(c.train_number for c in dec_t0.candidate_services) or "<none>"
    ntes_pre_cands_str = ",".join(c.train_number for c in dec_t1.candidate_services) or "<none>"
    ntes_post_cands_str = ",".join(c.train_number for c in dec_t2.candidate_services) or "<none>"

    record = ComparisonRecord(
        site_id=site_id,
        installation_id=passage.installation_id,
        passage_id=passage.passage_id,
        passage_end_ist=passage.passage_end or "",
        possible_platforms=",".join(passage.possible_platforms),
        production_train_number=prod_num,
        production_confidence=prod_conf,
        production_source=prod_src,
        shadow_visual_numbers=vis_nums_str,
        shadow_train_name=vis_name_str,
        shadow_endpoints=vis_ep_str,
        schedule_version=snapshot.version,
        schedule_candidates=sched_cands_str,
        ntes_pre_fetched_at=ntes_before.fetched_at if ntes_before else "",
        ntes_pre_age_s=ntes_before.snapshot_age_seconds_at_decision if ntes_before else 0.0,
        ntes_pre_candidates=ntes_pre_cands_str,
        ntes_post_fetched_at=ntes_after.fetched_at if ntes_after else "",
        ntes_post_age_s=ntes_after.snapshot_age_seconds_at_decision if ntes_after else 0.0,
        ntes_post_candidates=ntes_post_cands_str,
        post_fetch_kind=dec_t2.post_fetch_kind or "UNAVAILABLE",
        shadow_t0_number=dec_t0.shadow_final_number,
        shadow_t0_status=dec_t0.shadow_status,
        shadow_t1_number=dec_t1.shadow_final_number,
        shadow_t1_status=dec_t1.shadow_status,
        shadow_t2_number=dec_t2.shadow_final_number,
        shadow_t2_status=dec_t2.shadow_status,
        shadow_reason_codes=";".join(dec_t2.reason_codes),
        ground_truth_number=gt_num,
        ground_truth_reviewer=gt_rev,
        ground_truth_quality=gt_qual,
        comparison_category=cat,
        review_required=rev_req,
        resolver_version=dec_t2.algorithm_version,
        ocr_model_version=visual_evidence.ocr_model_version,
        site_config_hash=site_cfg_hash,
    )

    if not silent:
        # Evidence trace output
        pf_disp = passage.observed_platform or f"UNKNOWN [{','.join(passage.possible_platforms)}]"
        print("=" * 70)
        print(f"SITE: {site_id}    PASSAGE: {passage.passage_id}    PLATFORM: {pf_disp}")
        print(f"VISUAL NUMBERS: {vis_nums_str}   NAME: {vis_name_str}")
        print(f"TIMETABLE: candidates [{sched_cands_str}]; matching service day/time")
        if ntes_before:
            n_status = "FRESH" if not ntes_before.stale_fallback else "STALE"
            print(f"NTES: snapshot {ntes_before.fetched_at} fetched_at, {n_status}, candidates [{ntes_pre_cands_str}]")
        else:
            print("NTES: snapshot unavailable")
        print(f"PRELIMINARY: {dec_t1.shadow_final_number or '<none>'} / {dec_t1.shadow_status} / [{';'.join(dec_t1.reason_codes[:3])}]")
        if ntes_after:
            print(f"POST+180: snapshot {ntes_after.fetched_at} fetched_at ({dec_t2.post_fetch_kind}), candidate {dec_t2.shadow_final_number}")
        else:
            print("POST+180: <not available>")
        print(f"SHADOW FINAL: {dec_t2.shadow_final_number or '<none>'} / {dec_t2.shadow_status} (shadow only)")
        print(f"PRODUCTION: {prod_num or '<none>'} / {prod_conf or '<none>'}")
        print(f"GROUND TRUTH: {gt_num or '<not labeled yet>'}")
        print(f"COMPARISON: {cat}")
        print("=" * 70)

    return dec_t0, dec_t1, dec_t2, record


def cmd_replay(args: argparse.Namespace) -> int:
    if not args.site_id:
        print("[FAIL] --site-id is mandatory for replay!")
        return 1
    if not args.fixtures:
        print("[FAIL] --fixtures <path> is mandatory for replay!")
        return 1

    repo_root = get_repo_root()
    registry = SiteRegistry(shadow_lab_root=repo_root / "shadow_lab")

    fixture_path = repo_root / args.fixtures if not Path(args.fixtures).is_absolute() else Path(args.fixtures)
    if not fixture_path.exists():
        print(f"[FAIL] Fixture file not found: {fixture_path}")
        return 1

    fixture_data = load_fixture_data(fixture_path)
    try:
        run_single_replay(args.site_id, fixture_data, registry, repo_root, silent=False)
    except Exception as e:
        print(f"[FAIL] Replay error: {e}")
        return 1

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    repo_root = get_repo_root()
    registry = SiteRegistry(shadow_lab_root=repo_root / "shadow_lab")
    evaluator = ComparisonEvaluator()

    search_dirs = []
    if getattr(args, "passages_dir", None):
        p_dir = Path(args.passages_dir)
        search_dirs.append(repo_root / p_dir if not p_dir.is_absolute() else p_dir)
    else:
        # Standard workflow: check real workspace passages first
        workspace_passages = repo_root / "shadow_lab" / "workspace" / "passages"
        if workspace_passages.exists() and list(workspace_passages.glob("*.json")):
            search_dirs.append(workspace_passages)
        search_dirs.append(repo_root / "shadow_lab" / "tests" / "fixtures")

    passage_files = []
    for s_dir in search_dirs:
        if s_dir.exists():
            passage_files.extend(sorted(list(s_dir.glob("*.json"))))

    if not passage_files:
        print("[FAIL] No passage or fixture files found to compare!")
        return 1

    # Date-based filtering for --week-start
    week_start_str = args.week_start
    w_start_dt = None
    w_end_dt = None
    if week_start_str:
        try:
            w_start_dt = datetime.fromisoformat(week_start_str)
            w_end_dt = w_start_dt + timedelta(days=7)
        except Exception:
            pass

    print(f"Executing weekly benchmark replay for week starting {week_start_str} across {len(passage_files)} candidate files...")

    replayed_count = 0
    filtered_out_count = 0
    for f_path in passage_files:
        data = load_fixture_data(f_path)
        p_info = data.get("passage", {})
        site_id = p_info.get("site_id")
        if not site_id:
            continue

        # Filter by week range if start/end timestamps are present
        p_start_str = p_info.get("passage_start") or p_info.get("passage_end")
        if w_start_dt and p_start_str:
            try:
                p_dt = datetime.fromisoformat(p_start_str)
                if p_dt.tzinfo and not w_start_dt.tzinfo:
                    w_start_cmp = w_start_dt.replace(tzinfo=p_dt.tzinfo)
                    w_end_cmp = w_end_dt.replace(tzinfo=p_dt.tzinfo)
                elif not p_dt.tzinfo and w_start_dt.tzinfo:
                    w_start_cmp = w_start_dt.replace(tzinfo=None)
                    w_end_cmp = w_end_dt.replace(tzinfo=None)
                else:
                    w_start_cmp = w_start_dt
                    w_end_cmp = w_end_dt

                if not (w_start_cmp <= p_dt <= w_end_cmp):
                    filtered_out_count += 1
                    continue
            except Exception:
                pass

        try:
            _, _, _, record = run_single_replay(site_id, data, registry, repo_root, silent=True)
            evaluator.add_record(record)
            replayed_count += 1
        except Exception as e:
            print(f"[WARN] Skipping passage {f_path.name}: {e}")

    # Generate CSVs and markdown summary
    reports_dir = repo_root / "shadow_lab" / "workspace" / "reports"
    comp_csv, mis_csv = evaluator.write_csvs(reports_dir)
    sum_md = evaluator.generate_summary_markdown(reports_dir)

    print(f"[PASS] Comparison CSV written: {comp_csv}")
    print(f"[PASS] Mismatch CSV written: {mis_csv}")
    print(f"[PASS] Weekly Summary Report written: {sum_md}")
    if filtered_out_count:
        print(f"[INFO] {filtered_out_count} passages excluded by week filter.")

    # Print summary metrics to console
    categories: Dict[str, int] = {}
    for r in evaluator.records:
        categories[r.comparison_category] = categories.get(r.comparison_category, 0) + 1

    print("\n" + "=" * 60)
    print(" SHADOW vs PRODUCTION BENCHMARK SUMMARY")
    print("=" * 60)
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}")
    print("=" * 60)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m shadow_identity.cli",
        description="Multi-Site Train Identity Shadow Lab CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # preflight
    p_pre = subparsers.add_parser("preflight", help="Run shadow preflight health check")
    p_pre.add_argument("--site-id", default=None, help="Site ID to check (defaults to all)")

    # validate-schedule
    p_val = subparsers.add_parser("validate-schedule", help="Validate site timetable integrity")
    p_val.add_argument("--site-id", required=True, help="Site ID to validate")

    # replay
    p_rep = subparsers.add_parser("replay", help="Replay passage fixture through shadow evaluator")
    p_rep.add_argument("--site-id", required=True, help="Site ID for replay")
    p_rep.add_argument("--fixtures", required=True, help="Path to fixture JSON file")

    # verify-no-writes
    p_vnw = subparsers.add_parser("verify-no-writes", help="Verify zero changes to production ARDS")
    p_vnw.add_argument("--baseline-manifest", default=None, help="Path to original hashes JSON")

    # compare
    p_cmp = subparsers.add_parser("compare", help="Generate side-by-side comparison report")
    p_cmp.add_argument("--week-start", default="2026-10-12", help="Start date of evaluation week")
    p_cmp.add_argument("--passages-dir", default=None, help="Directory containing observed passage JSONs")
    p_cmp.add_argument("--all-sites", action="store_true", help="Include all registered sites")

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "preflight": cmd_preflight,
        "validate-schedule": cmd_validate_schedule,
        "replay": cmd_replay,
        "verify-no-writes": cmd_verify_no_writes,
        "compare": cmd_compare,
    }

    code = dispatch[args.subcommand](args)
    sys.exit(code)


if __name__ == "__main__":
    main()
