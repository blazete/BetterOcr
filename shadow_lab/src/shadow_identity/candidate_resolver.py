"""
Pure Deterministic Candidate Resolver for Shadow Identity Lab
Fuses visual OCR evidence (numbers and names), site-specific timetable matching,
and time-ordered NTES snapshot evidence into explainable candidate rankings.
No network calls, no production writes.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from shadow_identity.models import (
    CandidateService,
    NtesSnapshot,
    NtesTrainEntry,
    PassageInput,
    ShadowDecision,
    TimetableRow,
    VisualOcrEvidence,
)
from shadow_identity.reason_codes import (
    RC_AMBIGUOUS_MULTI_CANDIDATE,
    RC_CO_PRESENT_TRAINS_UNRESOLVED,
    RC_DIRECTION_CONFLICT,
    RC_DIRECTION_CORROBORATED,
    RC_MANUAL_REVIEW_REQUIRED,
    RC_NOT_SERVICE_EVENT,
    RC_NTES_CORROBORATED_FRESH,
    RC_NTES_CORROBORATED_STALE,
    RC_NTES_POST_180S_ABSENT_IGNORED,
    RC_NTES_POST_180S_CONFIRMED,
    RC_NTES_UNAVAILABLE,
    RC_NTES_WRONG_STATION_REJECTED,
    RC_NUMBER_NAME_CONFLICT,
    RC_PLATFORM_AMBIGUOUS_ISLAND,
    RC_PLATFORM_VERIFIED,
    RC_SCHEDULE_EXACT_NUMBER_MATCH,
    RC_SCHEDULE_NAME_MATCH,
    RC_SCHEDULE_OPERATING_DAY_MATCH,
    RC_SCHEDULE_OUTSIDE_WINDOW,
    RC_SCHEDULE_TIME_WINDOW_MATCH,
    RC_SCHEDULE_UNAVAILABLE,
    RC_SCHEDULE_WRONG_STATION_REJECTED,
    RC_TIMETABLE_NTES_CONFLICT,
    RC_VISUAL_BOARD_NOT_NUMBERED,
    RC_VISUAL_COMPLEX_BOARD,
    RC_VISUAL_NAME_STRONG,
    RC_VISUAL_NTES_CONFLICT,
    RC_VISUAL_NUMBERS_UNAVAILABLE,
    RC_VISUAL_PAIR_ACTIVE_SERVICE_UNCONFIRMED,
    RC_VISUAL_PAIR_CONFIRMED,
    RC_VISUAL_PAIR_PROVISIONAL,
    RC_VISUAL_SINGLE_CANDIDATE,
    RC_VISUAL_TIMETABLE_CONFLICT,
)
from shadow_identity.text_normalization import match_name_similarity, match_name_with_aliases
from shadow_identity.timetable_matcher import TimetableMatcher


def resolve_shadow(
    passage: PassageInput,
    visual_evidence: VisualOcrEvidence,
    site_schedule_rows: List[TimetableRow],
    ntes_before: Optional[NtesSnapshot] = None,
    ntes_after: Optional[NtesSnapshot] = None,
    site_aliases: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    stage: str = "T1",
) -> ShadowDecision:
    """
    Pure decision function implementing explainable candidate resolution.
    """
    reason_codes: List[str] = []
    candidates_map: Dict[str, CandidateService] = {}

    site_id = passage.site_id
    passage_id = passage.passage_id
    dt_str = passage.passage_start or passage.passage_end
    
    # Parse passage datetime
    try:
        if dt_str:
            passage_dt = datetime.fromisoformat(dt_str)
        else:
            passage_dt = datetime.now()
    except Exception:
        passage_dt = datetime.now()

    # 1. Check schedule availability
    matcher = TimetableMatcher(site_schedule_rows, site_id=site_id)
    if not matcher.rows:
        reason_codes.append(RC_SCHEDULE_UNAVAILABLE)

    # 2. Extract visual features
    v_cons = visual_evidence.passage_consensus
    num_cands = [c.number for c in v_cons.service_number_candidates if c.distinct_coaches >= 1]
    strong_num_cands = [c.number for c in v_cons.service_number_candidates if c.distinct_coaches >= 3]
    stable_pair = v_cons.stable_pair
    top_names = [c.name for c in v_cons.train_name_candidates]
    top_name = top_names[0] if top_names else None

    # Tag visual reason codes
    if v_cons.numeric_status == "PAIR_CONFIRMED":
        reason_codes.append(RC_VISUAL_PAIR_CONFIRMED)
    elif v_cons.numeric_status == "PAIR_PROVISIONAL":
        reason_codes.append(RC_VISUAL_PAIR_PROVISIONAL)
    elif v_cons.numeric_status == "NOT_NUMBERED":
        reason_codes.append(RC_VISUAL_BOARD_NOT_NUMBERED)
    elif v_cons.numeric_status == "COMPLEX_BOARD":
        reason_codes.append(RC_VISUAL_COMPLEX_BOARD)
    else:
        reason_codes.append(RC_VISUAL_NUMBERS_UNAVAILABLE)

    if top_name and v_cons.train_name_candidates and v_cons.train_name_candidates[0].distinct_coaches >= 2:
        reason_codes.append(RC_VISUAL_NAME_STRONG)

    # Platform ambiguity
    if not passage.observed_platform and len(passage.possible_platforms) > 1:
        reason_codes.append(RC_PLATFORM_AMBIGUOUS_ISLAND)

    # Extract platform list cleanly without precedence ambiguity
    if passage.possible_platforms:
        pfs = [str(p) for p in passage.possible_platforms]
    elif passage.observed_platform:
        pfs = [str(passage.observed_platform)]
    else:
        pfs = []

    # 3. Match against site timetable
    sched_matches = matcher.find_candidates(
        event_dt=passage_dt,
        possible_platforms=pfs,
        direction=passage.direction,
        target_numbers=stable_pair or num_cands,
        target_name=top_name,
        site_aliases=site_aliases,
    )

    for sm in sched_matches:
        t_no = sm["train_number"]
        c_obj = CandidateService(
            train_number=t_no,
            train_name=sm["train_name"],
            score=sm["score"],
            evidence_list=list(sm["evidence"]),
            contradictions=[],
        )
        candidates_map[t_no] = c_obj

    # Register visual candidates with >= 3 distinct coaches in candidates_map
    for v_num in strong_num_cands:
        if v_num not in candidates_map:
            candidates_map[v_num] = CandidateService(
                train_number=v_num,
                train_name=top_name or "",
                score=4.5,
                evidence_list=["VISUAL_NUMBER_GE3_COACHES"],
                contradictions=[],
            )
        else:
            candidates_map[v_num].evidence_list.append("VISUAL_NUMBER_GE3_COACHES")
            candidates_map[v_num].score += 4.5

    # 4. Process NTES before/at passage (Stage T1 baseline)
    ntes_active = ntes_before if ntes_before is not None else (ntes_after if stage == "T2" else None)
    is_fresh_ntes = False
    ntes_trains: List[NtesTrainEntry] = []

    if ntes_active and ntes_active.available:
        # P0 Check 1: Enforce Station and Site Isolation
        if ntes_active.site_id != site_id:
            reason_codes.append(RC_NTES_WRONG_STATION_REJECTED)
            reason_codes.append(RC_NTES_UNAVAILABLE)
            ntes_active = None
        elif config and config.get("station", {}).get("code") and ntes_active.station_code:
            expected_code = config.get("station", {}).get("code")
            if ntes_active.station_code != expected_code:
                reason_codes.append(RC_NTES_WRONG_STATION_REJECTED)
                reason_codes.append(RC_NTES_UNAVAILABLE)
                ntes_active = None

    if ntes_active and ntes_active.available:
        # P0 Check 2: Snapshot TTL and Freshness Validation
        ttl_seconds = config.get("ntes_default_ttl_s", 180) if config else 180
        snapshot_age = ntes_active.snapshot_age_seconds_at_decision

        if ntes_active.fetched_at:
            try:
                fetched_dt = datetime.fromisoformat(ntes_active.fetched_at)
                age_calc = (passage_dt - fetched_dt).total_seconds()
                if age_calc > 0:
                    snapshot_age = max(snapshot_age, age_calc)
            except Exception:
                pass

        if ntes_active.stale_fallback or snapshot_age > ttl_seconds:
            reason_codes.append(RC_NTES_CORROBORATED_STALE)
            is_fresh_ntes = False
        else:
            reason_codes.append(RC_NTES_CORROBORATED_FRESH)
            is_fresh_ntes = True

        # P0 Check 3: Strict Platform Filtering (Unrelated platforms MUST NOT influence resolution)
        ntes_trains: List[NtesTrainEntry] = []
        if pfs:
            for pf in pfs:
                ntes_trains.extend(ntes_active.get_trains_for_platform(pf))
        else:
            ntes_trains = list(ntes_active.trains)

        for nt in ntes_trains:
            t_no = nt.train_number
            if t_no not in candidates_map:
                candidates_map[t_no] = CandidateService(
                    train_number=t_no,
                    train_name=nt.train_name,
                    score=2.0,
                    evidence_list=[],
                    contradictions=[],
                )
            candidates_map[t_no].evidence_list.append(f"NTES_PRESENT_PF_{nt.platform}")
            candidates_map[t_no].score += 3.0

            # Corroborate with visual number pair
            if stable_pair and t_no in stable_pair:
                candidates_map[t_no].evidence_list.append("CORROBORATION_VISUAL_PAIR_AND_NTES")
                candidates_map[t_no].score += 4.0
            
            # Corroborate with visual name via alias-aware matching
            if top_name:
                sim = match_name_with_aliases(top_name, nt.train_name, site_aliases=site_aliases)
                if sim >= 0.6:
                    candidates_map[t_no].evidence_list.append("CORROBORATION_VISUAL_NAME_AND_NTES")
                    candidates_map[t_no].score += 3.0
    else:
        if RC_NTES_UNAVAILABLE not in reason_codes:
            reason_codes.append(RC_NTES_UNAVAILABLE)

    # 5. Process Stage T2 (+180s post-passage check)
    post_kind = "UNAVAILABLE"
    if stage == "T2":
        if ntes_after and ntes_after.available and ntes_after.site_id == site_id:
            post_kind = "NEW_FETCH" if not ntes_after.from_cache else "CACHE_REUSE"
            # Check if candidates are still confirmed or corroborated
            for t_no, cand in candidates_map.items():
                if any(nt.train_number == t_no for nt in ntes_after.trains):
                    cand.evidence_list.append("NTES_POST_180S_STILL_CONFIRMED")
                    cand.score += 1.5
                    reason_codes.append(RC_NTES_POST_180S_CONFIRMED)
                else:
                    # Train absence after passage is non-negative proof (train likely departed)
                    cand.evidence_list.append("NTES_POST_180S_ABSENT_POSSIBLE_DEPARTURE")
                    reason_codes.append(RC_NTES_POST_180S_ABSENT_IGNORED)
        else:
            post_kind = "UNAVAILABLE"

    # Platform-level Timetable vs NTES conflict check
    sched_active_numbers = {
        sm["train_number"]
        for sm in sched_matches
        if "SCHEDULE_OUTSIDE_TIME_WINDOW" not in sm.get("evidence", [])
    }
    ntes_active_numbers = {nt.train_number for nt in ntes_trains}
    if sched_active_numbers and ntes_active_numbers and not (sched_active_numbers & ntes_active_numbers):
        if RC_TIMETABLE_NTES_CONFLICT not in reason_codes:
            reason_codes.append(RC_TIMETABLE_NTES_CONFLICT)

    # 6. Rank candidates and resolve final decision
    ranked_candidates = sorted(candidates_map.values(), key=lambda c: c.score, reverse=True)

    final_number: Optional[str] = None
    final_status = "UNKNOWN"

    if not ranked_candidates:
        if v_cons.numeric_status == "NOT_NUMBERED" and not matcher.rows:
            final_status = "DATA_UNAVAILABLE"
        elif v_cons.numeric_status == "NOT_NUMBERED":
            final_status = "AMBIGUOUS"
        else:
            final_status = "UNKNOWN"

    else:
        top_cand = ranked_candidates[0]
        runner_up = ranked_candidates[1] if len(ranked_candidates) > 1 else None

        # Check for conflicts
        if stable_pair and top_cand.train_number not in stable_pair and v_cons.numeric_status == "PAIR_CONFIRMED":
            has_ntes = any("NTES_" in ev for ev in top_cand.evidence_list)
            has_tt = any("TIMETABLE_" in ev or "SCHEDULE_" in ev for ev in top_cand.evidence_list)
            if has_ntes:
                reason_codes.append(RC_VISUAL_NTES_CONFLICT)
            if has_tt:
                reason_codes.append(RC_VISUAL_TIMETABLE_CONFLICT)
            if not has_ntes and not has_tt:
                reason_codes.append(RC_VISUAL_NTES_CONFLICT)
            final_status = "CONFLICT"
            final_number = None

        elif len(strong_num_cands) == 1 and top_cand.train_number != strong_num_cands[0]:
            vis_single = strong_num_cands[0]
            has_ntes = any("NTES_" in ev for ev in top_cand.evidence_list)
            if has_ntes:
                reason_codes.append(RC_VISUAL_NTES_CONFLICT)
                final_status = "CONFLICT"
                final_number = None
            else:
                # Strong single-number visual candidate protected from timetable-only replacement (§1)
                reason_codes.append(RC_VISUAL_TIMETABLE_CONFLICT)
                final_status = "PROVISIONAL"
                final_number = vis_single

        elif top_cand.train_number.startswith("6"):
            reason_codes.append(RC_NOT_SERVICE_EVENT)
            final_status = "NOT_SERVICE_EVENT"
            final_number = top_cand.train_number

        # Pathway A: Confirmed Visual Pair
        elif v_cons.numeric_status == "PAIR_CONFIRMED":
            # If runner-up has nearly equal score within the pair (bidirectional ambiguity)
            if runner_up and runner_up.train_number in stable_pair and abs(top_cand.score - runner_up.score) < 0.5:
                reason_codes.append(RC_AMBIGUOUS_MULTI_CANDIDATE)
                final_status = "AMBIGUOUS"
                final_number = None
            else:
                # P0: Require unique, fresh corroboration that this candidate is the ACTIVE service
                has_fresh_ntes = any("CORROBORATION_VISUAL_PAIR_AND_NTES" in ev for ev in top_cand.evidence_list) and is_fresh_ntes
                has_direction = any(RC_DIRECTION_CORROBORATED in ev for ev in top_cand.evidence_list)
                has_strong_name = any("CORROBORATION_VISUAL_NAME" in ev or "TIMETABLE_NAME_MATCH" in ev for ev in top_cand.evidence_list)

                if sched_active_numbers and top_cand.train_number not in sched_active_numbers:
                    if RC_VISUAL_TIMETABLE_CONFLICT not in reason_codes:
                        reason_codes.append(RC_VISUAL_TIMETABLE_CONFLICT)

                if has_fresh_ntes or has_direction or has_strong_name:
                    final_status = "CONFIRMED"
                    final_number = top_cand.train_number
                else:
                    # Pair is verified on coach boards, but active running direction is unconfirmed without live corroboration
                    reason_codes.append(RC_VISUAL_PAIR_ACTIVE_SERVICE_UNCONFIRMED)
                    final_status = "PROVISIONAL"
                    final_number = top_cand.train_number

        # Pathway B: Provisional Visual Pair
        elif v_cons.numeric_status == "PAIR_PROVISIONAL":
            if len(strong_num_cands) == 1:
                vis_single = strong_num_cands[0]
                if sched_active_numbers and vis_single not in sched_active_numbers:
                    if RC_VISUAL_TIMETABLE_CONFLICT not in reason_codes:
                        reason_codes.append(RC_VISUAL_TIMETABLE_CONFLICT)
                if ntes_active_numbers and vis_single not in ntes_active_numbers:
                    if RC_VISUAL_NTES_CONFLICT not in reason_codes:
                        reason_codes.append(RC_VISUAL_NTES_CONFLICT)
                    final_status = "CONFLICT"
                    final_number = None
                else:
                    final_status = "PROVISIONAL"
                    final_number = vis_single
            else:
                final_status = "PROVISIONAL"
                final_number = top_cand.train_number

        # Pathway C: Complex 4/6-Number Board
        elif v_cons.numeric_status == "COMPLEX_BOARD":
            # Must intersect candidate numbers with train name / site aliases to isolate service family
            if top_name:
                resolved_cands = []
                for cand in ranked_candidates:
                    if match_name_with_aliases(top_name, cand.train_name, site_aliases=site_aliases) >= 0.6:
                        resolved_cands.append(cand)
                if resolved_cands:
                    best_cand = sorted(resolved_cands, key=lambda c: c.score, reverse=True)[0]
                    final_status = "PROVISIONAL"
                    final_number = best_cand.train_number
                else:
                    reason_codes.append(RC_AMBIGUOUS_MULTI_CANDIDATE)
                    final_status = "AMBIGUOUS"
                    final_number = None
            else:
                reason_codes.append(RC_AMBIGUOUS_MULTI_CANDIDATE)
                final_status = "AMBIGUOUS"
                final_number = None

        # Pathway D: Non-Numeric Board (Name + Route Endpoints Fallback)
        elif v_cons.numeric_status == "NOT_NUMBERED":
            if any("TIMETABLE_NAME_MATCH" in ev or "CORROBORATION_VISUAL_NAME" in ev for ev in top_cand.evidence_list):
                if runner_up and abs(top_cand.score - runner_up.score) < 0.5:
                    reason_codes.append(RC_AMBIGUOUS_MULTI_CANDIDATE)
                    final_status = "AMBIGUOUS"
                    final_number = None
                else:
                    final_status = "PROVISIONAL"
                    final_number = top_cand.train_number
            else:
                final_status = "UNKNOWN"
                final_number = None

        else:
            if runner_up and abs(top_cand.score - runner_up.score) < 0.5:
                reason_codes.append(RC_AMBIGUOUS_MULTI_CANDIDATE)
                final_status = "AMBIGUOUS"
                final_number = None
            else:
                final_status = "UNKNOWN"
                final_number = None

    return ShadowDecision(
        site_id=site_id,
        passage_id=passage_id,
        algorithm_version="shadow_resolver_v0.1",
        stage=stage,
        evaluated_at=datetime.now().isoformat(),
        candidate_services=ranked_candidates[:5],
        shadow_final_number=final_number,
        shadow_status=final_status,
        reason_codes=sorted(list(set(reason_codes))),
        post_fetch_kind=post_kind if stage == "T2" else None,
    )
