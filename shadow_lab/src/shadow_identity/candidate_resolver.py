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
    RC_VISUAL_BOARD_NOT_NUMBERED,
    RC_VISUAL_COMPLEX_BOARD,
    RC_VISUAL_NAME_STRONG,
    RC_VISUAL_NTES_CONFLICT,
    RC_VISUAL_NUMBERS_UNAVAILABLE,
    RC_VISUAL_PAIR_CONFIRMED,
    RC_VISUAL_PAIR_PROVISIONAL,
    RC_VISUAL_SINGLE_CANDIDATE,
)
from shadow_identity.text_normalization import match_name_similarity
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

    # 3. Match against site timetable
    sched_matches = matcher.find_candidates(
        event_dt=passage_dt,
        possible_platforms=passage.possible_platforms,
        direction=passage.direction,
        target_numbers=stable_pair or num_cands,
        target_name=top_name,
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

    # 4. Process NTES before/at passage (Stage T1)
    ntes_active = ntes_before if stage in ("T0", "T1") else (ntes_after or ntes_before)

    if ntes_active and ntes_active.available:
        if ntes_active.stale_fallback:
            reason_codes.append(RC_NTES_CORROBORATED_STALE)
        else:
            reason_codes.append(RC_NTES_CORROBORATED_FRESH)

        # Platforms to check
        pfs = passage.possible_platforms or [passage.observed_platform] if passage.observed_platform else []
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
            
            # Corroborate with visual name
            if top_name and match_name_similarity(top_name, nt.train_name) >= 0.6:
                candidates_map[t_no].evidence_list.append("CORROBORATION_VISUAL_NAME_AND_NTES")
                candidates_map[t_no].score += 3.0
    else:
        reason_codes.append(RC_NTES_UNAVAILABLE)

    # 5. Process Stage T2 (+180s post-passage check)
    post_kind = "UNAVAILABLE"
    if stage == "T2":
        if ntes_after and ntes_after.available:
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
            reason_codes.append(RC_VISUAL_NTES_CONFLICT)
            final_status = "CONFLICT"
            final_number = None

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
                final_status = "CONFIRMED"
                final_number = top_cand.train_number

        # Pathway B: Provisional Visual Pair
        elif v_cons.numeric_status == "PAIR_PROVISIONAL":
            final_status = "PROVISIONAL"
            final_number = top_cand.train_number

        # Pathway C: Complex 4/6-Number Board
        elif v_cons.numeric_status == "COMPLEX_BOARD":
            # Must have name match to corroborate
            if any("TIMETABLE_NAME_MATCH" in ev or "CORROBORATION_VISUAL_NAME" in ev for ev in top_cand.evidence_list):
                final_status = "PROVISIONAL"
                final_number = top_cand.train_number
            else:
                reason_codes.append(RC_AMBIGUOUS_MULTI_CANDIDATE)
                final_status = "AMBIGUOUS"
                final_number = None

        # Pathway D: Non-Numeric Board (Name + Route Endpoints Fallback)
        elif v_cons.numeric_status == "NOT_NUMBERED":
            if any("TIMETABLE_NAME_MATCH" in ev for ev in top_cand.evidence_list):
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
