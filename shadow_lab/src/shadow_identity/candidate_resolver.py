"""
Pure Deterministic Candidate Resolver for Shadow Identity Lab
Fuses visual OCR evidence (numbers and names), site-specific timetable matching,
and time-ordered NTES snapshot evidence into explainable candidate rankings.
No network calls, no production writes.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from shadow_identity.models import (
    CandidateService,
    DirectionEvidence,
    NtesSnapshot,
    NtesTrainEntry,
    PassageInput,
    ShadowDecision,
    TimetableRow,
    VisualOcrEvidence,
)
from shadow_identity.reason_codes import (
    RC_ACTIVE_DIRECTION_CORROBORATION,
    RC_AMBIGUOUS_MULTI_CANDIDATE,
    RC_CO_PRESENT_TRAINS_UNRESOLVED,
    RC_DIRECTION_CONFLICT,
    RC_DIRECTION_CORROBORATED,
    RC_MANUAL_REVIEW_REQUIRED,
    RC_NAME_MATCH,
    RC_NAME_ONLY_ACTIVE_DIRECTION_UNVERIFIED,
    RC_NAME_RECOGNIZED_COACH_UNVERIFIED,
    RC_NAME_RECOGNIZED_SINGLE_OBSERVATION,
    RC_NAME_REPEATED_VERIFIED_COACHES,
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
    RC_SERVICE_FAMILY_MAPPING,
    RC_SERVICE_FAMILY_MATCHED,
    RC_SERVICE_MAPPING_UNVERIFIED,
    RC_SITE_SCHEDULE_MATCH,
    RC_TIMETABLE_NTES_CONFLICT,
    RC_VISUAL_BOARD_NOT_NUMBERED,
    RC_VISUAL_COMPLEX_BOARD,
    RC_VISUAL_NAME_STRONG,
    RC_VISUAL_NAME_WEAK,
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

STATION_CODE_ALIASES: Dict[str, str] = {
    "ASR": "AMRITSAR",
    "LKO": "LUCKNOW",
    "LJN": "LUCKNOW",
    "SVDK": "SHRI MATA VAISHNO DEVI KATRA",
    "HAPA": "HAPA",
    "NDLS": "NEW DELHI",
    "DLI": "DELHI",
    "NZM": "HAZRAT NIZAMUDDIN",
    "VGLJ": "VGL JHANSI",
    "GWL": "GWALIOR",
    "AGC": "AGRA CANTT",
    "BSB": "VARANASI",
    "BSBS": "BANARAS",
    "CNB": "KANPUR CENTRAL",
    "HWH": "HOWRAH",
    "SDAH": "SEALDAH",
    "NED": "NANDED",
    "SBC": "BENGALURU",
    "MMCT": "MUMBAI CENTRAL",
}


def canonicalize_endpoint(ep: str) -> str:
    cleaned = ep.strip().upper()
    return STATION_CODE_ALIASES.get(cleaned, cleaned)


def is_direction_verified(dir_ev: Optional[DirectionEvidence]) -> bool:
    """
    Pure logical predicate for direction verification (§Part 3 A1):
    active_direction_verified = (
        direction_evidence is not None
        and direction_evidence.authentic is True
        and direction_evidence.fresh is True
        and direction_evidence.passage_link_verified is True
        and direction_evidence.source_authorized is True
        and direction_evidence.direction_matches_candidate is True
    )
    Never default missing properties to True.
    """
    if dir_ev is None:
        return False
    return bool(
        dir_ev.authentic is True
        and dir_ev.fresh is True
        and dir_ev.passage_link_verified is True
        and dir_ev.source_authorized is True
        and dir_ev.direction_matches_candidate is True
        and not getattr(dir_ev, "has_conflict", False)
        and not getattr(dir_ev, "is_synthetic", False)
    )


def resolve_shadow(
    passage: PassageInput,
    visual_evidence: VisualOcrEvidence,
    site_schedule_rows: List[TimetableRow],
    ntes_before: Optional[NtesSnapshot] = None,
    ntes_after: Optional[NtesSnapshot] = None,
    site_aliases: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    stage: str = "T1",
    direction_evidence: Optional[DirectionEvidence] = None,
) -> ShadowDecision:
    """
    Pure decision function implementing explainable candidate resolution.
    """
    reason_codes: List[str] = []
    candidates_map: Dict[str, CandidateService] = {}

    site_id = passage.site_id
    passage_id = passage.passage_id
    dt_str = passage.passage_start or passage.passage_end

    if site_aliases is None and site_id:
        try:
            alias_path = Path(__file__).resolve().parents[2] / "sites" / site_id / "aliases.yaml"
            if alias_path.is_file():
                import yaml
                with open(alias_path, "r", encoding="utf-8") as f:
                    site_aliases = yaml.safe_load(f)
        except Exception:
            pass
    
    # Parse passage datetime: missing timestamps must explicitly disable time corroboration
    passage_dt = None
    if dt_str:
        try:
            passage_dt = datetime.fromisoformat(dt_str)
        except Exception:
            passage_dt = None

    effective_dir_ev = direction_evidence if direction_evidence is not None else getattr(passage, "direction_evidence", None)
    if effective_dir_ev is not None:
        if (
            getattr(effective_dir_ev, "has_conflict", False)
            or (effective_dir_ev.authentic and not effective_dir_ev.direction_matches_candidate)
        ):
            reason_codes.append(RC_DIRECTION_CONFLICT)

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
    recognized_train_name = getattr(v_cons, "recognized_train_name", None) or top_name

    alias_dict = site_aliases.get("aliases", {}) if site_aliases else {}
    service_family_candidates: List[str] = []

    lookup_name = recognized_train_name or top_name
    if lookup_name:
        matched_num_sets: List[List[str]] = []
        if alias_dict:
            for alias_key, alias_data in alias_dict.items():
                can_n = alias_data.get("canonical_name", alias_key)
                vars_list = alias_data.get("variations", [])
                all_n = [can_n, alias_key] + vars_list
                if any(match_name_similarity(lookup_name, v) >= 0.6 or lookup_name.upper() == v.upper() for v in all_n):
                    ver_state = alias_data.get("verification_state", "VERIFIED")
                    if ver_state == "VERIFIED":
                        s_nos = [str(n) for n in alias_data.get("service_numbers", [])]
                        if s_nos:
                            matched_num_sets.append(s_nos)

            distinct_num_tuples = {tuple(sorted(nos)) for nos in matched_num_sets}
            if len(distinct_num_tuples) > 1:
                # Contradictory service mappings -> fail closed (§A4 Rule 6)
                service_family_candidates = []
                reason_codes.append(RC_SERVICE_MAPPING_UNVERIFIED)
            elif len(distinct_num_tuples) == 1:
                service_family_candidates = list(matched_num_sets[0])
                reason_codes.append(RC_NAME_MATCH)
                reason_codes.append(RC_SERVICE_FAMILY_MATCHED)
                reason_codes.append(RC_SERVICE_FAMILY_MAPPING)

        if not service_family_candidates and RC_SERVICE_MAPPING_UNVERIFIED not in reason_codes:
            from shadow_identity.shadow_ocr.text_entities import TRAIN_NAME_REGISTRY
            matching_entries = []
            for reg_name, reg_entry in TRAIN_NAME_REGISTRY.items():
                all_n = [reg_name, reg_entry.canonical_name] + reg_entry.aliases
                if any(match_name_similarity(lookup_name, v) >= 0.6 or lookup_name.upper() == v.upper() for v in all_n):
                    matching_entries.append(reg_entry)
            if len(matching_entries) == 1:
                reg_entry = matching_entries[0]
                if getattr(reg_entry, "verification_state", "VERIFIED") == "VERIFIED":
                    service_family_candidates = list(reg_entry.service_family_numbers)
                    if service_family_candidates:
                        reason_codes.append(RC_NAME_MATCH)
                        reason_codes.append(RC_SERVICE_FAMILY_MATCHED)
                        reason_codes.append(RC_SERVICE_FAMILY_MAPPING)
            elif len(matching_entries) > 1:
                reason_codes.append(RC_AMBIGUOUS_MULTI_CANDIDATE)
                reason_codes.append(RC_SERVICE_MAPPING_UNVERIFIED)
        if not service_family_candidates and RC_SERVICE_MAPPING_UNVERIFIED not in reason_codes:
            reason_codes.append(RC_SERVICE_MAPPING_UNVERIFIED)

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

    if lookup_name and v_cons.train_name_candidates:
        top_name_cand = v_cons.train_name_candidates[0]
        coaches_verified = getattr(passage, "coach_intervals_verified", False) or (
            getattr(v_cons, "distinct_coaches_total", 0) > 0
            or getattr(v_cons, "coach_tracking_quality", "") == "VERIFIED"
        )
        distinct_coaches = getattr(top_name_cand, "distinct_coaches", 1)
        total_obs = getattr(top_name_cand, "total_observations", 1)

        if not coaches_verified:
            reason_codes.append(RC_NAME_RECOGNIZED_COACH_UNVERIFIED)
        elif distinct_coaches >= 2:
            reason_codes.append(RC_NAME_REPEATED_VERIFIED_COACHES)
            reason_codes.append(RC_VISUAL_NAME_STRONG)
        else:
            reason_codes.append(RC_NAME_RECOGNIZED_SINGLE_OBSERVATION)
            reason_codes.append(RC_VISUAL_NAME_WEAK)

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

    # Register provisional visual candidates (e.g. from stable pair or service numbers)
    for v_num in (stable_pair or num_cands):
        if v_num not in strong_num_cands:
            if v_num not in candidates_map:
                candidates_map[v_num] = CandidateService(
                    train_number=v_num,
                    train_name=top_name or "",
                    score=3.5,
                    evidence_list=["VISUAL_NUMBER_PROVISIONAL"],
                    contradictions=[],
                )
            else:
                candidates_map[v_num].evidence_list.append("VISUAL_NUMBER_PROVISIONAL")
                candidates_map[v_num].score += 3.5

    # Register verified service family candidates for name-only boards (§Task 3)
    if v_cons.numeric_status == "NOT_NUMBERED" and service_family_candidates:
        for s_num in service_family_candidates:
            if s_num not in candidates_map:
                candidates_map[s_num] = CandidateService(
                    train_number=s_num,
                    train_name=lookup_name or "",
                    score=3.0,
                    evidence_list=[f"CURATED_NAME_ALIAS_MATCH:{lookup_name}", "SERVICE_FAMILY_CANDIDATE"],
                    contradictions=[],
                )

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

        if ntes_active.fetched_at and passage_dt is not None:
            try:
                fetched_dt = datetime.fromisoformat(ntes_active.fetched_at)
                age_calc = (passage_dt - fetched_dt).total_seconds()
                if age_calc > 0:
                    snapshot_age = max(snapshot_age, age_calc)
            except Exception:
                pass

        is_synthetic = (
            getattr(ntes_active, "is_synthetic", False)
            or "SYNTHETIC" in (getattr(ntes_active, "source", "") or "").upper()
            or getattr(ntes_active, "raw_hash", "").startswith("hash_synthetic")
        )

        if is_synthetic:
            # Synthetic NTES fixture never counts as live corroboration (§A1)
            is_fresh_ntes = False
        elif ntes_active.stale_fallback or snapshot_age > ttl_seconds:
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

            # Corroborate with visual number pair only if fresh & authentic
            if is_fresh_ntes:
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

    # 5.5 Curated Route Endpoint and Alias Matching (Package P1.2)
    obs_endpoints = [canonicalize_endpoint(ep) for ep in (v_cons.endpoint_candidates or [])]
    alias_dict = site_aliases.get("aliases", {}) if site_aliases else {}

    for cand_no, cand in list(candidates_map.items()):
        for alias_key, alias_data in alias_dict.items():
            srv_nos = [str(n) for n in alias_data.get("service_numbers", [])]
            if cand_no in srv_nos:
                # 1. Curated Name Alias Match
                if top_name:
                    canonical_name = alias_data.get("canonical_name", alias_key)
                    variations = alias_data.get("variations", [])
                    all_names = [canonical_name] + variations
                    if any(match_name_similarity(top_name, name_var) >= 0.6 for name_var in all_names):
                        ev_name = f"CURATED_NAME_ALIAS_MATCH:{canonical_name}"
                        if ev_name not in cand.evidence_list:
                            cand.evidence_list.append(ev_name)
                            cand.score += 2.0

                # 2. Curated Route Endpoint Match
                curated_eps = [canonicalize_endpoint(ep) for ep in alias_data.get("endpoints", [])]
                matching_eps = [ep for ep in obs_endpoints if ep in curated_eps]
                distinct_matches = list(dict.fromkeys(matching_eps))
                if len(distinct_matches) >= 2:
                    ev_ep = f"ROUTE_ENDPOINT_PAIR_MATCH:{'+'.join(distinct_matches)}"
                    if ev_ep not in cand.evidence_list:
                        cand.evidence_list.append(ev_ep)
                        cand.score += 3.5
                elif len(distinct_matches) == 1:
                    ev_ep = f"ROUTE_ENDPOINT_MATCH:{distinct_matches[0]}"
                    if ev_ep not in cand.evidence_list:
                        cand.evidence_list.append(ev_ep)
                        cand.score += 2.0

    # 6. Rank candidates and resolve final decision
    ranked_candidates = sorted(candidates_map.values(), key=lambda c: c.score, reverse=True)

    final_number: Optional[str] = None
    final_status = "UNKNOWN"
    active_service_number: Optional[str] = None
    active_direction_verified: bool = False

    if not ranked_candidates:
        if v_cons.numeric_status == "NOT_NUMBERED":
            if service_family_candidates:
                final_status = "AMBIGUOUS"
                reason_codes.append(RC_NAME_ONLY_ACTIVE_DIRECTION_UNVERIFIED)
                reason_codes.append(RC_AMBIGUOUS_MULTI_CANDIDATE)
            elif lookup_name:
                final_status = "UNKNOWN"
                if RC_SERVICE_MAPPING_UNVERIFIED not in reason_codes:
                    reason_codes.append(RC_SERVICE_MAPPING_UNVERIFIED)
            elif not matcher.rows:
                final_status = "DATA_UNAVAILABLE"
            else:
                final_status = "UNKNOWN"
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
            runner_up_in_pair = runner_up and runner_up.train_number in (stable_pair or [])
            is_score_tied = runner_up_in_pair and abs(top_cand.score - runner_up.score) < 0.5

            if is_score_tied:
                reason_codes.append(RC_AMBIGUOUS_MULTI_CANDIDATE)
                final_status = "AMBIGUOUS"
                final_number = None
            else:
                has_fresh_ntes = any("CORROBORATION_VISUAL_PAIR_AND_NTES" in ev for ev in top_cand.evidence_list) and is_fresh_ntes
                has_direction = is_direction_verified(effective_dir_ev)
                top_has_diff = any(
                    "TIMETABLE_TIME_WINDOW" in ev or "TIMETABLE_EXACT_NUMBER" in ev or "ROUTE_ENDPOINT" in ev
                    for ev in top_cand.evidence_list
                )

                if sched_active_numbers and top_cand.train_number not in sched_active_numbers:
                    if RC_VISUAL_TIMETABLE_CONFLICT not in reason_codes:
                        reason_codes.append(RC_VISUAL_TIMETABLE_CONFLICT)

                if has_fresh_ntes or has_direction:
                    final_status = "CONFIRMED"
                    final_number = top_cand.train_number
                elif top_has_diff:
                    reason_codes.append(RC_VISUAL_PAIR_ACTIVE_SERVICE_UNCONFIRMED)
                    final_status = "PROVISIONAL"
                    final_number = top_cand.train_number
                else:
                    reason_codes.append(RC_VISUAL_PAIR_ACTIVE_SERVICE_UNCONFIRMED)
                    final_status = "AMBIGUOUS"
                    final_number = None

        # Pathway B: Provisional Visual Pair
        elif v_cons.numeric_status == "PAIR_PROVISIONAL":
            # Conflict check for single visual candidate
            if len(strong_num_cands) == 1:
                vis_single = strong_num_cands[0]
                if sched_active_numbers and vis_single not in sched_active_numbers:
                    if RC_VISUAL_TIMETABLE_CONFLICT not in reason_codes:
                        reason_codes.append(RC_VISUAL_TIMETABLE_CONFLICT)
                if ntes_active_numbers and vis_single not in ntes_active_numbers:
                    if RC_VISUAL_NTES_CONFLICT not in reason_codes:
                        reason_codes.append(RC_VISUAL_NTES_CONFLICT)

            has_independent_corroboration = (
                (any("CORROBORATION_VISUAL_PAIR_AND_NTES" in ev for ev in top_cand.evidence_list) and is_fresh_ntes)
                or is_direction_verified(effective_dir_ev)
                or (passage_dt is not None and bool(pfs) and any("TIMETABLE_TIME_WINDOW" in ev for ev in top_cand.evidence_list))
                or (passage_dt is not None and bool(pfs) and any("ROUTE_ENDPOINT" in ev for ev in top_cand.evidence_list))
            )

            if not has_independent_corroboration:
                if len(num_cands) >= 2 or (stable_pair and len(stable_pair) >= 2):
                    reason_codes.append(RC_AMBIGUOUS_MULTI_CANDIDATE)
                    final_status = "AMBIGUOUS"
                    final_number = None
                elif len(strong_num_cands) == 1:
                    final_status = "PROVISIONAL"
                    final_number = strong_num_cands[0]
                else:
                    final_status = "UNKNOWN"
                    final_number = None
            else:
                if len(strong_num_cands) == 1:
                    vis_single = strong_num_cands[0]
                    if ntes_active_numbers and vis_single not in ntes_active_numbers:
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
            has_route_or_name = any(
                "TIMETABLE_NAME_MATCH" in ev
                or "CORROBORATION_VISUAL_NAME" in ev
                or "ROUTE_ENDPOINT" in ev
                or "CURATED_NAME_ALIAS_MATCH" in ev
                for ev in top_cand.evidence_list
            )
            has_live_or_sched_direction = any(
                "CORROBORATION_VISUAL_NAME_AND_NTES" in ev
                or "TIMETABLE_TIME_WINDOW" in ev
                or RC_DIRECTION_CORROBORATED in ev
                for ev in top_cand.evidence_list
            )
            is_score_tied = (runner_up is not None) and (abs(top_cand.score - runner_up.score) < 0.5)
            dir_verified = is_direction_verified(effective_dir_ev)

            if has_route_or_name:
                if is_score_tied or not has_live_or_sched_direction:
                    reason_codes.append(RC_NAME_ONLY_ACTIVE_DIRECTION_UNVERIFIED)
                    if len(service_family_candidates) > 1 or is_score_tied:
                        reason_codes.append(RC_AMBIGUOUS_MULTI_CANDIDATE)
                        final_status = "AMBIGUOUS"
                    elif len(service_family_candidates) == 0:
                        reason_codes.append(RC_SERVICE_MAPPING_UNVERIFIED)
                        final_status = "UNKNOWN"
                    else:
                        reason_codes.append(RC_AMBIGUOUS_MULTI_CANDIDATE)
                        final_status = "AMBIGUOUS"
                    final_number = None
                    active_service_number = None
                    active_direction_verified = False
                else:
                    final_status = "PROVISIONAL"
                    final_number = top_cand.train_number
                    if dir_verified:
                        active_service_number = final_number
                        active_direction_verified = True
                        reason_codes.append(RC_ACTIVE_DIRECTION_CORROBORATION)
                    else:
                        active_service_number = None
                        active_direction_verified = False
                        reason_codes.append(RC_NAME_ONLY_ACTIVE_DIRECTION_UNVERIFIED)
            else:
                if not service_family_candidates:
                    reason_codes.append(RC_SERVICE_MAPPING_UNVERIFIED)
                final_status = "UNKNOWN"
                final_number = None
                active_service_number = None
                active_direction_verified = False

        else:
            if runner_up and abs(top_cand.score - runner_up.score) < 0.5:
                reason_codes.append(RC_AMBIGUOUS_MULTI_CANDIDATE)
                final_status = "AMBIGUOUS"
                final_number = None
            else:
                final_status = "UNKNOWN"
                final_number = None
            active_service_number = None
            active_direction_verified = False

    if final_number and v_cons.numeric_status != "NOT_NUMBERED":
        if is_direction_verified(effective_dir_ev):
            active_service_number = final_number
            active_direction_verified = True
            reason_codes.append(RC_ACTIVE_DIRECTION_CORROBORATION)
        else:
            active_service_number = None
            active_direction_verified = False

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
        recognized_train_name=lookup_name if lookup_name else None,
        service_family_candidates=service_family_candidates,
        active_service_number=active_service_number,
        active_direction_verified=active_direction_verified,
    )
