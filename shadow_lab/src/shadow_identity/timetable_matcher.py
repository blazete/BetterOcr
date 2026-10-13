"""
Timetable Matcher for Shadow Identity Lab
Matches visual candidates against site-specific timetable rows within
asymmetric service windows, respecting operating days and platform ambiguity.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

from shadow_identity.models import TimetableRow
from shadow_identity.reason_codes import (
    RC_DIRECTION_CONFLICT,
    RC_DIRECTION_CORROBORATED,
    RC_SCHEDULE_OUTSIDE_WINDOW,
)
from shadow_identity.text_normalization import match_name_with_aliases


def time_to_minutes(t_str: str) -> int:
    """Convert HH:MM to minutes from midnight."""
    h, m = t_str.split(":")
    return int(h) * 60 + int(m)


class TimetableMatcher:
    def __init__(
        self,
        schedule_rows: Optional[List[TimetableRow]] = None,
        site_id: str = "",
        early_window_min: int = 15,
        late_window_min: int = 90,
        rows: Optional[List[TimetableRow]] = None,
        max_deviation_min: int = 180,
    ):
        raw_rows = schedule_rows if schedule_rows is not None else (rows or [])
        self.site_id = site_id
        # Strict isolation: only retain rows for this specific site_id
        self.rows = [r for r in raw_rows if r.site_id == site_id]
        self.early_window_min = early_window_min
        self.late_window_min = late_window_min
        self.max_deviation_min = max_deviation_min

    def find_candidates(
        self,
        event_dt: datetime,
        possible_platforms: Optional[List[str]] = None,
        direction: str = "unknown",
        target_numbers: Optional[List[str]] = None,
        target_name: Optional[str] = None,
        site_aliases: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find candidate timetable rows for an event.
        Returns ranked list of candidate dictionaries with evidence tags.
        """
        event_min = event_dt.hour * 60 + event_dt.minute
        weekday = event_dt.weekday()

        candidates: List[Dict[str, Any]] = []

        for row in self.rows:
            # 1. Operating day check
            if weekday not in row.operating_days:
                # Strict overnight check: only allow previous operating day if event is
                # early morning (<= 06:00 / 360 min) AND train arrival is overnight
                prev_day = (weekday - 1) % 7
                is_overnight = False
                if prev_day in row.operating_days and event_min <= 360:
                    for time_str in (row.arrival_local, row.departure_local):
                        if time_str:
                            t_m = time_to_minutes(time_str)
                            if t_m <= 360 or t_m >= 1320:
                                is_overnight = True
                                break
                if not is_overnight:
                    continue

            # 2. Time window check
            sched_times = []
            if row.arrival_local:
                sched_times.append(("arr", time_to_minutes(row.arrival_local)))
            if row.departure_local:
                sched_times.append(("dep", time_to_minutes(row.departure_local)))

            if not sched_times:
                continue

            time_matched = False
            best_delta = 999
            event_type = "halt"

            for kind, s_min in sched_times:
                diff = event_min - s_min
                if diff < -720: diff += 1440
                if diff > 720: diff -= 1440

                if -self.early_window_min <= diff <= self.late_window_min:
                    time_matched = True
                    if abs(diff) < abs(best_delta):
                        best_delta = diff
                        event_type = kind
                elif abs(diff) < abs(best_delta):
                    best_delta = diff
                    event_type = kind

            # Hard cutoff: trains farther than max_deviation_min away cannot be candidates
            if abs(best_delta) > self.max_deviation_min:
                continue

            # Check number match
            number_match = bool(target_numbers and row.train_number in target_numbers)

            # Check train name match with aliases
            name_score = 0.0
            if target_name:
                name_score = match_name_with_aliases(
                    target_name,
                    row.train_name,
                    target_aliases=row.aliases,
                    site_aliases=site_aliases,
                )

            name_match = (name_score >= 0.6)

            if not (time_matched or number_match or name_match):
                continue

            # 3. Platform check
            platform_plausible = True
            if possible_platforms and row.platform:
                platform_plausible = (str(row.platform) in [str(p) for p in possible_platforms])

            # 4. Direction check
            dir_status = "neutral"
            if direction and direction.upper() in ("UP", "DN"):
                req_dir = direction.upper()
                inferred_dir = None
                # Check origin/dest in train name
                upper_name = row.train_name.upper()
                if "NDLS-" in upper_name:
                    inferred_dir = "DN"
                elif "-NDLS" in upper_name:
                    inferred_dir = "UP"
                elif row.departure_local and not row.arrival_local:
                    inferred_dir = "DN"
                elif row.arrival_local and not row.departure_local:
                    inferred_dir = "UP"

                if inferred_dir:
                    if inferred_dir == req_dir:
                        dir_status = "corroborated"
                    else:
                        dir_status = "conflict"

            # 5. Evidence collection
            evidence: List[str] = []
            score = 0.0

            if number_match:
                evidence.append("TIMETABLE_EXACT_NUMBER_MATCH")
                score += 5.0
            if name_match:
                evidence.append(f"TIMETABLE_NAME_MATCH_{int(name_score*100)}PCT")
                score += 3.0 * name_score
            if time_matched:
                evidence.append(f"TIMETABLE_TIME_WINDOW_DELTA_{best_delta}MIN")
                score += max(0.0, 2.0 - abs(best_delta) / 45.0)
            else:
                evidence.append(RC_SCHEDULE_OUTSIDE_WINDOW)
                score -= 2.0

            if platform_plausible and row.platform:
                evidence.append(f"TIMETABLE_PLATFORM_{row.platform}")
                score += 1.0

            if dir_status == "corroborated":
                evidence.append(RC_DIRECTION_CORROBORATED)
                score += 1.5
            elif dir_status == "conflict":
                evidence.append(RC_DIRECTION_CONFLICT)
                score -= 3.0

            candidates.append({
                "train_number": row.train_number,
                "train_name": row.train_name,
                "platform": row.platform,
                "score": round(score, 3),
                "time_delta_min": best_delta,
                "event_type": event_type,
                "evidence": evidence,
                "row_obj": row,
                "direction_status": dir_status,
            })

        # Rank candidates by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates
