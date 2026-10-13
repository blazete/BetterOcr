"""
Timetable Matcher for Shadow Identity Lab
Matches visual candidates against site-specific timetable rows within
asymmetric service windows, respecting operating days and platform ambiguity.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

from shadow_identity.models import TimetableRow
from shadow_identity.text_normalization import match_name_similarity


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
    ):
        raw_rows = schedule_rows if schedule_rows is not None else (rows or [])
        self.site_id = site_id
        # Strict isolation: only retain rows for this specific site_id
        self.rows = [r for r in raw_rows if r.site_id == site_id]
        self.early_window_min = early_window_min
        self.late_window_min = late_window_min

    def find_candidates(
        self,
        event_dt: datetime,
        possible_platforms: Optional[List[str]] = None,
        direction: str = "unknown",
        target_numbers: Optional[List[str]] = None,
        target_name: Optional[str] = None,
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
                # Also check yesterday in case of overnight arrival/departure
                prev_day = (weekday - 1) % 7
                if prev_day not in row.operating_days:
                    continue

            # 2. Time window check
            # Evaluate against arrival, departure, or both
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
                # Handle midnight boundary
                diff = event_min - s_min
                if diff < -720: diff += 1440
                if diff > 720: diff -= 1440

                # Asymmetric check: train can be slightly early (-early_window) or late (+late_window)
                if -self.early_window_min <= diff <= self.late_window_min:
                    time_matched = True
                    if abs(diff) < abs(best_delta):
                        best_delta = diff
                        event_type = kind

            # If target_numbers provided and row matches number, relax time window slightly
            number_match = bool(target_numbers and row.train_number in target_numbers)
            
            # Check train name match
            name_score = 0.0
            if target_name:
                name_score = match_name_similarity(target_name, row.train_name)
                for alias in row.aliases:
                    a_score = match_name_similarity(target_name, alias)
                    if a_score > name_score:
                        name_score = a_score

            name_match = (name_score >= 0.6)

            if not (time_matched or number_match or name_match):
                continue

            # 3. Platform check
            platform_plausible = True
            if possible_platforms and row.platform:
                platform_plausible = (row.platform in possible_platforms)

            # 4. Evidence collection
            evidence: List[str] = []
            if number_match:
                evidence.append("TIMETABLE_EXACT_NUMBER_MATCH")
            if name_match:
                evidence.append(f"TIMETABLE_NAME_MATCH_{int(name_score*100)}PCT")
            if time_matched:
                evidence.append(f"TIMETABLE_TIME_WINDOW_DELTA_{best_delta}MIN")
            if platform_plausible and row.platform:
                evidence.append(f"TIMETABLE_PLATFORM_{row.platform}")

            score = 0.0
            if number_match: score += 5.0
            if name_match: score += 3.0 * name_score
            if time_matched: score += max(0.0, 2.0 - abs(best_delta) / 45.0)
            if platform_plausible: score += 1.0

            candidates.append({
                "train_number": row.train_number,
                "train_name": row.train_name,
                "platform": row.platform,
                "score": round(score, 3),
                "time_delta_min": best_delta,
                "event_type": event_type,
                "evidence": evidence,
                "row_obj": row,
            })

        # Rank candidates by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates
