"""
Timetable Importer for Multi-Site Railway Schedules
Parses site-specific CSV/XLSX schedules into canonical TimetableRow records,
handling decimal times ('5.10' -> '05:10'), leading zero preservation ('02563'),
operating day parsing ('Daily', 'Except Tuesday', etc.), and rake links.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shadow_identity.models import SiteScheduleSnapshot, TimetableRow


def parse_schedule_time(raw_val: Any) -> Optional[str]:
    """
    Parse schedule time string/float.
    Crucial Rule: '5.10' means 05:10, NOT 05:01!
    '6.05' means 06:05.
    '22.00' or '22.0' means 22:00.
    """
    if raw_val is None:
        return None
    s = str(raw_val).strip()
    if not s or s.lower() in ("nan", "none", "-"):
        return None

    # Handle standard HH:MM
    m_std = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if m_std:
        hh = int(m_std.group(1))
        mm = int(m_std.group(2))
        return f"{hh:02d}:{mm:02d}"

    # Handle Indian Railways decimal timetable format e.g. 5.10, 6.05, 23.55
    m_dec = re.match(r"^(\d{1,2})\.(\d{1,2})$", s)
    if m_dec:
        hh = int(m_dec.group(1))
        mm_str = m_dec.group(2)
        # If single digit e.g. 22.0 -> 22:00, or 7.3 -> 07:30
        if len(mm_str) == 1:
            mm = int(mm_str) * 10
        else:
            mm = int(mm_str)
        if 0 <= hh < 24 and 0 <= mm < 60:
            return f"{hh:02d}:{mm:02d}"

    return None


def parse_operating_days(remarks_text: str) -> List[int]:
    """
    Parse operating days from Remarks or Schedule text.
    Returns list of weekday integers (0=Monday, 6=Sunday).
    """
    if not remarks_text:
        return [0, 1, 2, 3, 4, 5, 6]

    txt = remarks_text.strip().lower()

    if "daily" in txt:
        return [0, 1, 2, 3, 4, 5, 6]

    days_map = {
        "mon": 0, "monday": 0,
        "tue": 1, "tues": 1, "tuesday": 1,
        "wed": 2, "wednesday": 2,
        "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
        "fri": 4, "friday": 4,
        "sat": 5, "saturday": 5,
        "sun": 6, "sunday": 6
    }

    # Except cases e.g. "Except Tuesday", "Except Sat,Sun."
    if "except" in txt:
        excluded = set()
        for k, v in days_map.items():
            if re.search(rf"\b{k}\b", txt):
                excluded.add(v)
        return [d for d in range(7) if d not in excluded]

    # Only cases e.g. "Only Sunday", "Only Tue, Fri"
    if "only" in txt or any(k in txt for k in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")):
        included = set()
        for k, v in days_map.items():
            if re.search(rf"\b{k}\b", txt):
                included.add(v)
        if included:
            return sorted(list(included))

    return [0, 1, 2, 3, 4, 5, 6]


def parse_rake_links(remarks_text: str) -> Optional[str]:
    """Extract rake linking info (e.g. 'Rake will go as 64465' -> '64465')."""
    if not remarks_text:
        return None
    m = re.search(r"rake\s+(?:will\s+go\s+as|of)\s+(\d{4,5})", remarks_text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


class TimetableImporter:
    @staticmethod
    def load_schedule_csv(
        file_path: Path, site_id: str, station_code: str, schedule_version: str
    ) -> List[TimetableRow]:
        """Load and parse CSV timetable into canonical TimetableRow records."""
        if not file_path.is_file():
            raise FileNotFoundError(f"Timetable file not found: {file_path}")

        rows: List[TimetableRow] = []
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, r in enumerate(reader, start=1):
                # Normalize keys
                clean_r = {k.strip(): (v.strip() if v else "") for k, v in r.items() if k}
                
                raw_num = clean_r.get("Train No.", clean_r.get("train_number", ""))
                if not raw_num:
                    continue
                # Preserve leading zero: format as 5 digits
                raw_num_clean = raw_num.split()[0].replace(",", "")
                if raw_num_clean.isdigit():
                    train_no = f"{int(raw_num_clean):05d}"
                else:
                    train_no = raw_num_clean

                train_name = clean_r.get("Train Name", clean_r.get("train_name", ""))
                platform = clean_r.get("PF", clean_r.get("platform", ""))
                arr_raw = clean_r.get("Arrl.", clean_r.get("arrival", ""))
                dep_raw = clean_r.get("Dep.", clean_r.get("departure", ""))
                remarks = clean_r.get("Remarks", clean_r.get("remarks", ""))

                arr_clean = parse_schedule_time(arr_raw)
                dep_clean = parse_schedule_time(dep_raw)
                operating_days = parse_operating_days(remarks)
                rake_link = parse_rake_links(remarks)

                stype = "passenger"
                if "emu" in train_name.lower() or "memu" in train_name.lower() or train_no.startswith("6"):
                    stype = "emu_memu"
                elif "spl" in train_name.lower() or "clone" in train_name.lower() or train_no.startswith("0"):
                    stype = "special"

                row_obj = TimetableRow(
                    site_id=site_id,
                    schedule_version=schedule_version,
                    station_code=station_code,
                    train_number=train_no,
                    train_name=train_name,
                    aliases=[],
                    platform=platform,
                    arrival_local=arr_clean,
                    departure_local=dep_clean,
                    operating_days=operating_days,
                    service_type=stype,
                    rake_next_service=rake_link,
                    source_row=idx,
                    validation_status="verified",
                )
                rows.append(row_obj)

        return rows

    @classmethod
    def import_schedule(
        cls,
        site_id: str,
        station_code: str,
        csv_path: str,
        version: str,
        effective_from: Optional[str] = None,
        effective_until: Optional[str] = None,
    ) -> SiteScheduleSnapshot:
        p = Path(csv_path)
        rows = cls.load_schedule_csv(p, site_id, station_code, version)
        
        # Compute sha256
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        checksum = h.hexdigest()

        return SiteScheduleSnapshot(
            site_id=site_id,
            station_code=station_code,
            version=version,
            checksum_sha256=checksum,
            rows=rows,
            effective_from=effective_from,
            effective_until=effective_until,
        )
