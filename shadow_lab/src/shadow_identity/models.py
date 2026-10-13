"""
Canonical Data Models for Shadow Identity Lab
Defines strictly typed dataclasses for Passage Input, Visual OCR Evidence,
Timetable Rows, NTES Snapshots, Shadow Decisions, and Comparison Records.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PassageInput:
    schema_version: str = "1.0"
    site_id: str = ""
    installation_id: str = ""
    passage_id: str = ""
    camera_id: str = "side_a"
    passage_start: str = ""  # ISO 8601 string
    passage_end: str = ""    # ISO 8601 string
    possible_platforms: List[str] = field(default_factory=list)
    observed_platform: Optional[str] = None
    direction: str = "unknown"
    source_files: List[Dict[str, str]] = field(default_factory=list)
    production_inspection_id: Optional[str] = None
    production_readonly_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisualObservation:
    coach_segment_id: str
    frame_id: int
    raw_text: str
    normalized_text: str
    entity_type: str  # "SERVICE_NUMBER" | "TRAIN_NAME" | "ENDPOINT" | "NEGATIVE_STENCIL"
    timestamp_offset_ms: int = 0
    bbox: List[int] = field(default_factory=list)
    ocr_confidence: float = 0.0
    image_ref: Optional[str] = None


@dataclass
class NumberCandidate:
    number: str
    distinct_coaches: int
    total_observations: int = 1
    mean_confidence: float = 0.90


@dataclass
class NameCandidate:
    name: str
    distinct_coaches: int
    total_observations: int = 1
    mean_confidence: float = 0.90


@dataclass
class PassageVisualConsensus:
    service_number_candidates: List[NumberCandidate] = field(default_factory=list)
    stable_pair: List[str] = field(default_factory=list)
    train_name_candidates: List[NameCandidate] = field(default_factory=list)
    endpoint_candidates: List[str] = field(default_factory=list)
    number_board_type: str = "UNKNOWN"  # "TWO_NUMBER", "FOUR_NUMBER", "SIX_NUMBER", "NOT_NUMBERED", "UNKNOWN"
    numeric_status: str = "PAIR_UNCONFIRMED"  # "PAIR_CONFIRMED", "PAIR_PROVISIONAL", "PAIR_UNCONFIRMED", "NOT_NUMBERED"


@dataclass
class VisualOcrEvidence:
    passage_id: str
    ocr_model_version: str = "paddleocr_ppocrv6"
    preprocessing_profile: str = "2x_bicubic"
    per_observation: List[VisualObservation] = field(default_factory=list)
    passage_consensus: PassageVisualConsensus = field(default_factory=PassageVisualConsensus)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimetableRow:
    site_id: str
    schedule_version: str
    station_code: str
    train_number: str  # Exactly 5 digits preserved (e.g. "02563")
    train_name: str
    aliases: List[str] = field(default_factory=list)
    platform: str = ""
    arrival_local: Optional[str] = None   # "HH:MM"
    departure_local: Optional[str] = None # "HH:MM"
    operating_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6]) # 0=Mon, 6=Sun
    service_type: str = "passenger" # "passenger", "emu_memu", "special"
    rake_next_service: Optional[str] = None
    source_row: int = 0
    validation_status: str = "verified"

    def is_operational_on(self, dt: datetime) -> bool:
        return dt.weekday() in self.operating_days


@dataclass
class SiteScheduleSnapshot:
    site_id: str
    station_code: str
    version: str
    checksum_sha256: str
    rows: List[TimetableRow] = field(default_factory=list)
    effective_from: Optional[str] = None
    effective_until: Optional[str] = None


@dataclass
class NtesTrainEntry:
    train_number: str  # 5-digit normalized string
    train_name: str
    platform: str
    time_min: Optional[int] = None  # Minutes from midnight IST
    time_raw: str = ""
    is_originating: bool = False
    is_terminating: bool = False
    dep_min: Optional[int] = None

    def is_emu(self) -> bool:
        return self.train_number.startswith("6")


@dataclass
class NtesSnapshot:
    site_id: str
    station_code: str
    requested_at: str   # ISO string
    fetched_at: str     # ISO string
    window_hr: int = 2
    available: bool = True
    from_cache: bool = False
    stale_fallback: bool = False
    snapshot_age_seconds_at_decision: float = 0.0
    source: str = "archive"  # "archive", "replay", "live"
    trains: List[NtesTrainEntry] = field(default_factory=list)
    raw_hash: str = ""

    def get_trains_for_platform(self, pf: str) -> List[NtesTrainEntry]:
        return [t for t in self.trains if t.platform == str(pf)]


@dataclass
class CandidateService:
    train_number: str
    train_name: str
    score: float = 0.0
    evidence_list: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)


@dataclass
class ShadowDecision:
    site_id: str
    passage_id: str
    algorithm_version: str = "shadow_resolver_v0.1"
    stage: str = "T0"  # "T0" (Visual+Schedule), "T1" (Pre-NTES), "T2" (Post-NTES +180s)
    evaluated_at: str = ""
    candidate_services: List[CandidateService] = field(default_factory=list)
    shadow_final_number: Optional[str] = None
    shadow_status: str = "UNKNOWN" # "CONFIRMED", "PROVISIONAL", "AMBIGUOUS", "CONFLICT", "UNKNOWN", "NOT_SERVICE_EVENT", "DATA_UNAVAILABLE"
    reason_codes: List[str] = field(default_factory=list)
    post_fetch_kind: Optional[str] = None # "NEW_FETCH", "CACHE_REUSE", "STALE_FALLBACK", "UNAVAILABLE"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonRecord:
    site_id: str
    installation_id: str
    passage_id: str
    passage_end_ist: str
    possible_platforms: str
    production_train_number: Optional[str]
    production_confidence: Optional[str]
    production_source: Optional[str]
    shadow_visual_numbers: str
    shadow_train_name: str
    shadow_endpoints: str
    schedule_version: str
    schedule_candidates: str
    ntes_pre_fetched_at: str
    ntes_pre_age_s: float
    ntes_pre_candidates: str
    ntes_post_fetched_at: str
    ntes_post_age_s: float
    ntes_post_candidates: str
    post_fetch_kind: str
    shadow_t0_number: Optional[str]
    shadow_t0_status: str
    shadow_t1_number: Optional[str]
    shadow_t1_status: str
    shadow_t2_number: Optional[str]
    shadow_t2_status: str
    shadow_reason_codes: str
    ground_truth_number: Optional[str]
    ground_truth_reviewer: Optional[str]
    ground_truth_quality: Optional[str]
    comparison_category: str
    review_required: bool
    resolver_version: str
    ocr_model_version: str
    site_config_hash: str
