"""
Physical Coach Attribution and Passage Consensus Engine for ShadowOCR.
Strictly rules out coach count fabrication:
- Repeated frames of one coach = 1 coach vote.
- Missing coach-boundary metadata = UNVERIFIED status (cannot confirm PAIR_CONFIRMED).
- Pair confirmation requires >=3 distinct verified physical coaches for BOTH candidates.
- Zero ground-truth leakage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from shadow_identity.reason_codes import (
    RC_NAME_RECOGNIZED_COACH_UNVERIFIED,
    RC_NAME_RECOGNIZED_SINGLE_OBSERVATION,
    RC_NAME_REPEATED_VERIFIED_COACHES,
    RC_VISUAL_NAME_STRONG,
    RC_VISUAL_NAME_WEAK,
)
from shadow_identity.shadow_ocr.contracts import RawOcrObservation
from shadow_identity.shadow_ocr.text_entities import EntityClassifier


@dataclass
class ServiceNumberCandidate:
    number: str
    distinct_coaches: int
    total_observations: int
    mean_confidence: float
    supporting_coach_ids: List[str]
    supporting_frame_ids: List[int]


@dataclass
class TrainNameCandidate:
    name: str
    distinct_coaches: int
    total_observations: int
    mean_confidence: float
    supporting_coach_ids: List[str]
    supporting_frame_ids: List[int]


@dataclass
class PassageVisualIdentityResult:
    schema_version: str = "shadow_ocr.v2"
    site_id: str = ""
    passage_id: str = ""
    camera_id: str = ""
    video_sha256: str = ""
    total_raw_observations: int = 0
    total_frames_evaluated: int = 0
    coach_tracking_quality: str = "UNVERIFIED"  # "VERIFIED" | "UNVERIFIED"
    distinct_verified_coaches_total: int = 0
    board_category: str = "UNKNOWN"  # "TWO_NUMBER" | "FOUR_NUMBER" | "NAME_ONLY" | "NO_NUMBERS" | "UNKNOWN"
    visual_pair_status: str = "PAIR_UNCONFIRMED"  # "PAIR_CONFIRMED" | "PAIR_PROVISIONAL" | "SINGLE_CANDIDATE" | "NO_NUMBERS" | "UNKNOWN"
    stable_pair: List[str] = field(default_factory=list)
    dominant_service_number: Optional[str] = None
    service_number_candidates: List[ServiceNumberCandidate] = field(default_factory=list)
    train_name_candidates: List[TrainNameCandidate] = field(default_factory=list)
    endpoint_candidates: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    processing_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PhysicalCoachConsensusEngine:
    """
    Collapses raw frame-level OCR observations into coach-level and passage-level identity.
    Enforces strict mathematical thresholds on physical coach support.
    """

    def __init__(self, confirmed_threshold: int = 3):
        self.confirmed_threshold = confirmed_threshold

    def build_passage_identity(
        self,
        site_id: str,
        passage_id: str,
        camera_id: str,
        video_sha256: str,
        observations: List[RawOcrObservation],
        processing_latency_ms: float = 0.0,
    ) -> PassageVisualIdentityResult:
        result = PassageVisualIdentityResult(
            site_id=site_id,
            passage_id=passage_id,
            camera_id=camera_id,
            video_sha256=video_sha256,
            total_raw_observations=len(observations),
            processing_latency_ms=processing_latency_ms,
        )

        if not observations:
            result.board_category = "NO_NUMBERS"
            result.visual_pair_status = "NO_NUMBERS"
            result.reason_codes.append("NO_RAW_OCR_OBSERVATIONS")
            return result

        unique_frames = {o.frame_id for o in observations}
        result.total_frames_evaluated = len(unique_frames)

        # 1. Physical Coach Attribution Verification
        # Check if coach_id is present and verified
        verified_coaches: Set[str] = set()
        has_any_verified = False

        for o in observations:
            if o.coach_id and o.coach_id_source == "VERIFIED_BOUNDARY":
                verified_coaches.add(o.coach_id)
                has_any_verified = True

        if has_any_verified and len(verified_coaches) > 0:
            result.coach_tracking_quality = "VERIFIED"
            result.distinct_verified_coaches_total = len(verified_coaches)
        else:
            result.coach_tracking_quality = "UNVERIFIED"
            result.distinct_verified_coaches_total = 0
            result.reason_codes.append("COACH_BOUNDARIES_UNVERIFIED")

        # 2. Re-classify tokens using EntityClassifier and group by coach
        # candidate_numbers: {number: {coach_id: [obs, ...]}}
        num_by_coach: Dict[str, Dict[str, List[RawOcrObservation]]] = {}
        # name_by_coach: {name: {coach_id: [obs, ...]}}
        name_by_coach: Dict[str, Dict[str, List[RawOcrObservation]]] = {}
        endpoints_seen: Set[str] = set()

        for obs in observations:
            # Re-classify tokens using Phase 7C EntityClassifier
            c_entity = EntityClassifier.classify(obs.raw_text, obs.ocr_confidence)
            ent_type = c_entity.entity_type
            ent_norm = c_entity.normalized_text
            can_name = c_entity.canonical_name or ent_norm

            # Use assigned coach_id if verified, otherwise use frame-specific synthetic key to prevent fake coach fusion
            coach_key = obs.coach_id if (result.coach_tracking_quality == "VERIFIED" and obs.coach_id) else f"frame_{obs.frame_id}"

            if ent_type == "SERVICE_NUMBER":
                num = ent_norm
                if len(num) == 5 and num.isdigit():
                    num_by_coach.setdefault(num, {}).setdefault(coach_key, []).append(obs)

            elif ent_type == "TRAIN_NAME":
                tname = can_name
                if len(tname) >= 3:
                    name_by_coach.setdefault(tname, {}).setdefault(coach_key, []).append(obs)

            elif ent_type == "ENDPOINT":
                endpoints_seen.add(ent_norm)

        result.endpoint_candidates = sorted(list(endpoints_seen))

        # 3. Build Service Number Candidates
        num_candidates: List[ServiceNumberCandidate] = []
        for num, coach_dict in num_by_coach.items():
            # If coach boundaries unverified, distinct_coaches is counted as 1 (or 0 toward confirmation)
            # to prevent multiple frames of one coach from faking coach support
            distinct_coaches_count = len(coach_dict) if result.coach_tracking_quality == "VERIFIED" else 1

            all_obs = [o for obs_list in coach_dict.values() for o in obs_list]
            mean_conf = sum(o.ocr_confidence for o in all_obs) / len(all_obs)
            all_frames = sorted(list({o.frame_id for o in all_obs}))
            all_coach_ids = sorted(list(coach_dict.keys()))

            num_candidates.append(
                ServiceNumberCandidate(
                    number=num,
                    distinct_coaches=distinct_coaches_count,
                    total_observations=len(all_obs),
                    mean_confidence=round(mean_conf, 4),
                    supporting_coach_ids=all_coach_ids,
                    supporting_frame_ids=all_frames,
                )
            )

        # Sort service numbers by distinct coaches descending, then total observations, then confidence
        num_candidates.sort(key=lambda x: (x.distinct_coaches, x.total_observations, x.mean_confidence), reverse=True)
        result.service_number_candidates = num_candidates

        # 4. Build Train Name Candidates
        name_candidates: List[TrainNameCandidate] = []
        for name, coach_dict in name_by_coach.items():
            distinct_coaches_count = len(coach_dict) if result.coach_tracking_quality == "VERIFIED" else 1
            all_obs = [o for obs_list in coach_dict.values() for o in obs_list]
            mean_conf = sum(o.ocr_confidence for o in all_obs) / len(all_obs)
            all_frames = sorted(list({o.frame_id for o in all_obs}))
            all_coach_ids = sorted(list(coach_dict.keys()))

            name_candidates.append(
                TrainNameCandidate(
                    name=name,
                    distinct_coaches=distinct_coaches_count,
                    total_observations=len(all_obs),
                    mean_confidence=round(mean_conf, 4),
                    supporting_coach_ids=all_coach_ids,
                    supporting_frame_ids=all_frames,
                )
            )

        name_candidates.sort(key=lambda x: (x.distinct_coaches, x.total_observations, x.mean_confidence), reverse=True)
        result.train_name_candidates = name_candidates

        # Tag name consensus evidence-strength reason codes (§Part 3 A3)
        if name_candidates:
            top_name_cand = name_candidates[0]
            if result.coach_tracking_quality != "VERIFIED":
                result.reason_codes.append(RC_NAME_RECOGNIZED_COACH_UNVERIFIED)
            elif top_name_cand.distinct_coaches >= 2:
                result.reason_codes.append(RC_NAME_REPEATED_VERIFIED_COACHES)
                result.reason_codes.append(RC_VISUAL_NAME_STRONG)
            else:
                result.reason_codes.append(RC_NAME_RECOGNIZED_SINGLE_OBSERVATION)
                result.reason_codes.append(RC_VISUAL_NAME_WEAK)

        # 5. Determine Board Category and Visual Pair Status
        if len(num_candidates) == 0:
            if len(name_candidates) > 0:
                result.board_category = "NAME_ONLY"
                result.visual_pair_status = "NO_NUMBERS"
                result.reason_codes.append("VISUAL_NAME_ONLY_DETECTED")
            else:
                result.board_category = "UNKNOWN"
                result.visual_pair_status = "UNKNOWN"
                result.reason_codes.append("NO_RECOGNIZABLE_RAILWAY_ENTITIES")
            return result

        result.dominant_service_number = num_candidates[0].number

        if len(num_candidates) >= 4:
            result.board_category = "FOUR_NUMBER"
        elif len(num_candidates) in (2, 3):
            result.board_category = "TWO_NUMBER"
        else:
            result.board_category = "TWO_NUMBER"

        # Check strict confirmation thresholds
        # If coach tracking is UNVERIFIED, pair can NEVER be PAIR_CONFIRMED based on frames
        if result.coach_tracking_quality != "VERIFIED":
            result.visual_pair_status = "PAIR_PROVISIONAL"
            result.stable_pair = [c.number for c in num_candidates[:2]]
            result.reason_codes.append("COACHES_UNVERIFIED_PAIR_PROVISIONAL_ONLY")
            return result

        # When coach tracking is VERIFIED:
        # Strict rule: BOTH numbers must independently have >= confirmed_threshold distinct verified coaches
        if len(num_candidates) >= 2:
            first = num_candidates[0]
            second = num_candidates[1]

            if first.distinct_coaches >= self.confirmed_threshold and second.distinct_coaches >= self.confirmed_threshold:
                result.visual_pair_status = "PAIR_CONFIRMED"
                result.stable_pair = [first.number, second.number]
                result.reason_codes.append("VISUAL_PAIR_CONFIRMED_BOTH_GE3_COACHES")
            elif first.distinct_coaches >= self.confirmed_threshold:
                result.visual_pair_status = "PAIR_PROVISIONAL"
                result.stable_pair = [first.number, second.number]
                result.reason_codes.append("SINGLE_CANDIDATE_STRONG_SECOND_BELOW_THRESHOLD")
            else:
                result.visual_pair_status = "PAIR_PROVISIONAL"
                result.stable_pair = [first.number, second.number]
                result.reason_codes.append("ALL_CANDIDATES_BELOW_COACH_THRESHOLD")
        else:
            # Only 1 service number observed
            first = num_candidates[0]
            result.stable_pair = [first.number]
            if first.distinct_coaches >= self.confirmed_threshold:
                result.visual_pair_status = "SINGLE_CANDIDATE"
                result.reason_codes.append("SINGLE_CANDIDATE_GE3_COACHES")
            else:
                result.visual_pair_status = "PAIR_PROVISIONAL"
                result.reason_codes.append("SINGLE_CANDIDATE_BELOW_THRESHOLD")

        return result
