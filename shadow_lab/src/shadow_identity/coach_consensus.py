"""
Coach-Level and Passage-Level Consensus Engine
Collapses multi-frame observations into distinct physical coach votes
and applies the frozen distinct-coach threshold rule.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from shadow_identity.models import (
    NameCandidate,
    NumberCandidate,
    PassageVisualConsensus,
    VisualObservation,
)
from shadow_identity.text_normalization import (
    extract_5digit_candidates,
    is_endpoint_text,
    is_negative_class,
    normalize_train_name,
)


class CoachConsensusAggregator:
    def __init__(self, confirmed_threshold: int = 3, min_distinct_coaches: Optional[int] = None):
        if min_distinct_coaches is not None:
            self.confirmed_threshold = min_distinct_coaches
        else:
            self.confirmed_threshold = confirmed_threshold
        self.observations: List[VisualObservation] = []

    def add_observation(self, obs: VisualObservation) -> None:
        self.observations.append(obs)

    def compute_consensus(self) -> PassageVisualConsensus:
        return self.aggregate_observations(self.observations)

    def aggregate_observations(
        self, observations: List[VisualObservation]
    ) -> PassageVisualConsensus:
        """
        Aggregate per-frame observations into a passage-level visual consensus.
        Enforces: Repeated frames of one physical coach count as exactly ONE coach observation!
        """
        if not observations:
            return PassageVisualConsensus(
                number_board_type="NOT_NUMBERED",
                numeric_status="NOT_NUMBERED"
            )

        # 1. Group observations by coach_segment_id
        coaches_numbers: Dict[str, Set[str]] = defaultdict(set)
        number_confidences: Dict[str, List[float]] = defaultdict(list)
        number_total_obs: Counter = Counter()

        coaches_names: Dict[str, Set[str]] = defaultdict(set)
        name_confidences: Dict[str, List[float]] = defaultdict(list)
        name_total_obs: Counter = Counter()

        endpoint_candidates: Set[str] = set()

        for obs in observations:
            cid = obs.coach_segment_id
            etype = obs.entity_type
            raw = obs.raw_text

            if etype == "SERVICE_NUMBER":
                cands = extract_5digit_candidates(raw)
                for c in cands:
                    coaches_numbers[c].add(cid)
                    number_confidences[c].append(obs.ocr_confidence)
                    number_total_obs[c] += 1

            elif etype == "TRAIN_NAME":
                if is_endpoint_text(raw):
                    norm_ep = raw.strip().upper()
                    if norm_ep and len(norm_ep) > 2:
                        endpoint_candidates.add(norm_ep)
                    continue
                norm_n = normalize_train_name(raw)
                if norm_n and len(norm_n) > 3 and not is_negative_class(norm_n):
                    coaches_names[norm_n].add(cid)
                    name_confidences[norm_n].append(obs.ocr_confidence)
                    name_total_obs[norm_n] += 1

            elif etype == "ENDPOINT":
                norm_ep = raw.strip().upper()
                if norm_ep and len(norm_ep) > 2:
                    endpoint_candidates.add(norm_ep)

        # 2. Build NumberCandidate list sorted by distinct coaches then total observations
        number_cand_objs: List[NumberCandidate] = []
        for num, coach_set in coaches_numbers.items():
            dist_coaches = len(coach_set)
            tot_obs = number_total_obs[num]
            m_conf = (
                sum(number_confidences[num]) / len(number_confidences[num])
                if number_confidences[num]
                else 0.0
            )
            number_cand_objs.append(
                NumberCandidate(
                    number=num,
                    distinct_coaches=dist_coaches,
                    total_observations=tot_obs,
                    mean_confidence=round(m_conf, 4),
                )
            )
        number_cand_objs.sort(
            key=lambda x: (x.distinct_coaches, x.total_observations, x.mean_confidence),
            reverse=True,
        )

        # 3. Build NameCandidate list
        name_cand_objs: List[NameCandidate] = []
        for name_str, coach_set in coaches_names.items():
            dist_coaches = len(coach_set)
            tot_obs = name_total_obs[name_str]
            m_conf = (
                sum(name_confidences[name_str]) / len(name_confidences[name_str])
                if name_confidences[name_str]
                else 0.0
            )
            name_cand_objs.append(
                NameCandidate(
                    name=name_str,
                    distinct_coaches=dist_coaches,
                    total_observations=tot_obs,
                    mean_confidence=round(m_conf, 4),
                )
            )
        name_cand_objs.sort(
            key=lambda x: (x.distinct_coaches, x.total_observations),
            reverse=True,
        )

        # 4. Determine numeric status and stable pair according to strict frozen rule
        top_numbers = [c.number for c in number_cand_objs]
        stable_pair: List[str] = []
        board_type = "UNKNOWN"
        numeric_status = "PAIR_UNCONFIRMED"

        cands_ge3 = [c for c in number_cand_objs if c.distinct_coaches >= self.confirmed_threshold]
        cands_ge1 = [c for c in number_cand_objs if c.distinct_coaches >= 1]

        if len(number_cand_objs) == 0:
            board_type = "NOT_NUMBERED" if name_cand_objs else "UNKNOWN"
            numeric_status = "NOT_NUMBERED"

        elif len(cands_ge3) >= 2:
            # Both top numbers independently verified on >= 3 distinct physical coaches
            stable_pair = [cands_ge3[0].number, cands_ge3[1].number]
            board_type = "TWO_NUMBER"
            numeric_status = "PAIR_CONFIRMED"

        elif len(cands_ge3) == 1:
            # Primary candidate confirmed on >= 3 coaches
            primary = cands_ge3[0].number
            secondary_cands = [c for c in number_cand_objs if c.number != primary]
            if secondary_cands and secondary_cands[0].distinct_coaches >= 1:
                stable_pair = [primary, secondary_cands[0].number]
                board_type = "TWO_NUMBER"
                # Strict frozen rule: Asymmetric support is marked PROVISIONAL
                numeric_status = "PAIR_PROVISIONAL"
            else:
                stable_pair = [primary]
                board_type = "TWO_NUMBER"
                numeric_status = "PAIR_PROVISIONAL"

        elif len(cands_ge1) >= 2:
            stable_pair = [cands_ge1[0].number, cands_ge1[1].number]
            board_type = "TWO_NUMBER"
            numeric_status = "PAIR_UNCONFIRMED"

        else:
            board_type = "UNKNOWN"
            numeric_status = "PAIR_UNCONFIRMED"

        # Check for complex boards (>= 4 distinct 5-digit numbers seen with high confidence)
        strong_candidates = [c for c in number_cand_objs if c.distinct_coaches >= 2]
        if len(strong_candidates) >= 4:
            board_type = "FOUR_NUMBER"
            numeric_status = "COMPLEX_BOARD"

        return PassageVisualConsensus(
            service_number_candidates=number_cand_objs,
            stable_pair=sorted(stable_pair),
            train_name_candidates=name_cand_objs,
            endpoint_candidates=sorted(list(endpoint_candidates)),
            number_board_type=board_type,
            numeric_status=numeric_status,
        )
