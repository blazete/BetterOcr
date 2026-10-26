"""
Compatibility Adapter between ShadowOCR Phase 2 Visual Identity
and existing BetterOcr VisualOcrEvidence / CoachConsensus models.
Zero modifications to the existing candidate resolver.
"""

from __future__ import annotations

from typing import List

from shadow_identity.models import (
    NameCandidate,
    NumberCandidate,
    PassageVisualConsensus,
    VisualObservation,
    VisualOcrEvidence,
)
from shadow_identity.shadow_ocr.consensus import PassageVisualIdentityResult
from shadow_identity.shadow_ocr.contracts import RawOcrObservation


class ShadowOcrToBetterOcrAdapter:
    """
    Converts ShadowOCR Phase 2 PassageVisualIdentityResult into canonical VisualOcrEvidence.
    """

    @staticmethod
    def to_visual_ocr_evidence(
        identity_result: PassageVisualIdentityResult,
        raw_observations: List[RawOcrObservation],
    ) -> VisualOcrEvidence:
        # 1. Map raw observations to VisualObservation
        obs_list: List[VisualObservation] = []
        for o in raw_observations:
            obs = VisualObservation(
                coach_segment_id=o.coach_id or f"frame_{o.frame_id}",
                frame_id=o.frame_id,
                timestamp_offset_ms=o.timestamp_ms,
                raw_text=o.raw_text,
                normalized_text=o.normalized_text,
                entity_type=o.entity_type,
                bbox=o.bbox_full_frame,
                ocr_confidence=o.ocr_confidence,
            )
            obs_list.append(obs)

        # 2. Map Service Number candidates
        service_nums: List[NumberCandidate] = []
        for sn in identity_result.service_number_candidates:
            service_nums.append(
                NumberCandidate(
                    number=sn.number,
                    distinct_coaches=sn.distinct_coaches,
                    total_observations=sn.total_observations,
                    mean_confidence=sn.mean_confidence,
                )
            )

        # 3. Map Train Name candidates
        train_names: List[NameCandidate] = []
        for tn in identity_result.train_name_candidates:
            train_names.append(
                NameCandidate(
                    name=tn.name,
                    distinct_coaches=tn.distinct_coaches,
                    total_observations=tn.total_observations,
                    mean_confidence=tn.mean_confidence,
                )
            )

        # 4. Map numeric status
        status_map = {
            "PAIR_CONFIRMED": "PAIR_CONFIRMED",
            "PAIR_PROVISIONAL": "PAIR_PROVISIONAL",
            "SINGLE_CANDIDATE": "PAIR_PROVISIONAL",
            "NO_NUMBERS": "NOT_NUMBERED",
            "UNKNOWN": "PAIR_UNCONFIRMED",
        }
        mapped_status = status_map.get(identity_result.visual_pair_status, "PAIR_UNCONFIRMED")

        top_name = train_names[0].name if train_names else None

        # 5. Build PassageVisualConsensus
        consensus = PassageVisualConsensus(
            service_number_candidates=service_nums,
            stable_pair=identity_result.stable_pair,
            train_name_candidates=train_names,
            endpoint_candidates=identity_result.endpoint_candidates,
            number_board_type=identity_result.board_category,
            numeric_status=mapped_status,
            recognized_train_name=top_name,
        )

        return VisualOcrEvidence(
            passage_id=identity_result.passage_id,
            ocr_model_version="PP-OCRv6",
            preprocessing_profile="P1_COLOR_2X_BICUBIC",
            per_observation=obs_list,
            passage_consensus=consensus,
            recognized_train_name=top_name,
        )
