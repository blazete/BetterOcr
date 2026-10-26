"""
Evidence Fusion and Deterministic Identity Decision Engine (Package C)
Implements:
C1 — 10 distinct evidence categories and typed provenance records.
C2 — Evidence extraction and candidate generation without active-service preselection.
C3 — Deterministic active service decision gating:
     VERIFIED_ACTIVE, FAMILY_CANDIDATE, UNKNOWN, CONFLICT.
C4 — Atomic output serialization following schema_version 'shadow_identity_v1'.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from shadow_identity.candidate_resolver import is_direction_verified, resolve_shadow
from shadow_identity.models import (
    CandidateService,
    DirectionEvidence,
    PassageInput,
    ShadowDecision,
    TimetableRow,
    VisualOcrEvidence,
)
from shadow_identity.reason_codes import (
    RC_ACTIVE_DIRECTION_CORROBORATION,
    RC_AMBIGUOUS_MULTI_CANDIDATE,
    RC_DIRECTION_CONFLICT,
    RC_NAME_ONLY_ACTIVE_DIRECTION_UNVERIFIED,
    RC_SERVICE_FAMILY_MAPPING,
    RC_SERVICE_FAMILY_MATCHED,
    RC_SERVICE_MAPPING_UNVERIFIED,
    RC_TIMETABLE_NTES_CONFLICT,
    RC_VISUAL_NTES_CONFLICT,
    RC_VISUAL_TIMETABLE_CONFLICT,
)

EVIDENCE_CATEGORIES = (
    "VISUAL_SERVICE_NUMBER",
    "VISUAL_TRAIN_NAME",
    "VISUAL_ENDPOINT",
    "VALIDATED_SERVICE_REGISTRY",
    "VALIDATED_SITE_TIMETABLE",
    "AUTHENTIC_LIVE_TRAIN_FEED",
    "PHYSICAL_PASSAGE_TIMESTAMP",
    "VERIFIED_PLATFORM",
    "VERIFIED_TRACK_DIRECTION",
    "VERIFIED_PHYSICAL_COACH",
)


@dataclass
class IdentityEvidenceRecord:
    evidence_type: str
    original_source: str
    observed_value: Any
    source_passage_id: str
    observation_time_utc: Optional[str] = None
    validation_result: bool = True
    verification_provenance: Optional[str] = None
    contradictions: List[str] = field(default_factory=list)
    can_establish_service_family: bool = False
    can_establish_active_identity: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class V1IdentityDecision:
    site_id: str
    passage_id: str
    identity_status: str  # VERIFIED_ACTIVE | FAMILY_CANDIDATE | UNKNOWN | CONFLICT
    active_service_number: Optional[str]
    active_direction_verified: bool
    service_family_candidates: List[str]
    recognized_train_names: List[str]
    visual_service_numbers: List[str]
    recognized_endpoints: List[str]
    reason_codes: List[str]
    evidence_provenance: List[Dict[str, Any]]
    contradictions: List[str]
    candidate_services: List[CandidateService]
    shadow_decision: ShadowDecision
    decision_created_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "shadow_identity_v1",
            "passage_id": self.passage_id,
            "site_id": self.site_id,
            "identity_status": self.identity_status,
            "active_service_number": self.active_service_number,
            "active_direction_verified": self.active_direction_verified,
            "service_family_candidates": self.service_family_candidates,
            "recognized_train_names": self.recognized_train_names,
            "visual_service_numbers": self.visual_service_numbers,
            "recognized_endpoints": self.recognized_endpoints,
            "reason_codes": self.reason_codes,
            "evidence_provenance": self.evidence_provenance,
            "contradictions": self.contradictions,
            "decision_created_utc": self.decision_created_utc,
        }


def collect_evidence_records(
    passage: PassageInput,
    visual_evidence: VisualOcrEvidence,
    site_schedule_rows: List[TimetableRow],
    direction_evidence: Optional[DirectionEvidence] = None,
) -> Tuple[List[IdentityEvidenceRecord], List[str]]:
    """
    C1 / C2: Extract normalized evidence records with provenance and detect explicit contradictions.
    """
    records: List[IdentityEvidenceRecord] = []
    contradictions: List[str] = []

    p_id = passage.passage_id
    v_cons = visual_evidence.passage_consensus

    # 1. Visual Service Numbers
    for c in v_cons.service_number_candidates:
        records.append(
            IdentityEvidenceRecord(
                evidence_type="VISUAL_SERVICE_NUMBER",
                original_source=f"consensus_{c.distinct_coaches}_coaches",
                observed_value=c.number,
                source_passage_id=p_id,
                validation_result=True,
                verification_provenance="visual_consensus",
                can_establish_service_family=True,
                can_establish_active_identity=False,
            )
        )

    # 2. Visual Train Name
    if v_cons.train_name_candidates:
        for nc in v_cons.train_name_candidates:
            records.append(
                IdentityEvidenceRecord(
                    evidence_type="VISUAL_TRAIN_NAME",
                    original_source=f"train_name_{nc.distinct_coaches}_coaches",
                    observed_value=nc.name,
                    source_passage_id=p_id,
                    validation_result=True,
                    verification_provenance="visual_consensus",
                    can_establish_service_family=True,
                    can_establish_active_identity=False,
                )
            )

    # 3. Visual Endpoints
    for ep in v_cons.endpoint_candidates:
        records.append(
            IdentityEvidenceRecord(
                evidence_type="VISUAL_ENDPOINT",
                original_source="endpoint_ocr",
                observed_value=ep,
                source_passage_id=p_id,
                validation_result=True,
                verification_provenance="visual_consensus",
                can_establish_service_family=True,
                can_establish_active_identity=False,
            )
        )

    # 4. Validated Site Timetable
    if site_schedule_rows:
        records.append(
            IdentityEvidenceRecord(
                evidence_type="VALIDATED_SITE_TIMETABLE",
                original_source=f"site_{passage.site_id}_timetable",
                observed_value=len(site_schedule_rows),
                source_passage_id=p_id,
                validation_result=True,
                verification_provenance="timetable_importer",
                can_establish_service_family=True,
                can_establish_active_identity=False,
            )
        )

    # 5. Verified Track Direction
    eff_dir = direction_evidence or passage.direction_evidence
    if eff_dir:
        is_dir_valid = is_direction_verified(eff_dir)
        has_dir_conflict = getattr(eff_dir, "has_conflict", False) or (eff_dir.authentic and not eff_dir.direction_matches_candidate)
        if has_dir_conflict:
            contradictions.append("DIRECTION_CONFLICT")
        records.append(
            IdentityEvidenceRecord(
                evidence_type="VERIFIED_TRACK_DIRECTION",
                original_source=eff_dir.source or "direction_sensor",
                observed_value=eff_dir.direction,
                source_passage_id=p_id,
                validation_result=is_dir_valid,
                verification_provenance=eff_dir.provenance or "hardware_direction",
                contradictions=["DIRECTION_CONFLICT"] if has_dir_conflict else [],
                can_establish_service_family=False,
                can_establish_active_identity=is_dir_valid,
            )
        )

    return records, contradictions


def decide_identity_v1(
    passage: PassageInput,
    visual_evidence: VisualOcrEvidence,
    site_schedule_rows: List[TimetableRow],
    ntes_before: Optional[Any] = None,
    ntes_after: Optional[Any] = None,
    stage: str = "T0",
    direction_evidence: Optional[DirectionEvidence] = None,
    config: Optional[Dict[str, Any]] = None,
) -> V1IdentityDecision:
    """
    C3: Deterministic Identity Decision Engine.
    Executes existing arbitration and translates into target presentation statuses:
    - VERIFIED_ACTIVE
    - FAMILY_CANDIDATE
    - UNKNOWN
    - CONFLICT
    """
    eff_dir = direction_evidence or passage.direction_evidence

    # Call canonical resolve_shadow
    shadow_dec: ShadowDecision = resolve_shadow(
        passage=passage,
        visual_evidence=visual_evidence,
        site_schedule_rows=site_schedule_rows,
        ntes_before=ntes_before,
        ntes_after=ntes_after,
        stage=stage,
        direction_evidence=eff_dir,
        config=config,
    )

    ev_records, detected_conflicts = collect_evidence_records(
        passage=passage,
        visual_evidence=visual_evidence,
        site_schedule_rows=site_schedule_rows,
        direction_evidence=eff_dir,
    )

    # Check reason codes for conflicts
    has_conflict_rc = any(
        rc in shadow_dec.reason_codes
        for rc in (
            RC_DIRECTION_CONFLICT,
            RC_VISUAL_NTES_CONFLICT,
            RC_VISUAL_TIMETABLE_CONFLICT,
            RC_TIMETABLE_NTES_CONFLICT,
        )
    )
    if has_conflict_rc:
        for rc in shadow_dec.reason_codes:
            if rc in (
                RC_DIRECTION_CONFLICT,
                RC_VISUAL_NTES_CONFLICT,
                RC_VISUAL_TIMETABLE_CONFLICT,
                RC_TIMETABLE_NTES_CONFLICT,
            ) and rc not in detected_conflicts:
                detected_conflicts.append(rc)

    is_conflict = (shadow_dec.shadow_status == "CONFLICT") or bool(detected_conflicts)

    # Extract visual features
    v_cons = visual_evidence.passage_consensus
    visual_nums = [c.number for c in v_cons.service_number_candidates]
    rec_names = [v_cons.recognized_train_name] if v_cons.recognized_train_name else [c.name for c in v_cons.train_name_candidates]
    rec_endpoints = list(v_cons.endpoint_candidates)

    # Exact set-membership and provenance gating (§Item 1):
    # Only independently validated members of an approved registered service family qualify.
    family_cands = list(shadow_dec.service_family_candidates)
    if not family_cands:
        # Check if visual numbers strictly match an approved registered service family set
        from shadow_identity.shadow_ocr.text_entities import TRAIN_NAME_REGISTRY
        candidate_num_set = set(v_cons.stable_pair) if v_cons.stable_pair else set(visual_nums)
        if candidate_num_set:
            matched_family = None
            site_aliases = config.get("site_aliases") if config else None
            alias_dict = site_aliases.get("aliases", {}) if site_aliases else {}
            for alias_data in alias_dict.values():
                if alias_data.get("verification_state", "VERIFIED") == "VERIFIED":
                    s_nos = {str(n) for n in alias_data.get("service_numbers", [])}
                    if s_nos and candidate_num_set == s_nos:
                        matched_family = sorted(list(s_nos))
                        break
            if not matched_family:
                for reg_entry in TRAIN_NAME_REGISTRY.values():
                    if getattr(reg_entry, "verification_state", "VERIFIED") == "VERIFIED":
                        r_nos = {str(n) for n in reg_entry.service_family_numbers}
                        if r_nos and candidate_num_set == r_nos:
                            matched_family = sorted(list(r_nos))
                            break
            if matched_family:
                family_cands = matched_family

    now_utc = datetime.now(timezone.utc).isoformat()

    # Gating per Part 5 — C3
    if is_conflict:
        identity_status = "CONFLICT"
        active_num = None
        dir_verified = False

    elif (
        is_direction_verified(eff_dir)
        and shadow_dec.active_direction_verified
        and shadow_dec.active_service_number is not None
    ):
        identity_status = "VERIFIED_ACTIVE"
        active_num = shadow_dec.active_service_number
        dir_verified = True

    elif family_cands:
        identity_status = "FAMILY_CANDIDATE"
        active_num = None
        dir_verified = False

    else:
        identity_status = "UNKNOWN"
        active_num = None
        dir_verified = False

    return V1IdentityDecision(
        site_id=passage.site_id,
        passage_id=passage.passage_id,
        identity_status=identity_status,
        active_service_number=active_num,
        active_direction_verified=dir_verified,
        service_family_candidates=family_cands,
        recognized_train_names=[n for n in rec_names if n],
        visual_service_numbers=visual_nums,
        recognized_endpoints=rec_endpoints,
        reason_codes=shadow_dec.reason_codes,
        evidence_provenance=[r.to_dict() for r in ev_records],
        contradictions=detected_conflicts,
        candidate_services=shadow_dec.candidate_services,
        shadow_decision=shadow_dec,
        decision_created_utc=now_utc,
    )


def format_v1_identity_output(
    site_id: str,
    passage_id: str,
    input_video_sha256: str,
    recognized_train_names: List[str],
    visual_service_numbers: List[str],
    recognized_endpoints: List[str],
    service_family_candidates: List[str],
    active_service_number: Optional[str],
    active_direction_verified: bool,
    identity_status: str,
    reason_codes: List[str],
    evidence_provenance: List[Dict[str, Any]],
    contradictions: List[str],
    processing_metrics: Optional[Dict[str, Any]] = None,
    decision_created_utc: Optional[str] = None,
    passage_metadata: Optional[Dict[str, Any]] = None,
    visual_evidence: Optional[Dict[str, Any]] = None,
    resolver_decision: Optional[Dict[str, Any]] = None,
    performance_latency: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    C4: Produces atomic identity JSON matching the schema_version: shadow_identity_v1 contract.
    Preserves backward-compatible nested fields so existing consumers don't break.
    """
    created_utc = decision_created_utc or datetime.now(timezone.utc).isoformat()

    doc: Dict[str, Any] = {
        "schema_version": "shadow_identity_v1",
        "passage_id": passage_id,
        "site_id": site_id,
        "input_video_sha256": input_video_sha256,
        "recognized_train_names": list(recognized_train_names),
        "visual_service_numbers": list(visual_service_numbers),
        "recognized_endpoints": list(recognized_endpoints),
        "service_family_candidates": list(service_family_candidates),
        "active_service_number": active_service_number,
        "active_direction_verified": active_direction_verified,
        "identity_status": identity_status,
        "reason_codes": list(reason_codes),
        "evidence_provenance": list(evidence_provenance),
        "contradictions": list(contradictions),
        "processing_metrics": dict(processing_metrics or {}),
        "decision_created_utc": created_utc,
    }

    # Backward compatibility
    if passage_metadata is not None:
        doc["passage_metadata"] = passage_metadata
    if visual_evidence is not None:
        doc["visual_evidence"] = visual_evidence
    if resolver_decision is not None:
        doc["resolver_decision"] = resolver_decision
    if performance_latency is not None:
        doc["performance_latency"] = performance_latency

    return doc
