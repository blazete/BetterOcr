# BetterOcr / ShadowOCR V1 — Validation Checklist

**Date**: 2026-10-26  
**Target**: ShadowOCR V1 Release Candidate  
**Audit Scope**: Core Architecture, Evidence Policies, and Safeguards  

---

## 1. Service-Family Mapping Provenance
- [x] **Authoritative Grounding**: Canonical train names and their service family number pairs (e.g., `12445 / 12446` for *Uttar Sampark Kranti*, `12715 / 12716` for *Sachkhand Express*, `12919 / 12920` for *Malwa Express*) originate from official Northern Railway timetables (`ndls_validated.csv`) and validated railway schedules.
- [x] **No Unverified Hardcoding of Active Service**: Curated entries in `TRAIN_NAME_REGISTRY` (in `shadow_identity.shadow_ocr.text_entities`) and `aliases.yaml` associate names strictly with their bidirectional *service family*, never assuming or hardcoding an active direction.
- [x] **Deterministic Association Fallback**: `candidate_resolver.py` looks up site-specific `aliases.yaml` first, followed by the curated `TRAIN_NAME_REGISTRY`, ensuring 100% resolution for all known services without unverified manual entries.
- [x] **Decoupled Output Contract**: Service family candidate numbers are emitted into `service_family_candidates` and never prematurely assigned to `active_service_number` or `shadow_final_number`.

---

## 2. Single-Coach Name Evidence
- [x] **Typed Entity Disambiguation**: Text observed on physical coaches is strictly classified into typed railway entities (`SERVICE_NUMBER`, `TRAIN_NAME`, `ENDPOINT`, `COACH_SERIAL`, `COACH_DESIGNATION`, `OTHER_TEXT`).
- [x] **Consensus vs Single-Coach Hierarchy**:
  - 5-digit service numbers require $\ge 3$ distinct physical coaches with verified hardware boundaries to reach `PAIR_CONFIRMED`.
  - A destination board observed on a single coach with high confidence (e.g. *Uttar Sampark Kranti* at Frame 1660 with 0.9931 confidence) is preserved as a `TRAIN_NAME` candidate in consensus.
- [x] **Strict Non-Promotion to Active Number**: Single-coach name evidence identifies the `service_family_candidates` with status `AMBIGUOUS`, but cannot confirm an active running service number (`active_service_number = None`, `active_direction_verified = False`).
- [x] **Negative Stencil and Coach Serial Immunity**:
  - 6-digit coach serials (e.g. `243560`, `124922`) and slash serials (`217152/C`) are strictly typed `COACH_SERIAL` and rejected from service numbers.
  - Seating stencils (`TO SEAT 1-90`) and safety markings (`EMERGENCY WINDOW`) are strictly rejected with `SEATING_SAFETY_SIGN`.
  - Coach exterior class stencils (`GENERAL`, `LUGGAGE`, `SECOND CLASS`) are strictly typed `COACH_DESIGNATION`.

---

## 3. Active-Direction Confirmation
- [x] **No Direction Guessing**: Active direction is never inferred from the first-listed number in a pair, arbitrary timetable ordering, or train name alone.
- [x] **Rigorous Corroboration Prerequisites**: Active direction confirmation (`active_direction_verified = True`, `shadow_status = CONFIRMED`) requires either:
  1. An authentic, fresh NTES snapshot acquired $\ge +180$ seconds after passage completion matching platform and time window; OR
  2. Verified hardware track movement telemetry (inbound vs outbound track sensors).
- [x] **Safe Abstention Protocol**: In the absence of authentic NTES or verified directional movement telemetry, the pipeline emits:
  - `shadow_status: AMBIGUOUS`
  - `active_direction_verified: False`
  - `active_service_number: None`
  - `shadow_final_number: None`
  - Reason codes: `RC_NAME_ONLY_ACTIVE_DIRECTION_UNVERIFIED` and `RC_AMBIGUOUS_MULTI_CANDIDATE`.
- [x] **Anti-Leakage Safeguard**: Synthetic or delayed test fixtures are isolated from operational pipelines and cannot trigger live confirmation.

---

## 4. Remaining Board-Visibility Uncertainties
- [x] **Video 11 (`coach_id11.mp4`)**:
  - Crops captured coach serials (`243560`, `252002`) and class stencils (`GENERAL`).
  - Destination board visibility remains `INCONCLUSIVE / UNDETERMINED` pending human operator bounding-box annotations.
  - Fail-closed output: `status: UNKNOWN`, `shadow_final_number: None` (zero false predictions).
- [x] **Videos 13 & 14 (`coach_id13.mp4`, `coach_id14.mp4`)**:
  - Utkrisht livery (ochre/yellow body bands) and yellow window frames generated 550+ candidate proposals, crowding out mid-coach destination boards if present.
  - Crops captured `"TO SEAT 1-90"`, `"EMERGENCY WINDOW"`, `"LUGGAGE"`.
  - Destination board visibility remains `INCONCLUSIVE / UNDETERMINED` pending human operator bounding-box annotations.
  - Fail-closed output: `status: UNKNOWN`, `shadow_final_number: None` (zero false predictions).
- [x] **No Fabricated Assertions**: The system reports `accuracy_status: INSUFFICIENT_DATA` for unverified held-out cases rather than fabricating accuracy percentages or assuming `BOARD_NOT_VISIBLE`.
