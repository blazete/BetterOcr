# Current ARDS Identity & Resolution Behavior Inventory

**Document Type:** Read-Only Codebase Inventory & Audit  
**Date:** 2026-10-13  
**Audited Target:** `ARDS-main/` (Unchanged, read-only reference)  
**Author:** Antigravity Shadow Lab Implementation Engine  

---

## 1. Executive Summary & Zero-Write Pledge

This document inventories the existing train identity and resolution mechanisms in `ARDS-main/` before implementing the isolated shadow evaluation software in `shadow_lab/`. 

**Non-Negotiable Isolation Contract:**
- `ARDS-main/` remains strictly **read-only**. No production file, database, configuration, model, log, or script will be modified, overwritten, or deleted.
- The shadow software resides exclusively inside `shadow_lab/`.
- All operational network queries (`requests.get/post` to NTES) are disabled by default (`allow_network: false`, `mode: replay_only`).
- The shadow software enforces multi-site isolation, requiring an explicit `--site-id` on all invocations, with no silent fallback to New Delhi (NDLS) defaults.

---

## 2. Anchor Baseline Hashes of Critical Production Files

To prove immutability across the entire shadow evaluation lifecycle, the baseline SHA256 checksums of the critical `ARDS-main` components were computed on 2026-10-13 prior to any shadow development:

| Production File Path | SHA256 Checksum (Prefix) | Role in Production |
|---|---|---|
| `ARDS-main/config/camera_config.yaml` | `19fda1f025c966c5...` | Base system & NTES configuration |
| `ARDS-main/orchestrator/schedule/ntes_cache.py` | `be0619617ceaf270...` | NTES live-board scraper & 3-window cache |
| `ARDS-main/orchestrator/schedule/matcher.py` | `d55149f4e96e6a90...` | Static schedule matcher & service window gate |
| `ARDS-main/orchestrator/inspection/identity/resolver.py` | `984767a31f2dfd20...` | Unified NTES-first identity decision engine |
| `ARDS-main/orchestrator/inspection_session.py` | `49e1ae5e4d0b8c61...` | Inspection session lifecycle & finalize commit |
| `ARDS-main/orchestrator/ocr_pipeline.py` | `3bccf56aafadc429...` | Side-camera coach body serial OCR worker |

---

## 3. Production Component Behavior & Invocation Chain

### 3.1 Session Finalize & Resolution Chain
- **File:** `ARDS-main/orchestrator/inspection_session.py` (lines 1980–2022)
- **Trigger:** Invoked during `InspectionSession.finalize()` when a train passage ends.
- **Workflow:**
  1. Retrieves observed travel direction (`direction_cfg`, line 1980).
  2. Runs static timetable match via `self._schedule_matcher.match(datetime.fromtimestamp(started_at, tz=_IST), platform=None, direction=obs_dir)` (lines 1986–1990).
  3. If no matcher exists (e.g. Katra site), falls back to `MatchResult(candidates=[], confidence="LOW", is_service_event=False)` (lines 1994–1995).
  4. Pulls live NTES snapshot via `self._ntes_cache.get_snapshot()` (line 2000).
  5. Calls `resolve_train_identity(res, self._coach_roster, live_coaches, ntes_snapshot=ntes_snap, direction=obs_dir, recent=self._recent, platforms=...)` (lines 2002–2007).
  6. **Commit Gate (Line 2011–2012):**
     ```python
     auto = fused['is_service_event'] and fused['confidence'] == 'HIGH' and fused['train_number']
     ```
     An automated train number is committed to the inspection record *only* when the event is classified as an active service event and confidence is strictly `HIGH`. Otherwise, `train_number` is saved as `None` (record-only / yard / unconfirmed) with suggestions preserved.

### 3.2 NTES Cache & Scraping Mechanism
- **File:** `ARDS-main/orchestrator/schedule/ntes_cache.py`
- **Class:** `NtesCache` (lines 306–578)
- **TTL Discrepancy:**
  - Class constructor `__init__` sets default `ttl_s = 600` (10 minutes, line 329).
  - Tracked base config `camera_config.yaml` overrides this to `ttl_s = 180` (3 minutes, line 200).
  - *Shadow Rule:* Never assume hardcoded 600s or 180s; inspect and record the effective site profile configuration.
- **Network Requests:** Scrapes `https://enquiry.indianrail.gov.in/mntes/q?opt=LiveStation&subOpt=show` across 3 windows (`nHr = 2, 4, 8`) using `requests.Session` and CSRF tokens (lines 270–301).
- **Audit Persistence:** Appends raw JSONL records to `<audit_dir>/ntes_YYYYMMDD.jsonl` (7-day retention).
- **Passage Audit Thread:** `start_passage_audit()` runs a background thread sampling every 90s (lines 365–400), opt-in per site (`shadow_passage_audit: false`).

### 3.3 Static Schedule Matcher
- **File:** `ARDS-main/orchestrator/schedule/matcher.py`
- **Class:** `ScheduleMatcher` (lines 56–263)
- **Data Source:** Watches `config/train_schedule.json` for mtime changes.
- **Service Gating Windows (lines 41–42):**
  - `_SERVICE_EARLY_MIN = 10`
  - `_SERVICE_LATE_MIN = 90`
  - Asymmetric window accounting for delayed arrivals and on-time departures.
- **Parser Limitations (`scripts/parse_schedule.py`):**
  - Assumes a hardcoded NDLS Excel source file.
  - Hardcodes platform logic and heuristics (e.g. `starts with 6 = EMU`).
  - *Shadow Rule:* Generalize multi-site timetable import; do not hardcode single-site rules.

### 3.4 Unified Identity Resolver
- **File:** `ARDS-main/orchestrator/inspection/identity/resolver.py`
- **Function:** `resolve_train_identity` (lines 350–746)
- **Decision Hierarchy:**
  1. Builds NTES candidate set within `_NTES_PRESENCE_MIN = 20` minutes (lines 37–40).
  2. If NTES candidate set is non-empty, decides from NTES (`_decide_ntes`).
  3. Breaks ties using coach-roster fingerprinting (`RosterScore.strong`).
  4. Falls back to static timetable + roster (`_decide_schedule`) if NTES is absent.
  5. Checks recent assignments for deduplication (`RecentAssignments.duplicate_of`).

### 3.5 Side-View Coach OCR Scope
- **File:** `ARDS-main/orchestrator/ocr_pipeline.py` (lines 1–1406)
- **Role:** Extracts **coach body serial numbers** (e.g. `217152/C` or class stencils `S7`) from side cameras to establish coach sequence and coach-roster fingerprints.
- **Key Realization:** Production ARDS currently **does not extract 5-digit train service numbers or train names from destination boards**. The destination board OCR pipeline developed in `train-identity-lab` is completely additive and will reside entirely within `shadow_lab/`.

---

## 4. Write Side-Effect & Safety Audit

Importing `ARDS-main` modules directly into an external script carries severe side-effect risks:

| Module / Function | Write Side Effect | Reason Why Direct Import is Unsafe |
|---|---|---|
| `poc_orchestrator.py` | Spawns hardware camera threads, binds Web UI port 8080 | Would conflict with live services and hardware cameras |
| `orchestrator/inspection_session.py` | Creates/mutates `runs/inspections/INS_*` directories | Would write incomplete or corrupted inspection bundles |
| `orchestrator/schedule/ntes_cache.py` | Writes to `runs/ntes_audit/ntes_YYYYMMDD.jsonl` | Would mix experimental queries into production audits |
| `orchestrator/inspection/identity/recent_assignments.py` | Writes to in-memory / persistent deduplication memory | Would poison production deduplication state |
| `scripts/parse_schedule.py` | Overwrites `config/train_schedule.json` | Would overwrite site-wide production schedule |

**Shadow Lab Architecture Solution:**
`shadow_lab` will be an independent, self-contained Python package (`shadow_identity`). It will adapt mathematical parsing logic in clean, pure functions, reading production files exclusively in read-only mode, and writing all outputs into `shadow_lab/workspace/`.

---

## 5. Critical Audit: Video 008 Historical Consensus Discrepancy

In accordance with Phase 0 audit requirements, we investigated the discrepancy in the prior `train-identity-lab` summary:
- In `results/final_numeric_ocr_validation.csv`, Video 008 was marked as `passage_correct = YES` and status `PAIR_CONFIRMED` for ground-truth pair `["12919", "12920"]`.
- However, the distinct coach support was:
  - `12920`: **5 distinct physical coaches**
  - `12919`: **2 distinct physical coaches**
- **The Discrepancy:** The declared confirmation rule in prior reports stated "confirmation threshold $\ge 3$ distinct coaches". Under a strict per-candidate threshold, candidate `12919` (with only 2 coaches) failed this threshold!
- **Code Inspection of Prior Runner:** In `experiments/run_final_validation_batch.py` (line 294), the status logic was:
  ```python
  status = "PAIR_CONFIRMED" if (len(stable_pair) == 2 and any(len(overall_coaches[c]) >= 3 for c in stable_pair)) else "PAIR_PROVISIONAL"
  ```
  The script used an **asymmetric rule**: *at least one candidate has $\ge 3$ coaches and both appear at least once*.
- **Frozen Rule for Shadow Lab:** To eliminate ambiguity and maintain strict scientific rigor:
  1. `PAIR_CONFIRMED`: **Both** candidate numbers must independently achieve $\ge 3$ distinct physical coaches.
  2. `PAIR_PROVISIONAL`: At least one candidate achieves $\ge 3$ coaches while the second achieves 1 or 2 coaches.
  3. `PAIR_UNCONFIRMED`: Neither candidate achieves $\ge 3$ distinct coaches.

Under this frozen rule, Video 008 is formally classified as **`PAIR_PROVISIONAL`** (with primary active service `12920` confirmed across 5 coaches).

---

## 6. Phase 0 Deliverable Sign-Off

The baseline audit is complete, files are fingerprinted, risks are mitigated, and the implementation of Phase 1 (`shadow_lab`) can proceed with 100% safety.
