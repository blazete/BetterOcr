# Phase 7 — Logic-First 14-Recording Evaluation Plan

## 1. Executive Summary & Authority

This document defines the formal, frozen Phase 7 evaluation plan for the ShadowOCR train identity pipeline. 

The evaluation suite is strictly bounded to exactly **14 passage recordings**:
1. **10 Development Recordings**: `video_001.mp4` through `video_010.mp4` (established in Train Identity Lab, benchmarked in Phase 6, re-evaluated under logic-first rules in Phase 7).
2. **4 Independent Held-Out Evaluation Recordings**: `coach_id11.mp4` through `coach_id14.mp4`, staged in an isolated environment (`heldout_4/`) for single-shot, truth-isolated blind evaluation once all four are available.

All production ARDS code, databases, identities, recordings, sensor logs, and defect data remain unmodified. Live deployment remains **strictly OFF**.

---

## 2. Balanced Two-Pool Selection Algorithm

The evaluation configuration freezes the balanced two-pool selection algorithm in `two_stage_sampler.py` and `cli.py`:

- **Pool A (High-Quality Board Proposals)**:
  - Up to **15 OCR crop calls** dedicated to valid board proposals (`confidence >= 0.35`).
  - Evaluated across 2.0-second temporal coverage buckets (`timestamp_ms // 2000`).
  - Maximum **2 crops per 2-second bucket** to guarantee temporal diversity across passage transit.
  - Ranked by `(confidence_score, frame_sharpness, -frame_index)`.

- **Pool B (Unused Proposals & High-Quality Fallbacks)**:
  - Up to **10 OCR crop calls** for the highest-quality unused crops.
  - Sources: remaining unselected proposals and sharpest screened fallback crops.
  - **Spatial Deduplication**: Rejects crops on the same frame with bounding box IoU $> 0.5$.
  - **Temporal Deduplication**: Rejects identical fallback crops captured $< 300\text{ ms}$ apart.

- **Hard Inference Limit**:
  - Exactly **25 maximum actual `engine.predict()` calls**, never exceeded under any input conditions.

---

## 3. Status of the Three Frozen Logic Corrections

Prior to held-out evaluation, the three core logic corrections specified for Phase 7 have been fully implemented, verified via RED $\to$ GREEN regression suites, and **frozen**:

### Correction 1: Ground Truth Metric Separation & Directional Integrity
- **Implemented In**: `shadow_lab/src/shadow_identity/shadow_ocr/worker.py` (`evaluate_against_ground_truth`).
- **Logic**:
  - Removed the unverified 4-digit prefix heuristic (`actual[:4] == expected[:4]`).
  - Separated `exact_active_service_match` (evaluated **only** when GT contains an independently verified single active service number; set to `None` for bidirectional pairs) from `pair_membership_match` and `service_family_match`.
  - Added explicit tracking of `direction_label_quality`, `abstained`, `incorrect_provisional`, `false_confirmed`, and `gt_provenance`.
  - Preserved the Phase 6 benchmark results (`shadow_lab/data/phase_6_evaluation_results.json`) as immutable historical record.

### Correction 2: Strict Token-Level Name & Endpoint Route Matching
- **Implemented In**: `shadow_lab/src/shadow_identity/text_normalization.py` & `shadow_lab/src/shadow_identity/candidate_resolver.py`.
- **Logic**:
  - Eliminated the unrestricted substring shortcut, replacing it with whole-token Jaccard similarity over non-generic train tokens (`EXP`, `EXPRESS`, `SF`, `SUPERFAST`, `SPL`, `SPECIAL`, `MAIL` filtered). Short tokens ($< 4$ characters) match only when explicitly registered in curated site aliases.
  - Curated route endpoint matching enriches candidate services with `ROUTE_ENDPOINT_MATCH` (+2.0) and `ROUTE_ENDPOINT_PAIR_MATCH` (+3.5) exclusively against official site metadata (`sites/{site_id}/aliases.yaml`).
  - **Hard Gate**: Route endpoint or name evidence alone **never** produces `CONFIRMED` status.

### Correction 3: Fail-Safe Decision Gates & Contradiction Detection
- **Implemented In**: `shadow_lab/src/shadow_identity/candidate_resolver.py`.
- **Logic**:
  - **Video 004 Safeguard**: Single numbers seen only on unverified frames / 1 coach window with no independent corroboration return `UNKNOWN, final_number=None` (preserving raw candidates in candidate services). This permanently eliminates the false provisional prediction `02276`.
  - **Bidirectional Pair Ambiguity**: When $\ge 2$ visual numbers or a stable pair are present without independent active-direction corroboration (e.g. fresh NTES or hardware tracking), the resolver declares `AMBIGUOUS` with `final_number=None` rather than guessing an arbitrary first member.
  - Leading zeros are strictly preserved (e.g., `02276`, `02563`, `02570`).

---

## 4. Held-Out Evaluation Protocol (`heldout_4/`)

To prevent overfitting, hyperparameter tuning leaks, or data snooping, a dedicated, truth-isolated held-out evaluation environment is established:

### A. Directory Architecture
```
heldout_4/  (symlinked to shadow_lab/data/heldout_4/)
├── README.md                          # Protocol rules & instructions
├── inbox/                             # Read-only intake for 4 held-out MP4s (+ .done markers)
├── manifest/
│   └── ground_truth_manifest.json     # Read-only GT manifest (chmod 444)
├── output/                            # Destination for frozen initial predictions & report
└── storage/                           # Isolated staging directory for TempMp4LifecycleManager
```

### B. Read-Only Ground Truth Manifest
- Stored at `heldout_4/manifest/ground_truth_manifest.json` with read-only permissions (`chmod 444`).
- Contains exactly **4 recording slots**:
  1. `coach_id11.mp4` (`pas_coach_id11`)
  2. `coach_id12.mp4` (`pas_coach_id12`)
  3. `coach_id13.mp4` (`pas_coach_id13`)
  4. `coach_id14.mp4` (`pas_coach_id14`)
- Strictly isolated from inference: read only after all initial predictions and raw OCR evidence are durably persisted to disk.

### C. Strict Blind-Testing Rules
1. **Zero Pre-Inspection**: The 4 new videos must **not** be inspected, previewed, or probed for algorithm tuning before the initial benchmark run.
2. **Zero Ground-Truth Ingestion**: The ground-truth manifest must **never** be passed to, read by, or used as an input to OCR, frame selection, candidate ranking, or candidate resolution during inference.
3. **All-or-Nothing Finalization Requirement**:
   - Requires all four recordings (`coach_id11.mp4` through `coach_id14.mp4`) and authoritative finalization markers (`.done`) before starting inference.
   - If any recording or finalization marker is missing, the runner cleanly halts with `BLOCKED_DATA` without processing a partial set.
4. **Single-Shot Initial Run**: When all 4 recordings and authoritative signals are staged, they will be processed exactly once with the frozen configuration, and initial raw outputs durably persisted before any post-hoc analysis.
5. **Frozen Algorithm Integrity**: No thresholds, weights, sampling ratios, or code paths may be altered between development and held-out evaluation runs.
6. **Separation of Results**: The 4 held-out results will be compared and reported separately from the 10 development-video results.

---

## 5. Frozen Evaluation Configuration

Both development and held-out evaluation use the identical frozen parameter suite:

| Parameter | Frozen Value | Architectural Rationale |
|---|---|---|
| **Screening Frame Rate** | `5.0 FPS` | Ensures uniform sampling density across passage transit |
| **Max OCR Call Budget** | `25 calls` | Enforced at crop-level inference call (bounded strictly to 25 PaddleOCR calls) |
| **Sampling Strategy** | `balanced_two_pool_selection` | Pool A (max 15 proposals, 2s buckets, max 2/bucket) + Pool B (max 10 deduplicated fallbacks) |
| **OCR Engine** | `PaddleOCR PP-OCRv6_medium` | CPU inference, textline orientation enabled, angle classification disabled |
| **Preprocessing Profile** | `P1_COLOR_2X_BICUBIC` | 2x bicubic RGB upscaling with contrast preservation |
| **Timetable Source** | `ndls_validated.csv` | Canonical site schedule; no current-time fallbacks |
| **NTES Corroboration** | `HONEST_ABSENT` | Honest missing report; strictly zero synthetic NTES substitution |
| **Finalization Required** | `True` | Requires authoritative `.done` marker / metadata sidecar |
| **Max Concurrency** | `1 worker` | Ensures deterministic, repeatable latency and resource benchmarking |

---

## 6. Metric Evaluation Standard (P0.3 Strict Alignment)

Outputs are scored strictly according to independent evidence provenance:

1. **Exact Active-Service Accuracy**:
   - **Evaluated ONLY** where ground truth contains an independently verified, single-number active running direction.
   - Evaluated as `True` if prediction matches the active service; `False` if discordant; `None` if GT is a bidirectional rake pair.
2. **Pair Membership Accuracy**:
   - Evaluated as `True` if prediction is a member of the independently verified service pair.
3. **Service-Family Accuracy**:
   - Evaluated via explicit family mapping / alias dictionary.
4. **Abstention Rate**:
   - Fraction of passages safely declared `UNKNOWN` or `AMBIGUOUS` with `final_number=None`.
5. **Incorrect Provisionals**:
   - Count of uncorroborated or incorrect provisional predictions (target: strictly 0).
6. **False Confirmations**:
   - Count of uncorroborated or incorrect confirmed predictions (target: strictly 0).
7. **Performance & Provenance**:
   - Latency percentiles ($p_{50}, p_{90}, p_{95}$), memory RSS peak, actual OCR calls executed, board buckets covered, and full SHA-256 recording signatures.

---

## 7. Execution Command Reference

### A. Development Suite (10 Passages)
```bash
# Replay development suite and update phase_7_evaluation_results.json
PYTHONPATH=.:shadow_lab/src ./train-identity-lab/.venv/bin/python shadow_lab/scripts/run_phase7_evaluation.py
```

### B. Held-Out Suite (4 Passages)
```bash
# Execute single-shot blind evaluation on heldout_4/
PYTHONPATH=.:shadow_lab/src ./train-identity-lab/.venv/bin/python shadow_lab/scripts/run_heldout_evaluation.py
```

If recordings are not yet present in `heldout_4/inbox/`, the runner cleanly reports:
```text
[BLOCKED_DATA] Held-out evaluation requires all 4 finalized recordings:
  Expected: ['coach_id11.mp4', 'coach_id12.mp4', 'coach_id13.mp4', 'coach_id14.mp4']
  Missing MP4 files (4): ['coach_id11.mp4', 'coach_id12.mp4', 'coach_id13.mp4', 'coach_id14.mp4']
  Currently valid and finalized: 0 / 4
- Benchmark will NOT execute a partial set. Aborting inference until all 4 recordings are finalized.
```

---

## 8. Post-Run Failure Analysis Protocol

Following the initial execution on `heldout_4/`:
1. Durably store all initial predictions, raw observations, and summary metrics in `heldout_4/output/heldout_4_evaluation_results.json`.
2. Analyze discrepancies between predictions and ground truth, identifying root optical, temporal, or schedule causes.
3. **Strict Prohibition**: Do **not** modify algorithms, tune thresholds, or alter candidate weights based on held-out results until reviewed and approved by the operator.
4. Compare the 4 held-out results separately from the 10 development-video baseline.
5. Verify bitwise integrity of all 456 production ARDS files via `verify_originals_unchanged.py`.
6. Keep live deployment **OFF**.
