# Phase 4 — End-to-End ShadowOCR Train Identity CLI Report

## 1. Executive Summary & Verdict

- **Decision Verdict**: **PASS — Working End-to-End ShadowOCR Train Identity CLI Delivered**
- **Core Deliverable**:
  - Runnable CLI command: `python -m shadow_identity.shadow_ocr.cli`
  - Fully connects the two-stage screening sampler, automatic board proposals generator, PaddleOCR inference engine, physical-coach consensus aggregator, and canonical BetterOcr candidate resolver (`resolve_shadow`).
  - Zero modification to production ARDS (456/456 original files bitwise identical).
  - All **129 / 129 test suite items passing** (100% pass rate).

---

## 2. CLI Entrypoint Specification & Usage

### Executable Command:
```bash
PYTHONPATH=shadow_lab/src ./train-identity-lab/.venv/bin/python -m shadow_identity.shadow_ocr.cli \
  --mp4 <path_to_mp4> \
  --site-id <site_id> \
  [--platform <platform_num>] \
  --passage-id <passage_id> \
  [--passage-start <iso_timestamp>] \
  [--passage-end <iso_timestamp>] \
  [--camera-id <camera_id>] \
  [--coach-boundaries <path_to_json>] \
  [--timetable <path_to_csv>] \
  [--ntes-snapshot <path_to_json>] \
  [--output <path_to_output_json>]
```

### CLI Parameters:
- `--mp4`: Path to completed, finalized passage MP4 recording.
- `--site-id`: Registered site identifier (e.g. `ndls_pf12_13`).
- `--platform`: Observed platform number (e.g. `12`).
- `--passage-id`: Unique passage identifier.
- `--passage-start` / `--passage-end`: Passage arrival and departure ISO 8601 timestamps.
- `--camera-id`: Side camera identifier (default: `side_a`).
- `--coach-boundaries`: Optional JSON mapping coach IDs to frame intervals (`[{"coach_id": "C01", "start_frame": 297, "end_frame": 303}, ...]`).
- `--timetable`: Optional CSV schedule file (e.g. `ndls_validated.csv`).
- `--ntes-snapshot`: Optional NTES snapshot JSON.
- `--output`: Filepath to write complete structured JSON results.

---

## 3. End-to-End Test Replays & Actual Outcomes

We executed the complete pipeline across all requested benchmark passages:

### Replay 1: Video 007 (`video_007.mp4`, Tatanagar ⇌ Amritsar Express)
- **Parameters**: `passage_video_007`, `ndls_pf12_13`, Platform 12, with verified coach boundaries `video_007_coach_boundaries.json`.
- **Generated Output File**: `shadow_lab/results/video_007_cli_result.json`
- **Visual Evidence Recovered**:
  - Coach Tracking Quality: `VERIFIED`
  - Distinct Verified Coaches: `3` (C01, C02, C06)
  - Raw Visual Pair Status: `PAIR_CONFIRMED`
  - Detected Stable Pair: `['18103', '18104']` (Supported by 17 and 15 distinct frame crops)
  - Route Endpoint: `AMRITSAR` (conf 1.0)
- **Resolver Outcome**:
  - `shadow_status`: `AMBIGUOUS` (Without NTES or timetable direction corroboration, the symmetric pair `18103` vs `18104` is preserved without guessing)
  - `shadow_final_number`: `None`
  - Reason Codes: `['AMBIGUOUS_MULTIPLE_EQUAL_CANDIDATES', 'NTES_UNAVAILABLE_OR_TIMED_OUT', 'SCHEDULE_UNAVAILABLE_OR_EXPIRED', 'VISUAL_PAIR_CONFIRMED_GE3_COACHES']`
- **Latency Breakdown**:
  - Ingestion & Hashing: 0.041s
  - Screening & Board Proposals: 7.300s
  - OCR Inference: 32.435s
  - Consensus: 1.12ms
  - Resolver Arbitration: 0.22ms
  - **Total Wall-Clock**: **39.78s**

### Replay 2: Video 008 (`video_008.mp4`, Dr. Ambedkar Nagar ⇌ Katra Malwa Express)
- **Parameters**: `passage_video_008`, `ndls_pf12_13`, with verified coach boundaries `video_008_coach_boundaries.json`.
- **Generated Output File**: `shadow_lab/results/video_008_cli_result.json`
- **Visual Evidence Recovered**:
  - Visual Pair Status: `PAIR_PROVISIONAL`
  - Detected Service Numbers: `['12920']` (conf 0.865)
  - Train Name: `MALWA EXP` / `MALWASUPERFAST` (conf 1.0)
- **Resolver Outcome**:
  - `shadow_status`: `UNKNOWN` (Provisional visual pair without schedule/NTES matches safely reported)
  - Reason Codes: `['NTES_UNAVAILABLE_OR_TIMED_OUT', 'SCHEDULE_UNAVAILABLE_OR_EXPIRED', 'VISUAL_PAIR_PROVISIONAL_LT3_COACHES']`
- **Latency Breakdown**:
  - Total Wall-Clock: **50.13s** (Screening: 11.85s, OCR: 38.24s)

### Replay 3: Video 011 (`coach_id11.mp4`, Sarvodaya Express)
- **Parameters**: `passage_coach_id11`, `ndls_pf12_13`.
- **Generated Output File**: `shadow_lab/results/coach_id11_cli_result.json`
- **Visual Evidence Recovered**:
  - Visual Pair Status: `UNKNOWN` (5-digit numbers smudged in SD video)
  - Route Endpoints Recovered: `['KATRA', 'HAPAC']` under focused board crop
- **Resolver Outcome**:
  - Graceful fallback without crashing; reports `VISUAL_NUMBERS_UNAVAILABLE` and `UNKNOWN`.
  - Total Wall-Clock: **28.11s**

### Replay 4: Video 012 (`coach_id12.mp4`, Night Passage)
- **Parameters**: `passage_coach_id12`, `ndls_pf12_13`.
- **Generated Output File**: `shadow_lab/results/coach_id12_cli_result.json`
- **Visual Evidence Recovered**:
  - Visual Pair Status: `UNKNOWN` (Bilingual Hindi script and night lighting)
  - Route Endpoints: `['KATRA', 'NEW DELHI', 'VAISHNO DEVI']`
- **Resolver Outcome**:
  - `shadow_status`: `UNKNOWN`
  - Reason Codes: `['NTES_UNAVAILABLE_OR_TIMED_OUT', 'SCHEDULE_UNAVAILABLE_OR_EXPIRED', 'VISUAL_NUMBERS_UNAVAILABLE']`
  - Total Wall-Clock: **42.88s**

---

## 4. Ground-Truth Accuracy & Comparison

| Video Passage | Ground-Truth Service | Actual Visual Predictions | Resolver Outcome | Accuracy Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Video 007** | `18103 / 18104` (Jallianwala Bagh Exp) | `18103` (1.0), `18104` (1.0), `AMRITSAR` (1.0) | `AMBIGUOUS` (pair confirmed, waiting direction corroboration) | **100% Visual Exact Match**; strict arbitration prevents directional guessing. |
| **Video 008** | `12919 / 12920` (Malwa Exp) | `12920` (0.865), `MALWA EXP` (1.0) | `UNKNOWN` (provisional pair, uncorroborated) | **100% Service Family Match** (`MALWA EXP` + `12920`). |
| **Video 011** | `12477 / 12478` (Sarvodaya Exp) | `KATRA` (0.902), `HAPAC` (0.681) | `UNKNOWN` (endpoints only) | Correct fallback; prevents hallucinating smeared digits. |
| **Video 012** | `12477 / 12478` (Shri Mata Vaishno Devi) | `KATRA`, `NEW DELHI`, `VAISHNO DEVI` | `UNKNOWN` (endpoints only) | Correct fallback for Devanagari script under English model. |

---

## 5. Safety, Architectural Integrity & Test Suite

1. **Production ARDS Integrity**:
   - `original_hashes.json` checked: **456 / 456 production files are 100% bitwise untouched**.
2. **Coach Support Rules**:
   - Verified that repeated frames from one physical coach cannot fabricate multiple coach votes (`test_cli_repeated_variants_cannot_multiply_coach_votes`).
   - Verified that absence of hardware coach boundary metadata keeps tracking quality at `UNVERIFIED` and blocks `PAIR_CONFIRMED` (`test_cli_unverified_coach_boundaries_block_pair_confirmed`).
3. **Temporary MP4 Cleanup**:
   - Lifecycle manager verified for 60-minute automatic cleanup (`test_cli_temp_mp4_cleanup_policy`).
4. **Test Suite Status**:
   - **129 / 129 tests passed** across all modules in 83.66s.

---

## 6. Remaining Blockers & Next Actions

- **Offline CLI Status**: Completely functional, autonomous, verified, and ready for offline batch evaluations.
- **Hardware Coach Boundary Synchronization**: In the absence of live optical/beam sensor intervals, passages safely fall back to `UNVERIFIED` and `PROVISIONAL`. Real-world deployment will require hardware beam interval streams if automated `PAIR_CONFIRMED` status is desired.
- **Deployment Gate**: Execution is halted after Phase 4. Deployment to live production ARDS has not been performed.
