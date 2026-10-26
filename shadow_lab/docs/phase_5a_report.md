# Phase 5A — Hardened ShadowOCR CLI Report

## 1. Executive Summary & Verdict

- **Decision Verdict**: **PASS — Hardened ShadowOCR CLI Successfully Implemented & Verified**
- **Core Deliverable**:
  - Hardened CLI entrypoint: [cli.py](file:///Users/shreyanshmalviya/Desktop/BetterOcr/shadow_lab/src/shadow_identity/shadow_ocr/cli.py)
  - Regression Test Suite: [test_phase5a_hardening.py](file:///Users/shreyanshmalviya/Desktop/BetterOcr/shadow_lab/tests/shadow_ocr/test_phase5a_hardening.py)
  - Raw OCR Evidence JSONL output format for audit provenance
  - Production ARDS Integrity: **456 / 456 original files bitwise verified unchanged**
  - Complete Test Suite: **137 / 137 tests passing (100% pass rate)**

---

## 2. Implementation of Required Fixes

| Requirement | Implementation Detail | Verification Status |
| :--- | :--- | :--- |
| **1. Verified Source Timestamps** | Eliminated `datetime.now()` clock fabrication. If `passage_start` or `passage_end` is missing, `time_corroboration_enabled=False` and timestamps are left empty (`""`), explicitly disabling schedule window matches. | **VERIFIED** ([test_missing_timestamps_disable_time_corroboration](file:///Users/shreyanshmalviya/Desktop/BetterOcr/shadow_lab/tests/shadow_ocr/test_phase5a_hardening.py#L27)) |
| **2. Coach Boundary Provenance** | Implemented `validate_coach_boundaries()` enforcing trusted sensor source (`ARDS_HARDWARE_BEAM`, `SYNCHRONIZED_SENSOR`, `VERIFIED_AUDIT`), passage ID alignment, camera ID alignment, non-negative intervals, and bounds within video frame count. Untrusted/mismatched JSON is strictly downgraded to `UNVERIFIED`. | **VERIFIED** ([test_forged_coach_boundaries_downgraded_to_unverified](file:///Users/shreyanshmalviya/Desktop/BetterOcr/shadow_lab/tests/shadow_ocr/test_phase5a_hardening.py#L48)) |
| **3. TempMp4LifecycleManager Staging** | Added `--stage-temp-copy` and `--temp-dir` integration. Production-origin MP4 is safely copied into `shadow_lab/data/temp_mp4/`, verified, processed, recorded as `COMPLETED`, and scheduled for 60-minute cleanup without touching the original file. | **VERIFIED** ([test_temp_mp4_lifecycle_staging_and_cleanup](file:///Users/shreyanshmalviya/Desktop/BetterOcr/shadow_lab/tests/shadow_ocr/test_phase5a_hardening.py#L119)) |
| **4. NTES +180s Timing Contract** | Records snapshot `acquisition_time`, `availability`, relative offset `relative_to_passage_end_sec`, `freshness_valid`, and flags `late_verification` when the snapshot was acquired $\ge 180$ seconds post passage end. | **VERIFIED** ([test_ntes_timing_contract_and_late_verification](file:///Users/shreyanshmalviya/Desktop/BetterOcr/shadow_lab/tests/shadow_ocr/test_phase5a_hardening.py#L152)) |
| **5. Configurable Camera-Specific Resolution ROIs** | Replaced hardcoded `(0, 130, 768, 280)` fallback crop with camera-specific configurable ROIs from `ShadowOcrSettings` scaled dynamically to the native video resolution (`width`, `height`). Preserved Video 007 recognition and Video 011 KATRA regression fix. | **VERIFIED** ([cli.py:270-281](file:///Users/shreyanshmalviya/Desktop/BetterOcr/shadow_lab/src/shadow_identity/shadow_ocr/cli.py#L270-L281)) |
| **6. Raw OCR Observations JSONL Persistence** | Added `--raw-observations <path.jsonl>`. Every raw model observation is written out with `frame_id`, `timestamp_ms`, `board_roi_xyxy`, `raw_text`, `ocr_confidence`, `preprocessing_variant`, `coach_id`, `coach_id_source`, and `source_video_sha256`. | **VERIFIED** ([test_raw_ocr_observations_jsonl_persistence](file:///Users/shreyanshmalviya/Desktop/BetterOcr/shadow_lab/tests/shadow_ocr/test_phase5a_hardening.py#L176)) |
| **7. Security & Boundary Regression Tests** | 8 comprehensive regression tests created covering missing timestamps, forged coach boundaries, mismatched cameras/passages, out-of-bounds frames, staging lifecycle, NTES contracts, raw JSONL persistence, and end-to-end corridor corroboration. | **VERIFIED** (8/8 passing in 97.5s) |

---

## 3. Real Timetable + NTES Corridor Corroboration Replay

### Passage Test Scenario:
- **Video**: Video 007 (`train-identity-lab/videos/video_007.mp4`)
- **Visual Evidence**: Rake carries stable visual pair `18103` (17 coach observations) and `18104` (15 coach observations) with destination `AMRITSAR`.
- **Timetable Schedule**: Row for `18103 ASR EXP` scheduled at 12:00 at Platform 12.
- **NTES Snapshot**: Acquired at 12:03 (+180s contract) confirming `18103` on Platform 12.
- **Staging**: `--stage-temp-copy` enabled (staged in `shadow_lab/data/temp_mp4/temp_ndls_pf12_13_test_v007_pass_1791975049.mp4`).

### Resolver Outcome:
```json
{
  "resolver_decision": {
    "shadow_final_number": "18103",
    "shadow_status": "CONFIRMED",
    "reason_codes": [
      "NTES_CORROBORATED_FRESH_BOARD",
      "VISUAL_PAIR_CONFIRMED_GE3_COACHES"
    ],
    "ranked_candidates": [
      {
        "train_number": "18103",
        "train_name": "ASR EXP",
        "score": 19.5,
        "evidence_list": [
          "TIMETABLE_EXACT_NUMBER_MATCH",
          "TIMETABLE_TIME_WINDOW_DELTA_0MIN",
          "TIMETABLE_PLATFORM_12",
          "VISUAL_NUMBER_GE3_COACHES",
          "NTES_PRESENT_PF_12",
          "CORROBORATION_VISUAL_PAIR_AND_NTES"
        ]
      },
      {
        "train_number": "18104",
        "train_name": "",
        "score": 4.5,
        "evidence_list": [
          "VISUAL_NUMBER_GE3_COACHES"
        ]
      }
    ],
    "timetable_available": true,
    "ntes_available": true,
    "ntes_timing_contract": {
      "verified_timestamps_present": true,
      "ntes_available": true,
      "acquisition_time": "2026-10-14T12:03:00+05:30",
      "relative_to_passage_end_sec": 20.0,
      "late_verification": false,
      "freshness_valid": true
    }
  }
}
```

### Analysis:
- Without Timetable/NTES: The visual pair `18103` vs `18104` is symmetric and returns `shadow_status="AMBIGUOUS"` (no guessing).
- With Verified Timetable + NTES: Directional and platform corroboration resolves the candidate set to **`18103` CONFIRMED** with score 19.5 vs 4.5.
- The ground truth was never fed to the resolver or candidate selector.

---

## 4. Replay Comparison Across Benchmark Videos

| Video | Video Properties | Verified Coach Tracking | Visual Entities Recovered | Shadow Status (Standalone) | Shadow Status (Corroborated) | Raw JSONL Observations | Wall-Clock Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Video 007** | 768x576, 25fps, 3997 frames (159.9s) | `VERIFIED` (3 distinct coaches) | Pair: `18103`, `18104`; Endpoint: `AMRITSAR` | `AMBIGUOUS` (Pair confirmed, bidirectional) | `CONFIRMED` (`18103`, score 19.5) | 76 rows ([video_007_raw_observations.jsonl](file:///Users/shreyanshmalviya/Desktop/BetterOcr/shadow_lab/results/video_007_raw_observations.jsonl)) | 37.17s |
| **Video 008** | 768x576, 25fps, 6350 frames (254.0s) | `UNVERIFIED` (Asymmetric support) | Service: `12920`; Name: `MALWA EXPRESS` / `MALWASUPERFAST` | `UNKNOWN` (Provisional pair, no schedule match) | `UNKNOWN` | 77 rows ([video_008_raw_observations.jsonl](file:///Users/shreyanshmalviya/Desktop/BetterOcr/shadow_lab/results/video_008_raw_observations.jsonl)) | 43.79s |
| **Video 011** | 768x576, 25fps, 3717 frames (148.7s) | `UNVERIFIED` (No boundaries provided) | Endpoints: `KATRA`, `HAPA` | `UNKNOWN` (Graceful abstention on SD video) | `UNKNOWN` | 12 rows ([coach_id11_raw_observations.jsonl](file:///Users/shreyanshmalviya/Desktop/BetterOcr/shadow_lab/results/coach_id11_raw_observations.jsonl)) | 26.91s |
| **Video 012** | 768x576, 25fps, 3868 frames (154.7s) | `UNVERIFIED` (No boundaries provided) | Endpoints: `KATRA`, `NEW DELHI`, `VAISHNO DEVI` | `UNKNOWN` (Night passage, graceful fallback) | `UNKNOWN` | 28 rows ([coach_id12_raw_observations.jsonl](file:///Users/shreyanshmalviya/Desktop/BetterOcr/shadow_lab/results/coach_id12_raw_observations.jsonl)) | 42.78s |

---

## 5. Production ARDS Integrity Check

Verification run via `shadow_lab/scripts/verify_originals_unchanged.py`:
- Total original files checked: **456**
- Modified files: **0**
- Missing files: **0**
- Result: **[PASS] All 456 original ARDS files verified unchanged (zero modifications).**

---

## 6. Conclusion and Phase Gate

Phase 5A is complete. All 7 required fixes and the real timetable + NTES corridor test are verified with working code, unit tests, replays, and zero modifications to production ARDS. 

**Execution halted after Phase 5A for user review as instructed. No production deployment performed.**
