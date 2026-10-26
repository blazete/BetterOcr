# Phase 2 Gate Report: Railway OCR Entity Classification, Physical Coach Consensus & Passage Visual Identity

**Date:** 2026-10-14  
**Git Branch:** `feature/shadow-ocr-phase1`  
**Git Baseline Commit:** `f809d02`  
**Execution Environment:** Python 3.12.13, PaddlePaddle 3.3.1, PP-OCRv6, macOS Darwin  
**Gate Status:** **PASS**

---

## 1. Scope & Implementation Delivered

In Phase 2, we implemented typed entity classification, strict physical coach attribution, passage-level consensus aggregation, and the safe temporary MP4 retention manager under `shadow_lab/src/shadow_identity/shadow_ocr/`:

1. **Entity Classification (`text_entities.py`)**:
   - `SERVICE_NUMBER`: Identifies 5-digit service numbers (preserving leading zeros e.g. `02563`, and embedded pairs `12477/78`).
   - `TRAIN_NAME`: Normalized candidate train name (e.g. `TAJ EXPRESS`, `SACHKHAND EXPRESS`, `LUCKNOW MAIL`, `SHRI SHAKTI AC EXPRESS`).
   - `ENDPOINT`: Source/destination stations (e.g. `NEW DELHI`, `VGL JHANSI`, `KATRA`, `SVDK`, `JAMNAGAR`, `HAPA`).
   - `COACH_SERIAL`: Exterior coach serial stencils (e.g. `217152/C`).
   - `COACH_DESIGNATION`: Exterior seating classes and negative stencils (e.g. `S1`, `B2`, `3A`, `AC THREE TIER`).
   - `OTHER_TEXT`: Unclassified or ambiguous tokens (retaining full uncertainty).

2. **Physical Coach Consensus & Strict Attribution (`consensus.py`)**:
   - **Negative Rule**: Multiple frames from one single physical coach collapse into **exactly 1 coach vote**.
   - **Boundary Verification Gate**: If coach intervals are not verified (`UNVERIFIED`), distinct coach count is capped at 1 and `PAIR_CONFIRMED` is strictly prohibited.
   - **Strict Pair Confirmation**: Requires that **both** candidate service numbers independently achieve $\ge 3$ distinct verified physical coaches (`PAIR_CONFIRMED`).
   - **Asymmetric Support**: If candidate 1 has $\ge 3$ coaches but candidate 2 has $< 3$ coaches, the pair is held as `PAIR_PROVISIONAL` and candidate 1 is marked `SINGLE_CANDIDATE` (resolving the Video 008 discrepancy).
   - **Name-Only & 4-Number Boards**: Preserves full train-name candidates and endpoint evidence even in the complete absence of service numbers.

3. **BetterOcr Compatibility Adapter (`adapter.py`)**:
   - Seamlessly converts `PassageVisualIdentityResult` to `VisualOcrEvidence`.
   - Plugs directly into `resolve_shadow(passage_input, evidence, timetable_rows)` without altering downstream resolver logic.

4. **Temporary MP4 Lifecycle & Safe Retention (`temp_mp4_lifecycle.py`)**:
   - Staging confined strictly to `shadow_lab/data/temp_mp4/`.
   - Automated 60-minute retention policy (`completed_at + 3600s`) for completed runs and 24-hour retention for failed runs.
   - Guard rails preventing deletion of active/staged jobs, path traversal outside `temp_mp4/`, and any deletion of original ARDS recordings.

---

## 2. Test Verification & Safety Results

- **New Phase 2 Unit Tests**: `shadow_lab/tests/shadow_ocr/test_phase2_consensus.py` (12/12 passed).
  - Verified service number classification (exact 5-digit, leading zero `02563`, embedded `12477/78`).
  - Verified train name classification (`TAJ EXPRESS`, `LUCKNOW MAIL`, `SACHKHAND EXPRESS`, `SHRI SHAKTI AC EXPRESS`).
  - Verified endpoint classification (`NEW DELHI`, `VGL JHANSI`, `KATRA`, `HAPA`).
  - Verified coach serials and designations (`217152/C`, `S1`, `AC THREE TIER`, `3A`).
  - **Negative Test**: 10 repeated frames from 1 coach $\rightarrow$ exactly 1 coach vote, no confirmation.
  - Verified strict pair confirmation when both numbers have $\ge 3$ verified coaches.
  - Verified asymmetric support yields `PAIR_PROVISIONAL` (Video 008 rule).
  - Verified unverified coach boundaries prevent `PAIR_CONFIRMED`.
  - Verified name-only boards preserve train names and endpoints.
  - **Ground-Truth Anti-Leakage**: Demonstrated bitwise identical visual identity output regardless of fake ground-truth labels.
  - Verified BetterOcr adapter compatibility with existing `resolve_shadow()`.
  - Verified temporary MP4 60-minute retention cleanup and absolute protection of production original recordings.
- **Full Test Suite**: **115 passed** in 1.02s (95 baseline + 8 Phase 1 + 12 Phase 2).
- **Production ARDS Integrity**: All 456 production file SHA-256 hashes verified (0 mismatches, 0 missing).

---

## 3. Real MP4 Replay Results

Replayed the real raw observations from `coach_id11.mp4` and `coach_id12.mp4` through the Phase 2 consensus engine:
- `video_11_visual_identity.json` generated in `shadow_lab/results/dev_gates/phase_2/iter_1/video_11_visual_identity.json`.
- `video_12_visual_identity.json` generated in `shadow_lab/results/dev_gates/phase_2/iter_1/video_12_visual_identity.json`.
- **Truthful Attribution**: Correctly marked `coach_tracking_quality: UNVERIFIED` and `reason_codes: ["COACH_BOUNDARIES_UNVERIFIED"]` because video-level playback had no verified external coach boundary signals, demonstrating that our safeguards successfully prevent false coach confirmation claims on raw video without metadata.

---

## 4. Gate P2 Verdict: PASS

Phase 2 is complete, fully tested, and verified.
