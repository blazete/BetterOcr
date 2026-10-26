# ShadowOCR Implementation Progress

## Baseline
- **Date**: 2026-10-14
- **Branch**: `feature/shadow-ocr-phase1`
- **Baseline Git SHA**: `f809d02`
- **Initial Test Suite**: 95 passed in 0.53s
- **Production Integrity**: 456 files verified (0 mismatches, 0 missing)

## Phase 0: Audit & Specification Alignment
- **Status**: Completed (Gate P0: PASS)
- **Delivered**: `shadow_lab/docs/phase_0_audit_report.md`, `shadow_lab/docs/decision_log.md`
- **Findings**:
  1. ARDS video finalization lifecycle traced to `VideoRecorder.stop_recording()`.
  2. MP4 completion requires writer release and drain join.
  3. Ground-truth leakage in legacy lab script identified and eliminated from design.
  4. 10-video batch MP4s noted as missing from archive (`BLOCKED_DATA` for full historical replay); Videos 11 & 12 present and verified.

## Phase 2: Railway Entity Classification, Physical-Coach Consensus & Visual Identity
- **Status**: Completed (Gate P2: PASS)
- **Delivered**:
  - `shadow_identity/shadow_ocr/text_entities.py` (EntityClassifier)
  - `shadow_identity/shadow_ocr/consensus.py` (PhysicalCoachConsensusEngine)
  - `shadow_identity/shadow_ocr/adapter.py` (ShadowOcrToBetterOcrAdapter)
  - `shadow_identity/shadow_ocr/temp_mp4_lifecycle.py` (TempMp4LifecycleManager)
  - `tests/shadow_ocr/test_phase2_consensus.py` (15 unit & adversarial tests)
  - `docs/phase_2_gate_report.md`
  - `docs/phase_2_verification_addendum.md`
- **Audit Findings & Evidence**:
  1. Real outputs inspected: `video_11_visual_identity.json` and `video_12_visual_identity.json`.
  2. Physical-coach boundary attribution traced: absence of hardware beam interval metadata safely falls back to `UNVERIFIED` and blocks `PAIR_CONFIRMED`.
  3. Negative deduplication confirmed: multiple frames/variants from 1 coach equal 1 vote.
  4. Video 007 full passage replayed with zero ground-truth leakage: recovered `18103 / 18104` with 6 distinct coaches each.
  5. Temporary MP4 lifecycle manager verified for 60-minute retention, active job protection, crash recovery, and anti-traversal security.
  6. Phase 2 CPU overhead measured: 1.37 ms for 100 observations (<0.02% of pipeline latency).
  7. Test suite: 118/118 passing in 1.04s.
  8. Production ARDS integrity: 456/456 SHA-256 hashes verified (0 mismatches, 0 modified).

## Phase 3A: MP4 OCR Recognition & Frame-Sampling Diagnosis
- **Status**: Completed Diagnosis (Gate P3A: BLOCKED)
- **Delivered**: `shadow_lab/docs/phase_3a_diagnostic_report.md`
- **Key Findings**:
  1. **Sampling Stride**: At 1 FPS with `max_frames_per_passage = 10`, the uniform stride skipped the brief ~1-second board passings. A 1-2 second scan over the board passage successfully captures board sightings.
  2. **ROI Aspect Ratio**: A full 768x150 horizontal strip squashes board text during PaddleOCR resizing. A windowed/focused crop recovers bold destination names (`KATRA` with conf 0.98 under P3 CLAHE).
  3. **Resolution Mismatch**: The standalone video files are 768x576 (SD). Secondary 5-digit stencil numbers are only ~8-10px tall and suffer from motion blur at 25 FPS, whereas production ARDS cameras record at 5 Megapixels (2592x1944).
  4. **Multi-Variant Isolation**: Verified that P1, P2, and P3 detections share the same physical frame and coach ID and are never counted as independent coach votes.
  5. **Gate Verdict**: BLOCKED on raw 5-digit number recovery from 576p video alone, pending native high-resolution recordings or destination-name-first matching. Production ARDS integrity 100% verified (456/456 hashes).

## Phase 3C: High-Resolution Recording Availability Audit
- **Status**: Completed (Verdict B: No existing high-resolution MP4 available)
- **Delivered**: `shadow_lab/docs/phase_3c_report.md`
- **Audit Findings**:
  1. **ARDS Recording Pipeline Traced**: Audited `inspection_session.py:336-413` and `agent-work-log.md:8652`. Discovered that `raw_clip_path()` special-cases `coach_id` to NOT record a `_raw.mp4`. Only downscaled review clips (`REVIEW_CLIP_SCALE_W = 768`) are written for the OCR camera role to conserve disk and CPU bandwidth.
  2. **Recording Inventory**: 100% of available offline MP4s (23 of 23 files) are confirmed downscaled 768x576 clips. Zero 1080p or 5MP MP4s exist in the repository or offline datasets.
  3. **OCR Budget Allocation Benchmark**: Evaluated 15, 30, and 60 frame budgets on 768w video (`coach_id11.mp4`). Increasing budget from 15 to 60 calls roughly tripled runtime (21.02s to 65.45s) and increased raw token observations from 8 to 33, but produced 0 five-digit service numbers due to the fundamental optical limitation (~2.5–8px character height).
## Phase 4: End-to-End ShadowOCR Train Identity CLI
- **Status**: Completed (Gate P4: PASS)
- **Delivered**:
  - `shadow_identity/shadow_ocr/cli.py` (Unified CLI entry point)
  - `shadow_lab/tests/shadow_ocr/test_phase4_cli.py` (Integration tests)
  - `shadow_lab/docs/phase_4_report.md`
- **Key Achievements**:
  1. **Runnable Single Command**: Implemented `python -m shadow_identity.shadow_ocr.cli` taking `--mp4`, `--site-id`, `--passage-id`, `--platform`, `--coach-boundaries`, `--timetable`, and `--ntes-snapshot`.
  2. **End-to-End Replay Demonstrated**: Replayed Video 007 (`18103`/`18104` + `AMRITSAR`), Video 008 (`12920` + `MALWA EXP`), Video 011 (`KATRA` endpoint fallback), and Video 012 (night Devanagari script fallback).
  3. **Canonical Resolver Arbitration**: Adapts visual identity into canonical `VisualOcrEvidence` and invokes existing frozen `resolve_shadow()` without code changes.
  4. **Strict Safety Rules**: Verified that absence of coach boundaries blocks `PAIR_CONFIRMED`; verified that repeated frames/variants cannot multiply coach votes.
## Phase 5A: Harden the Working ShadowOCR CLI
- **Status**: Completed (Gate P5A: PASS)
- **Delivered**:
  - `shadow_identity/shadow_ocr/cli.py` (Hardened with timestamp verification, boundary provenance checks, TempMp4 staging, NTES +180s timing contract, resolution-aware fallback, and raw JSONL export)
  - `shadow_lab/tests/shadow_ocr/test_phase5a_hardening.py` (Unit and integration regression tests)
  - `shadow_lab/docs/phase_5a_report.md` (Detailed benchmark report and audit results)
  - Raw evidence JSONL exports for Videos 007, 008, 011, and 012

## Phase 5A Corrections & Automatic Shadow Worker
- **Status**: Completed (Gate P5A-Worker: PASS)
- **Delivered**:
  - `shadow_identity/shadow_ocr/worker.py` (Autonomous worker for discovery, size-stability checking, idempotent staging, asynchronous execution, and crash recovery)
  - `shadow_identity/shadow_ocr/cli.py` (Hardened boundary provenance with hardware sync requirement, strict +180s NTES timing contract, and atomic file persistence)
  - `shadow_lab/tests/shadow_ocr/test_phase5a_worker.py` (Worker unit tests for discovery, idempotency, lifecycle, and crash recovery)
  - `shadow_lab/scripts/demonstrate_shadow_worker.py` (End-to-end working demonstration script)
- **Key Corrections & Implementations**:
  1. **NTES Timing Contract & Synthetic Reclassification**:
     - Explicitly separated: scheduled verification time, actual fetch time, verification delay, snapshot freshness, stale fallback, and eligibility for resolver confirmation.
     - Enforced strict +180s verification rule: Snapshots fetched earlier than +180s (e.g. at +20s) or marked as stale cached fallback are ineligible for T2 post-passage confirmation.
     - Added negative regression tests preventing premature or stale confirmation.
     - Formally reclassified test fixtures as `SYNTHETIC_TEST_FIXTURE` and separated from actual archived snapshots.
  2. **Physical Coach Boundary Provenance**:
     - Banned blind acceptance of keyword strings (`ARDS_HARDWARE_BEAM`, `SYNCHRONIZED_SENSOR`).
     - Required verifiable hardware synchronization provenance (`hardware_sync_verified=True`, `sync_method`).
     - Enforced rejection of missing passage/camera IDs and strictly validated non-overlapping intervals (`sf <= last_end` rejected).
     - Without verified hardware sensor synchronization, visual pair confirmation remains disabled (`PAIR_PROVISIONAL` only).
  3. **Atomic Durable Persistence**:
     - Final decision JSON and raw evidence JSONL are written atomically (`.tmp_pid` rename + `os.fsync`) before marking staging jobs completed.
     - The 60-minute cleanup retention begins strictly *after* successful durable persistence.
  4. **Automatic Shadow Worker**:
     - Scans read-only inbox for finalized MP4s using file size stability and OpenCV frame count probing.
     - Stages copies idempotently in shadow storage (`worker_jobs.json` tracking).
     - Bounded concurrency with asynchronous pipeline execution.
     - Automatic recovery of interrupted/crashed jobs on startup (`PROCESSING` -> `STAGED`).
     - Ground truth evaluation comparing resolved numbers against independent ground truth.
  5. **Working Demonstration**:
     - Demonstrated discovery of real replayed MP4 (`video_007.mp4`, SHA-256: `f7f8ba65...`), staging, OCR inference, timetable corroboration, atomic persistence, ground truth comparison (`18103` == `18103`), restart crash recovery, and bitwise verification of original MP4.
  6. **Suite & Integrity**:
     - Full test suite passing.
     - Production ARDS integrity: **456 / 456 files verified 100% untouched**.
     - Live deployment remains strictly OFF.

## Phase 6: Automatic Multi-Passage ShadowOCR Evaluation
- **Status**: Completed (Gate P6: PASS for offline benchmark; Live trial: BLOCKED)
- **Delivered**:
  - `shadow_lab/scripts/run_phase6_evaluation.py` (Autonomous multi-passage benchmark script)
  - `shadow_lab/data/phase_6_evaluation_results.json` (Structured benchmark metrics & per-passage results)
  - `shadow_lab/docs/phase_6_report.md` (Comprehensive benchmark & diagnostic report)
  - Upgraded `shadow_identity/shadow_ocr/worker.py` (Collision-resistant keys, authoritative finalization, thread-safe persistence, crash recovery)
  - Hardened `shadow_identity/candidate_resolver.py` & `shadow_identity/timetable_matcher.py` (Eliminated fabricated wall-clock timetable matching when timestamps are missing)
- **Key Benchmark Findings**:
  1. **Multi-Passage Replay (10/10 Passages)**:
     - 100% of available Train Identity Lab videos (`video_001` through `video_010`) successfully evaluated under frozen settings.
     - **Exact Match Accuracy**: 40.0% (4 / 10 passages).
     - **Service-Family Match Accuracy**: 50.0% (5 / 10 passages: Videos 001, 007, 008, 009, 010).
     - **Safe Abstention Rate**: 40.0% (4 / 10 passages: Videos 002, 003, 005, 006 with non-numbered boards or night glare cleanly returned `None`).
     - **False Confirmations**: **0 (0.0%)** (Zero false confirmations across all 10 passages).
  2. **Latency & Throughput**:
     - Latency percentiles: p50 = 34.3s, p90 = 43.2s, p95 = 45.4s (well within <60s per passage requirement).
     - 2-worker concurrent throughput: 53.57s for 2 full video passages in parallel (26.78s effective throughput per passage).
  3. **Resource Consumption**:
     - Peak RAM (RSS): 1,521.8 MB under dual-worker inference.
     - Disk: Temporary staging automatically cleared after 60-minute retention.
  4. **Production ARDS Integrity**:
     - Verified bitwise identical: **456 / 456 files 100% untouched**.
     - Live deployment remains gated on hardware coach boundary logs and authentic NTES API.
