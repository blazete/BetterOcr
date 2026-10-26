# Phase 0 Audit Report: ARDS Recording Lifecycle & BetterOcr Integration Contracts

**Date:** 2026-10-14  
**Git Branch:** `feature/shadow-ocr-phase1`  
**Git Baseline Commit:** `f809d02`  
**Evaluation Scope:** Production ARDS Recording Lifecycle, BetterOcr Shadow Architecture, Train Identity Lab Pipeline.

---

## 1. Executive Summary & Verification Boundaries

- **Production ARDS Path**: Frozen and strictly read-only. Verified 456/456 file integrity SHA-256 hashes against `shadow_lab/workspace/fixtures/original_hashes.json` (0 mismatches, 0 missing).
- **Existing Baseline Test Suite**: 95/95 tests passing across all unit, integration, isolation, and safety suites.
- **Input Strategy Decision**: Locked to **completed, copied MP4 recordings** inside shadow-owned storage. No direct RTSP, no camera frame bridge, and no modification to production capture/recorder pipelines.

---

## 2. Actual ARDS Recording Lifecycle Audit

We audited the production ARDS recording subsystem in `ARDS-main/orchestrator/`:

### A. Recording Creation & Encoding
- **Source**: `ARDS-main/orchestrator/video_recorder.py` (`VideoRecorder` and `RecordingManager`).
- **Recorder Threading**: Dedicated daemon thread `_recording_loop` draining a thread-safe frame queue (`threading.Thread(target=self._recording_loop, daemon=True)`).
- **Encoding Engine**: ffmpeg subprocess via pipe or OpenCV `cv2.VideoWriter`. Encodes frames to MP4 containers (`h264_nvenc` or `libx264`).
- **Storage Target**: Configurable `output_dir` (default `recordings/` or per-inspection session directory under `runs/`).

### B. Recording Finalization & Readiness Signal
- **Trigger**: `OCRPipeline.stop_session(camera_id)` in `ARDS-main/orchestrator/ocr_pipeline.py:1298-1331` calls `recorder.stop_recording()`.
- **Drain & Flush**: `stop_recording()` sets `stop_event`, joins `recording_thread` with a `30.0s` timeout, calls `self._writer.release()`, and computes exact encoded duration from `frames_written / fps`.
- **Crucial Operational Insight**:
  1. An MP4 file on disk is **not safe to read** while `is_recording == True` or before ffmpeg/VideoWriter writes the final container `moov` atom header.
  2. The completion event returns a dictionary containing:
     - `filename`: Final MP4 filename.
     - `duration`: Exact duration in seconds.
     - `frames_written`: Count of written frames.
     - `first_capture_at` & `last_capture_at`: High-precision UTC/epoch timestamps.
     - `filesize_mb`: File size in megabytes.
  3. **Readiness Guarantee for Shadow Ingestion**:
     Shadow staging must **never** ingest by raw filename appearance alone. Staging requires:
     - Either an explicit passage completion manifest/signal containing `frames_written > 0` and `status == "stopped"`, OR
     - A strict file stability check (file size and mtime constant over an interval, plus OpenCV container probe verifying readable frame count).

---

## 3. BetterOcr Shadow Architecture & Compatibility Audit

### A. Current Components
- `shadow_identity/models.py`: Immutable dataclass contracts (`PassageInput`, `VisualObservation`, `NumberCandidate`, `NameCandidate`, `PassageVisualConsensus`, `VisualOcrEvidence`, `ShadowDecision`, `ComparisonRecord`).
- `shadow_identity/ocr_adapter.py`: `OcrAdapter.load_from_per_frame_csv()` loads historical CSVs, classifies entity types (`SERVICE_NUMBER`, `TRAIN_NAME`, `ENDPOINT`, `NEGATIVE_STENCIL`), and builds `PassageVisualConsensus` via `CoachConsensusAggregator`.
- `shadow_identity/candidate_resolver.py`: `resolve_shadow(passage, ocr_evidence, timetable_rows)` applies frozen arbitration rules.
- `shadow_identity/delayed_ntes_worker.py`: Manages the +180s delayed NTES verification loop.

### B. Required Compatibility Matrix

| Current Contract / Field | New ShadowOCR Producer | Mapping / Behavior in ShadowOCR |
|---|---|---|
| `PassageInput` | `PassageVideoInput` | `PassageVideoInput.to_passage_input()` maps `passage_id`, `site_id`, `passage_start`, `passage_end`, `direction`. |
| `VisualObservation` | `contracts.RawOcrObservation` | Maps `coach_segment_id`, `frame_id`, `timestamp_offset_ms`, `raw_text`, `normalized_text`, `entity_type`, `ocr_confidence`. |
| `PassageVisualConsensus` | `contracts.PassageVisualConsensus` | Identical schema. Preserves `service_number_candidates`, `stable_pair`, `train_name_candidates`, `endpoints`, `number_board_type`, `numeric_status`. |
| `VisualOcrEvidence` | `adapter.to_visual_ocr_evidence()` | Emits standard `VisualOcrEvidence` accepted by `resolve_shadow()` without modifying the resolver. |

---

## 4. Train Identity Lab Audit: Proven vs Experimental Logic

### A. Critical Audit Findings
1. **Ground Truth Leakage in Validation Script**:
   In `experiments/run_final_validation_batch.py:461`, `stable_pair` was populated using `if c in gt_set: stable_pair.append(c)`.
   **Remedy for ShadowOCR**: Strict prohibition. Ground truth is strictly evaluator-only; inference has zero knowledge of ground truth.
2. **Video 008 Confirmation Policy**:
   The validation report marked Video 008 as `PAIR_CONFIRMED` despite candidate 12920 having only 2 supporting coaches while 12919 had 5.
   **Remedy for ShadowOCR**: Both numbers require $\ge 3$ distinct verified coaches for `PAIR_CONFIRMED`; otherwise `PAIR_PROVISIONAL`.
3. **Manual Coach Centers vs Auto-Tracking**:
   The lab script hardcoded coach frame centers.
   **Remedy for ShadowOCR**: When physical coach boundaries are not externally provided, coach support is marked `UNVERIFIED` and cannot trigger `PAIR_CONFIRMED` via repeated frames.
4. **Gamma LUT Direction**:
   P2 uses $\gamma = 0.7$ with exponent $1/\gamma \approx 1.43$, which **darkens** midtones. This must be maintained for historical fidelity while logging its actual behavior.

---

## 5. Hardware, Model, and Missing Video Inventory

- **Model Environment**:
  - Python: 3.12.13 (in `train-identity-lab/.venv`)
  - PaddlePaddle: 3.3.1 (CPU mode)
  - PaddleOCR: PP-OCRv6 medium models cached locally in `~/.paddlex/official_models/` (`PP-OCRv6_medium_det`, `PP-OCRv6_medium_rec`).
- **Video Assets Inventory**:
  - Videos 1–10: The raw MP4 files are not in the repository. Per specification Section 1.0/7.0, full 10-video replay is marked `BLOCKED_DATA` until MP4s are supplied.
  - Videos 11 & 12: Available at `train-identity-lab/videos/coach_id11.mp4` and `train-identity-lab/videos/coach_id12.mp4` and tested successfully in preceding runs.
- **Production Isolation**:
  - `allow_production_writes = False`
  - `allow_network = False` (default)
  - `shadow_ocr_enabled = False` (feature-flagged off by default)

---

## 6. Phase 0 Gate Sign-Off

- [x] All 456 ARDS production files verified unchanged (0 mismatches).
- [x] All 95 existing BetterOcr tests pass.
- [x] ARDS recording lifecycle audited and documented.
- [x] Ground-truth anti-leakage rules established.
- [x] Phase 0 Gate: **PASS**.
