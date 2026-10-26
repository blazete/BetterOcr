# Phase 1 Gate Report: MP4 Staging, Frame Diversity & OCR Inference

**Date:** 2026-10-14  
**Git Branch:** `feature/shadow-ocr-phase1`  
**Git Baseline Commit:** `f809d02`  
**Execution Environment:** Python 3.12.13, PaddlePaddle 3.3.1, PP-OCRv6, macOS Darwin  
**Gate Status:** **PASS**

---

## 1. Scope & Implementation Delivered

In Phase 1, we implemented the read-only, completed MP4 passage ingestion and inference pipeline under `shadow_lab/src/shadow_identity/shadow_ocr/`:

1. **Contracts & Dataclasses (`contracts.py`)**:
   - `PassageVideoInput`: Strict validation of input MP4 path, site/installation ID, camera ID, timezone, duration, and SHA-256 file checksums.
   - `BoardRoiConfig`: Camera-specific ROI boundaries validated against camera resolution.
   - `SampledFrame`: Quality-scored, time-diverse frame descriptors with explicit coach boundary provenance (`VERIFIED_BOUNDARY` vs `UNVERIFIED`).
   - `RawOcrObservation`: Structured, machine-readable JSONL contract retaining model ID, preprocessing profile, raw text, confidence, polygon, and full-frame mapped coordinates.

2. **MP4 Source Ingestion (`sources/finalized_mp4.py`)**:
   - Strictly read-only MP4 container access with OpenCV `cv2.VideoCapture`.
   - Rejects non-existent files, 0-byte files, and invalid containers.

3. **Camera Board ROI (`board_roi.py`)**:
   - Extractor ensuring all crops remain within valid camera resolution boundaries with safe clamping.

4. **Time & Coach Diversity Sampler (`frame_sampler.py`)**:
   - Implements time-diverse uniform sampling across passage duration instead of naive global sharpness clustering.
   - Calculates Laplacian variance sharpness, brightness mean, and contrast standard deviation.

5. **Preprocessing Engine (`preprocess.py`)**:
   - P1 Primary: Exact 2x Bicubic interpolation on BGR color crops.
   - P2 Reproducibility: Pinned gamma LUT ($\gamma = 0.7$, $1/\gamma \approx 1.43$) demonstrating verified midtone darkening.
   - P3 Fallback: Grayscale -> CLAHE (`clipLimit=2.5`, `tileGridSize=(8,8)`) -> 2x Bicubic -> BGR.

6. **Pinned PaddleOCR Engine Wrapper (`paddle_engine.py`)**:
   - Singleton lazy-loading of cached `PP-OCRv6` models (`PP-OCRv6_medium_det`, `PP-OCRv6_medium_rec`).
   - Bounded error handling returning empty results on no-text detections without crashing.

7. **Pipeline Orchestrator (`pipeline.py`)**:
   - Connects: Finalized MP4 $\rightarrow$ Quality/Diversity Sampler $\rightarrow$ P1 Preprocessing $\rightarrow$ PaddleOCR $\rightarrow$ Raw Observations JSONL.

---

## 2. Test Verification & Safety Results

- **Unit Test Suite**: `shadow_lab/tests/shadow_ocr/test_phase1_pipeline.py` (8 tests) **100% PASS**.
  - Verified ROI bounds rejection (out-of-bound $x_2$, inverted $y_1 \ge y_2$).
  - Verified P1 2x scaling geometry.
  - Verified P2 gamma LUT mathematical darkening ($128 \rightarrow 95$).
  - Verified P3 CLAHE 3-channel BGR output.
  - Verified missing MP4 file safe error reporting.
  - Verified container metadata extraction on real passage video.
  - Verified time diversity ordering across video timestamps.
- **Full Shadow Lab Test Suite**: **103 passed in 0.87s** (up from 95 baseline).
- **Production ARDS Integrity**: All 456 production file SHA-256 hashes verified (0 mismatches, 0 missing).

---

## 3. Real MP4 Passage Ingestion Results

Phase 1 was validated directly on the real passage videos:

### A. Passage Video 11 (`coach_id11.mp4`)
- **Video Path**: `train-identity-lab/videos/coach_id11.mp4` (SHA-256 computed and recorded).
- **Duration**: 148.68s (3717 frames @ 25.0 FPS).
- **Output JSONL**: `shadow_lab/results/dev_gates/phase_1/iter_1/video_11_observations.jsonl` (2,517 bytes).
- **Sample Observations**:
  - Frame 725 (29.0s): `ENTR` (conf: 0.99, entity: `TRAIN_NAME`)
  - Frame 1100 (44.0s): `AC THREE TIER` (conf: 0.95, entity: `TRAIN_NAME`)
  - Frame 2250 (90.0s): `FHIFY` (conf: 0.57, entity: `TRAIN_NAME`)

### B. Passage Video 12 (`coach_id12.mp4`)
- **Video Path**: `train-identity-lab/videos/coach_id12.mp4` (SHA-256 computed and recorded).
- **Duration**: 154.72s (3868 frames @ 25.0 FPS).
- **Output JSONL**: `shadow_lab/results/dev_gates/phase_1/iter_1/video_12_observations.jsonl` (4,951 bytes).
- **Sample Observations**:
  - Frame 1150 (46.0s): `1630` (conf: 1.00, coach stencil)
  - Frame 1925 (77.0s): `3A` (conf: 0.99, coach class)
  - Frame 2700 (108.0s): `AC THREE TIER` (conf: 0.97, entity: `TRAIN_NAME`)

---

## 4. Gate P1 Verdict: PASS

Phase 1 is complete and verified. Ready for Phase 2 (Text parsing, candidate token classification, and multi-coach consensus aggregation).
