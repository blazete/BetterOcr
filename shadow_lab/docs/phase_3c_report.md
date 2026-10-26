# Phase 3C — High-Resolution Recording Availability Audit Report

## 1. Executive Summary & Verdict

- **Decision Verdict**: **B. No existing high-resolution MP4 available**
- **Core Recommendation**:
  - Keep **MP4-first with endpoint/name fallback** in ShadowOCR.
  - Formally document the unresolved physical stencil number-OCR limitation for downscaled 768-width clips.
  - Do **not** increase the deployed OCR frame evaluation budget beyond the default without evidence of improved service number recovery (experiments show 60 OCR calls on 768w clips double latency without recovering 5-digit service numbers).
  - Strictly maintain ARDS zero-modification safety boundaries.

---

## 2. Task 1 — ARDS Recording Output Tracing

An inspection of `ARDS-main/orchestrator/inspection_session.py`, `video_recorder.py`, `ocr_pipeline.py`, `_raw_record_loop`, and historical work logs was conducted to trace raw and review recording lifecycles.

### Detailed Findings:
1. **Native-Resolution Recording Behavior (`_raw_record_loop`)**:
   - `_raw_record_loop` (`inspection_session.py:336-413`) records unscaled frames using `FfmpegWriter(out_path, fps=fps, faststart=False)` into `{role}_raw.mp4`.
   - **Crucial Distinction**: `inspection_session.py:1045` and `agent-work-log.md:8652` explicitly state:
     > *"raw_clip_path() special-cases coach_id (no `_raw.mp4`)"*
   - The `coach_id` OCR camera role **does not record a separate raw MP4** in production ARDS. It writes solely to the review MP4 (`coach_id.mp4` / downscaled to `REVIEW_CLIP_SCALE_W = 768`).
2. **Review MP4 Resolution, Codec, & Retention**:
   - Resolution: `768x576` (or `768` width maintaining aspect ratio).
   - Codec: H.264 (AVC1), yuv420p, 25 or 30 FPS.
   - File Path Pattern: `recordings/{session_id}/{passage_id}/coach_id.mp4` (or `review_clips/{role}.mp4`).
   - Finalization: Handled by `FfmpegWriter.close()` upon passage completion before manifest compilation.
3. **Passage ID, Camera ID, and Timestamps**:
   - Preserved in passage session directory structures, session manifests (`session.json`), and filename metadata.
4. **Copy Safety**:
   - Finalized review MP4s are closed prior to session finalization and can be safely read or copied by ShadowOCR without lock contention or buffer tearing.
5. **Deployed Configuration Reality**:
   - Source-code capability exists for detector roles to record raw video, but deployed ARDS configuration for `coach_id` intentionally downsizes video to 768w review clips to bound disk I/O and CPU load during live passages.

---

## 3. Task 2 — Representative Recording Validation

A forensic scan of all MP4 assets across the entire repository and lab directories was executed:
- **Total MP4 Files Discovered**: 23
- **Resolution Distribution**:
  - `768x576`: 23 files (100%)
  - `1920x1080` (1080p): 0 files (0%)
  - `2560x1920` (5MP native): 0 files (0%)
- **Files Inspected**:
  - `video_001.mp4` through `video_010.mp4`: `768x576` @ 25/30 FPS
  - `coach_id11.mp4` (148.68s, 4460 frames): `768x576` @ 30 FPS, AVC1
  - `coach_id12.mp4` (154.72s, 4642 frames): `768x576` @ 30 FPS, AVC1
- **Conclusion**:
  - No native-resolution MP4 recordings exist in the repository or offline datasets. All recordings are downscaled 768w review clips.

---

## 4. Task 3 — OCR Budget Allocation Benchmark

Using `coach_id11.mp4` (768w clip), we benchmarked Phase 1 inference across 15, 30, and 60 frame evaluation budgets concentrated around detected board windows.

| Budget (Frames) | Observations Extracted | OCR Runtime (s) | Extracted Tokens | Service Numbers Recovered | Endpoints Recovered | Train Names Recovered |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **15** | 8 | 21.02s | `['AC', 'AC TO T', 'B', 'ENWSOFIPNE', '1', '1']` | 0 | 0 | 0 |
| **30** | 19 | 34.51s | `['A', 'Ca', 'AC', '2', 'AC TO T', 'B']` | 0 | 0 | 0 |
| **60** | 33 | 65.45s | `['ND CLASE', 'A', '153008/C', 'Ca', '三', 'AC']` | 0 | 0 | 0 |

### Key Benchmark Observations:
1. **Diminishing Returns on Resolution-Constrained Video**:
   - Increasing the budget from 15 to 60 frames roughly triples the execution time (21.02s → 65.45s) and increases observation count (8 → 33), but **fails to recover 5-digit service numbers**.
   - At 60 frames, coach serial stencils (`153008/C`) and class markings (`ND CLASE`, `AC`) are extracted, but the physical destination board 5-digit service numbers remain illegible due to pixel blur (~2.5–8px character height at 768w).
2. **Budget Policy Recommendation**:
   - Retain the baseline budget (~15–30 frames). Do not inflate the deployed default budget to 60 frames, as it incurs a ~65-second latency penalty without solving the root optical resolution deficit.

---

## 5. Task 4 — Architecture, Safety, & Integrity Audit

- **Production ARDS Integrity**:
  - `original_hashes.json` verification: **456 / 456 files bitwise identical (0 modified, 0 missing)**.
- **RTSP / Buffer Isolation**:
  - Zero live camera buffers accessed.
  - Zero RTSP ingestion bridges added.
- **Temporary Copy Retention**:
  - 60-minute automatic cleanup policy maintained in `temp_mp4_lifecycle.py`.
- **Test Suite Verification**:
  - All **123 tests passing** in `pytest shadow_lab/tests/` (3.88s runtime).

---

## 6. Formal Decision & Next Steps

### Verdict: **B. No existing high-resolution MP4 available**
1. **Grounding**:
   - `coach_id` raw recording is disabled by design in ARDS; only 768w review clips are written to disk.
   - All 23 representative MP4s in the test suite are 768x576.
2. **Strategy**:
   - Maintain the finalized MP4-first architecture.
   - Leverage **endpoint and train-name visual fallback** to identify train passages (which successfully resolve bold text like `KATRA`, `SACHKHAND`, etc.).
   - Explicitly document the physical stencil 5-digit number OCR limitation under downscaled 768w review video.
3. **Boundary**:
   - Phase 3C is complete. No live deployment or production ARDS modifications are performed.
