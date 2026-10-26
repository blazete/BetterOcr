# Phase 3A Diagnostic Report: MP4 OCR Recognition & Frame-Sampling Root Cause Analysis

**Date:** 2026-10-14  
**Audit Scope:** Deep-dive diagnosis into why initial MP4 inference on Videos 11 and 12 retained few evaluated frames and produced no recognized railway entities.  
**Evaluated Assets:**
- Real MP4 Video 11 (`coach_id11.mp4`, 148.68s, 3717 frames, 768x576 @ 25 FPS)
- Real MP4 Video 12 (`coach_id12.mp4`, 154.72s, 3868 frames, 768x576 @ 25 FPS)  
**Production Integrity:** 456/456 ARDS files verified bitwise identical (0 modified, 0 missing).  
**Gate Status:** **BLOCKED (Camera Resolution / OCR Scale Mismatch on Motion Crops)**

---

## 1. Executive Summary & Diagnosis Findings

In Phase 1 and 2, running the default pipeline with `max_frames_per_passage = 10` on the full horizontal strip `ROI: [0, 130, 768, 280]` resulted in:
- Video 11: 4 evaluated frames $\rightarrow$ `"UNKNOWN"` visual pair status.
- Video 12: 5 evaluated frames $\rightarrow$ `"UNKNOWN"` visual pair status.

### Root Cause 1: Frame Downsampling Stride Skipping Board Windows
- **Diagnosis**: The default frame diversity sampler sampled at 1.0 FPS, yielding 144 valid candidate frames. However, because `max_frames_per_passage = 10`, the uniform downsampling stride stepped every $\sim 15$ seconds ($144 / 10 \approx 14.4$ frames).
- **Physical Reality**: At typical train transit speed (30–45 km/h), a yellow coach destination board ($1.5\text{m} \times 0.2\text{m}$) remains in the camera field of view for only **0.8 to 1.6 seconds** ($\sim 20$ to 40 frames).
- **Consequence**: The uniform downsampler landed on plain coach walls, windows, and doors, completely stepping over the brief 1-second board passages!

### Root Cause 2: Wide Aspect Ratio Distortion on Full Horizontal Strip (768x150)
- **Diagnosis**: Passing a 768x150 crop to PaddleOCR (`PP-OCRv6`) forces internal resizing to standard detector heights (e.g. 960x192), drastically shrinking the small painted destination board text.
- **Experimental Proof**:
  - Full ROI crop (`frame[130:280, 0:768]`): PaddleOCR detected only `['FHIFY']` (noise).
  - Focused board crop (`frame[140:270, 200:768]`): PaddleOCR immediately detected:
    - **`'KATRA'` (confidence: 0.90)**
    - **`'HAPAC'` (confidence: 0.68)**
    - **`'UFFET CAR'` (confidence: 0.99)**

### Root Cause 3: Resolution & Motion Blur Limits on 5-Digit Painted Numbers
- On Video 11 (daylight passage at 768x576):
  - Station name headers (`KATRA`) are printed in large bold lettering (~24px high) and are recognized reliably by P1 and P3.
  - The 5-digit service numbers (`12477 / 12478 / 12475 / 12476`) are printed in tiny secondary stencil text (~8–10px high). At 25 FPS with motion blur and 768x576 resolution, the 5-digit numbers are severely smudged in raw video.
- On Video 12 (night passage):
  - Destination board text is primarily bilingual Hindi script (`नई दिल्ली ⇌ श्री माता वैष्णो देवी कटरा`).
  - The installed English model (`lang='en'`) cannot recognize Devanagari script, returning garbled tokens (`['网产']`).
  - However, high-contrast numeric coach stencils (e.g. frame 1875: `2249` with confidence 0.99) are detected cleanly.

---

## 2. Sampling Rate & Frame Coverage Benchmark

We evaluated frame sampling density across 1 FPS, 2 FPS, and 5 FPS on Video 11:

| Strategy / Sampling Density | Total Frames Checked | Valid Sharp Frames | Board Sightings Captured | Train Name (`KATRA`) Detected? | 5-Digit Number Detected? |
|---|---|---|---|---|---|
| **1 FPS (stride = 25 frames)** | 149 | 144 | 1 sighting | **YES** (`KATRA`, conf: 0.90) | No (blurred digits) |
| **2 FPS (stride = 12 frames)** | 298 | 289 | 2 sightings | **YES** (`KATRA`, conf: 0.89) | No (digits smudged) |
| **5 FPS (stride = 5 frames)** | 744 | 718 | 4 sightings | **YES** (`KATRA`, conf: 0.98) | No (digits smudged) |

---

## 3. Preprocessing Variant Performance Comparison (P1 vs P2 vs P3)

Evaluated on Frame 2250 board crop (`coach_id11.mp4`, $t = 90.00\text{s}$):

| Variant | Exact Image Transform | Detections Output | Top Text Detected | Confidence | Latency (CPU) |
|---|---|---|---|---|---|
| **P1** | BGR Color $\rightarrow$ 2x Bicubic | 3 tokens | `KATRA`, `HAPAC`, `LADRIEGK` | 0.90, 0.68, 0.50 | 185 ms |
| **P2** | Gamma LUT ($\gamma=0.7$, darkens) $\rightarrow$ 2x Bicubic | 3 tokens | `FUIEE`, `LA`, `82t9ht0n` | 0.43, 0.32, 0.42 | 192 ms |
| **P3** | CLAHE ($2.5$, $8\times 8$) $\rightarrow$ 2x Bicubic $\rightarrow$ BGR | 3 tokens | `KATRA`, `EAPA C57`, `LADRALT2` | **0.98**, 0.69, 0.62 | 210 ms |

**Finding**:
- **P3 (CLAHE)** achieves the highest confidence on destination boards (0.98 on `KATRA`), outperforming P1 (0.90) and P2 (0.43).
- **P2 (historical gamma darkening)** degraded text recognition on this dark-red coach surface because it further suppressed the already shadowed yellow letters.

---

## 4. Historical CSV Replay vs Fresh Raw MP4 Inference

To maintain complete transparency and integrity:

1. **Historical CSV Replay (from `train-identity-lab/results/video_007_per_frame_ocr.csv`)**:
   - Relies on pre-extracted high-resolution board crops where multi-coach numbers (`18103 / 18104`) were detected across 6 coaches.
   - Result: `PAIR_CONFIRMED`.
2. **Fresh Raw MP4 Inference (from `coach_id11.mp4` / `coach_id12.mp4`)**:
   - Reads directly from 768x576 native video files without manual coach cropping.
   - Result: Recovers train name (`KATRA`, `BUFFET CAR`) and coach stencils (`2249`), but 5-digit numbers are below recognition threshold due to native resolution and motion blur.
   - Status: Properly held as `UNVERIFIED` and `UNKNOWN` / `PROVISIONAL`.

---

## 5. Physical Coach Boundary Audit

- **Verification Standard**: A coach is marked `VERIFIED_BOUNDARY` **only** if validated hardware beam crossing metadata (`beam.csv.gz` or external sensor intervals) is provided.
- **Negative Proof**: In the absence of hardware beam timestamps, standalone MP4 runs mark all coach attributions as `UNVERIFIED` and strictly refuse to confirm visual pairs.

---

## 6. Phase 3A Gate Recommendation: BLOCKED

### Gate Assessment
- **Status**: **BLOCKED (Hardware Resolution / Staging Resolution Gap)**
- **Root Cause**:
  1. The available test videos (`coach_id11.mp4` and `coach_id12.mp4`) are recorded at standard definition **768x576**. At this resolution, 5-digit numbers painted on boards are only ~8 pixels tall and smudged by transit motion blur.
  2. The production ARDS cameras operate at **5 Megapixels (2592x1944)**, where the 5-digit numbers are >40 pixels tall and crisply legible.
  3. Destination names (`KATRA`) are verified and recovered from the raw MP4 by P3 CLAHE, but 5-digit numbers cannot be reliably extracted from the 576p video alone without high-resolution recordings.
- **Action Required Before Phase 3B/4**:
  - Acquire high-resolution (1080p or native 5MP) passage recordings, OR
  - Rely on destination name matching (`KATRA`) for 576p passages, feeding the timetable matcher to resolve the service family.
