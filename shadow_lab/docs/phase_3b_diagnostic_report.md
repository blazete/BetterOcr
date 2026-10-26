# Phase 3B Diagnostic Report: ARDS Recording Profile, Two-Stage Sampling, and Board Detection Benchmark

**Date:** 2026-10-14  
**Audit Scope:** Verification of actual ARDS recording profile from source code, implementation and benchmarking of two-stage screening vs OCR inference, automatic board proposals, 1/2/5 FPS comparison, entity classification safeguards (`KATRA` as endpoint, `BUFFET CAR` rejection), and failure analysis.  
**Production Integrity:** 456/456 ARDS files verified bitwise identical (0 modified, 0 missing).  
**Gate Recommendation:** **BLOCKED (Resolution / Board Region Legibility Gate)**

---

## 1. Actual ARDS Recording Profile Verification (from Source Code)

We inspected the actual ARDS orchestrator code in `ARDS-main/orchestrator/`:

### A. Recording Architecture & Dimensions
- **Source Files**:
  - `ARDS-main/orchestrator/ocr_pipeline.py:1231-1246`
  - `ARDS-main/orchestrator/video_recorder.py:21-35`
- **Native Capture**: The FLIR camera driver (`flir_camera_driver.py`) captures native frames at full sensor resolution (e.g. 5MP, $2592 \times 1944$ or $1920 \times 1080$).
- **Review Clip Encoding Downscaling**:
  In `video_recorder.py:31-34`:
  ```python
  REVIEW_CLIP_SCALE_W = 768
  REVIEW_CLIP_GOP = 15
  REVIEW_CLIP_QUALITY = 24  # crf (libx264) / cq (nvenc)
  ```
  `ocr_pipeline.py:1237-1245` explicitly initializes `VideoRecorder` for review clips with `scale_w=REVIEW_CLIP_SCALE_W (768)`.
- **Finding**: The saved `.mp4` recordings in ARDS are **deliberately downscaled to width 768** (`768x576` or `768x432`) to allow lightweight web browser playback on review pages without proxy transcoders!
- **Consequence for OCR**:
  - The MP4 files available for shadow evaluation are these 768-width clips.
  - At 768px width across a 24-meter train coach, 1 pixel $\approx 3.1\text{cm}$.
  - The tiny 5-digit stencil numbers on destination boards are only $\sim 8\text{cm}$ tall ($\approx 2.5\text{px}$ to $8\text{px}$ in the MP4).
  - This explains why destination station names (e.g. `KATRA`, printed $\sim 20\text{cm}$ tall) are detected, while the secondary 5-digit numbers are smudged.

---

## 2. Two-Stage Sampling Architecture (Screening vs Inference)

We separated low-cost frame screening from expensive PaddleOCR inference:
- **Stage 1 (Screening)**: Decodes frames at 1, 2, or 5 FPS; computes Laplacian variance on the board ROI strip; checks exposure (brightness and contrast). Throughput: **~40 frames/sec** on CPU.
- **Stage 2 (Targeted OCR Inference)**: Ranks frames by sharpness and board presence; caps OCR inference calls strictly to a configurable budget (`max_ocr_calls = 15`). Throughput: **~5 frames/sec** on CPU.

---

## 3. Automatic Board ROI Proposals Evaluation

We tested a lightweight HSV color-segmentation proposal generator tuned to standard Indian Railway yellow destination boards:

- **Method**: HSV range $H \in [15, 35], S \in [60, 255], V \in [60, 255]$ with rectangular morphological closing ($25 \times 7$) and aspect ratio filter ($\text{aspect} \ge 2.5$).
- **Recall**: Successfully proposed the destination board bounding box on Frames 2210, 2220, 2230, 2250, 2260 (e.g. Frame 2250: `[x=506, y=186, w=262, h=68]`, aspect 3.85).
- **False Proposals**: 2 false proposals on yellow-painted emergency door grab handles.
- **Processing Time**: **< 1.8 ms** per frame.
- **Key Finding**: Tight crops around the yellow board (`w=262, h=68`) cut off the upper stenciled numbers and borders. An expanded margin (`x1 -= 30, y1 -= 30`) is mandatory to keep full text context.

---

## 4. Controlled Benchmark at 1, 2, and 5 FPS (Video 11)

| Parameter / Metric | 1.0 FPS Strategy | 2.0 FPS Strategy | 5.0 FPS Strategy |
|---|---|---|---|
| **Decoded & Screened Frames** | 149 frames | 310 frames | 744 frames |
| **Sharp Candidates Retained** | 144 frames | 289 frames | 718 frames |
| **OCR Calls Budget** | 15 calls | 15 calls | 15 calls |
| **Total Wall-Clock Time** | 17.59 s | 15.16 s | 18.95 s |
| **Board Passage Coverage** | 1 board window captured | 2 board windows captured | 4 board windows captured |
| **Station Endpoints Recovered** | `KATRA` | `KATRA` | `KATRA` |
| **5-Digit Numbers Recovered** | 0 (smudged) | 0 (smudged) | 0 (smudged) |
| **False Candidate Tokens** | 4 | 1 | 2 |

---

## 5. Preprocessing Variant Verification (P1 vs P2 vs P3)

Tested on Frame 2250 board passage:
- **P1 (Primary Color 2x Bicubic)**:
  - Text: `KATRA` (conf 0.90), `HAPAC` (conf 0.68), `LADRIEGK` (conf 0.50).
  - Preserved color information and contrast on stainless steel coaches.
- **P2 (Gamma LUT $\gamma=0.7$, midtone darkening)**:
  - Text: `FUIEE` (conf 0.43), `LA` (conf 0.32).
  - Degraded legibility because darkening midtones suppressed the yellow text against the dark red coach.
- **P3 (CLAHE 2x Bicubic BGR)**:
  - Text: **`KATRA` (conf 0.98)**, `EAPA C57` (conf 0.69).
  - Highest contrast enhancement on reflective metal surfaces.

---

## 6. Entity Classification Safeguards Verification

- **`KATRA`**: Correctly classified as **`ENDPOINT`**, NOT a train name. In `text_entities.py`, `KATRA` is in `KNOWN_ENDPOINTS`, preventing the system from mistaking the destination station for the service name.
- **`BUFFET CAR` / `AC THREE TIER`**: Correctly classified as **`COACH_DESIGNATION`**, completely excluded from service-number candidates.
- **Coach Stencils (e.g. `2249`, `1630`)**: Correctly classified as **`COACH_SERIAL`** or 4-digit token, never treated as 5-digit service numbers.

---

## 7. Gate Recommendation: BLOCKED

### Gate Assessment
- **Status**: **BLOCKED (End-to-End Visual 5-Digit Number Extraction from Downscaled 768w MP4)**
- **Evidence**:
  1. ARDS review MP4 recordings are downscaled to 768px width by design (`REVIEW_CLIP_SCALE_W = 768`).
  2. While major station names (`KATRA`) and coach classes (`BUFFET CAR`, `AC THREE TIER`) are cleanly extracted, 5-digit train numbers on boards are insufficiently resolved at 768w under motion blur.
  3. Strict adherence to our core directive prevents us from relaxing candidate thresholds or fabricating 5-digit numbers from ground truth.

### Recommended Next Steps for Operator Review
1. To achieve full 5-digit pair extraction, configure shadow ingestion to pull from pristine raw full-resolution recordings (`_raw_record_loop`) before the 768w review-clip downscaling step, OR
2. Formally approve destination-name-first matching (`KATRA` $\rightarrow$ Timetable Matcher $\rightarrow$ Sarvodaya/Swaraj candidate set) for 768w MP4 passages.
