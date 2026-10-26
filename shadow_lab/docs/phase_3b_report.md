# Phase 3B Report: ARDS Recording Profile, Two-Stage Sampling, and Board Detection Benchmark

**Date:** 2026-10-14  
**Audit Scope:** Verification of actual ARDS recording profile from source code, implementation of two-stage screening vs OCR inference architecture, automatic board proposals generator, 1/2/5 FPS controlled benchmark, entity classification safeguards (`KATRA` as endpoint, `BUFFET CAR` rejection, 4-digit serials separation), and failure analysis.  
**Production Integrity:** 456/456 ARDS files verified bitwise identical (0 modified, 0 missing).  
**Gate Recommendation:** **BLOCKED (Resolution / Board Region Legibility Gate)**

---

## 1. Actual ARDS Recording Profile Verification (from Source Code & File Metadata)

We audited the production ARDS recording subsystem from source code in `ARDS-main/orchestrator/` and verified it against the saved MP4 files:

### A. Source Code Architecture & Downscaling Constant
- **Source Files**:
  - `ARDS-main/orchestrator/video_recorder.py:31-34`:
    ```python
    REVIEW_CLIP_SCALE_W = 768
    REVIEW_CLIP_GOP = 15
    REVIEW_CLIP_QUALITY = 24  # crf (libx264) / cq (nvenc)
    ```
  - `ARDS-main/orchestrator/ocr_pipeline.py:1237-1245`: Initializes `VideoRecorder` for review clips with `scale_w=REVIEW_CLIP_SCALE_W (768)`.
  - `ARDS-main/orchestrator/video_recorder.py:122-135`: In `FfmpegWriter.open()`, the frame is scaled down in OpenCV *before* piping to ffmpeg:
    `oh = int(round(h * self._scale_w / float(w)))` $\rightarrow$ $768 \times 576$ (preserving 4:3 aspect ratio).
- **Purpose**: Annotated review clips are downscaled to 768px width so human operators can seek/scrub natively in web browsers without proxy pipelines (`REVIEW_CLIP_GOP = 15` keyframes).

### B. Representative Saved MP4 Metadata Verification
Extracted using OpenCV container probe on the actual video files:
- **`coach_id11.mp4`**:
  - Resolution: **768x576** (1.33:1 aspect ratio)
  - Native FPS: 25.00
  - Total Frames: 3717 (Duration: 148.68s)
  - File Size: 37.24 MB
  - Codec FourCC: `h264` (avc1)
  - Average Bitrate: 2,101.4 kbps
- **`coach_id12.mp4`**:
  - Resolution: **768x576** (1.33:1 aspect ratio)
  - Native FPS: 25.00
  - Total Frames: 3868 (Duration: 154.72s)
  - File Size: 43.01 MB
  - Codec FourCC: `h264` (avc1)
  - Average Bitrate: 2,332.0 kbps

### C. Physical Stencil Scale Impact
- Coach Length: $\approx 24.0\text{ meters}$.
- Frame Width: $768\text{ pixels} \implies 1\text{ pixel} \approx 3.125\text{ cm}$.
- Yellow Destination Board: $\approx 1.5\text{m} \times 0.2\text{m} \implies \approx 48\text{px} \times 6.4\text{px}$.
- Bold Station Name (`KATRA`): Height $\approx 15\text{ cm} \implies \approx 18\text{px}$ to $24\text{px}$ tall (clearly legible).
- Secondary 5-Digit Stencil Numbers (`12477 / 12478`): Height $\approx 6\text{ cm} \implies \approx 2.5\text{px}$ to $8\text{px}$ tall. Under 25 FPS motion blur, these secondary numbers smudge into illegible strokes.

---

## 2. Two-Stage Sampling Architecture (Screening vs Inference)

Implemented in `shadow_lab/src/shadow_identity/shadow_ocr/two_stage_sampler.py`:
- **Stage 1 (Lightweight Screening)**: Decodes frames at 1, 2, or 5 FPS; computes Laplacian variance on the board strip; checks exposure (brightness and contrast); runs real-time yellow board detection. Throughput: **~40 frames/sec** on CPU.
- **Stage 2 (Targeted OCR Inference)**: Prioritizes frames with detected board proposals, ranked by sharpness, and strictly enforces the configurable budget (`max_ocr_budget = 15`). Throughput: **~5 frames/sec** on CPU.

---

## 3. Automatic Board ROI Proposals Evaluation

- **Method**: HSV color segmentation ($H \in [14, 36], S \in [55, 255], V \in [55, 255]$) targeting Indian Railway yellow destination boards, with rectangular morphological closing ($25 \times 7$) and aspect filter ($\text{aspect} \ge 2.2$).
- **Recall**: Successfully proposed destination board bounding boxes on Frames 2210, 2220, 2230, 2250, 2260 (e.g. Frame 2250: `[x=506, y=186, w=262, h=68]`, aspect 3.85).
- **False Proposals**: 2 false proposals on yellow-painted emergency door grab handles.
- **Processing Time**: **< 1.8 ms** per frame.
- **Expanded Safety Margin**: Tight bounding boxes cut off stencil text borders. An expanded margin (`margin_x=30, margin_y=25`) was implemented to preserve full text context.

---

## 4. Controlled Benchmark at 1, 2, and 5 FPS (Video 11)

| Strategy | Screened Frames | Retained Sharp Candidates | OCR Calls | Wall-Clock Time | Board Sightings | Recovered Entities |
|---|---|---|---|---|---|---|
| **1.0 FPS** | 149 frames | 144 frames | 15 calls | 17.59 s | 1 board sighting | `KATRA` (conf 0.90) |
| **2.0 FPS** | 310 frames | 289 frames | 15 calls | 15.16 s | 2 board sightings | `KATRA` (conf 0.90) |
| **5.0 FPS** | 744 frames | 718 frames | 15 calls | 18.95 s | 4 board sightings | `KATRA` (conf 0.98) |

---

## 5. Preprocessing Variant Verification (P1 vs P2 vs P3)

Tested on Frame 2250 board passage:
- **P1 (Primary Color 2x Bicubic)**:
  - Text: `KATRA` (conf 0.90), `HAPAC` (conf 0.68), `LADRIEGK` (conf 0.50) in 185 ms.
- **P2 (Gamma LUT $\gamma=0.7$, midtone darkening)**:
  - Text: `FUIEE` (conf 0.43), `LA` (conf 0.32) in 192 ms.
  - Darkening midtones degraded yellow text legibility against dark red coaches.
- **P3 (CLAHE 2x Bicubic BGR)**:
  - Text: **`KATRA` (conf 0.98)**, `EAPA C57` (conf 0.69) in 210 ms.
  - Superior contrast enhancement on reflective metal surfaces.

---

## 6. Entity Classification Safeguards Verification

- **`KATRA`**: Correctly categorized as **`ENDPOINT`**, NOT a train name. In `text_entities.py`, `KATRA` is listed in `KNOWN_ENDPOINTS`, preventing the system from mistaking the destination station for the service name.
- **`BUFFET CAR` / `AC THREE TIER`**: Correctly categorized as **`COACH_DESIGNATION`**, completely excluded from service-number candidates.
- **Coach Stencils (`2249`, `1630`)**: Correctly categorized as **`COACH_SERIAL`**, prevented from contaminating 5-digit service numbers.

---

## 7. Test Suite & Integrity Status

- **Unit & Regression Suite**: **123 passed in 3.87s** (up from 118 baseline).
- **Production ARDS Integrity**: All 456 production file SHA-256 hashes matched (`Mismatches: 0, Missing: 0`).
- **Safety**: `shadow_ocr_enabled = False` by default; zero production modification.

---

## 8. Gate Recommendation: BLOCKED

### Gate Assessment
- **Status**: **BLOCKED (End-to-End Visual 5-Digit Number Extraction from Downscaled 768w Review MP4s)**
- **Evidence**:
  1. ARDS review MP4 recordings are downscaled to 768px width by design (`REVIEW_CLIP_SCALE_W = 768`).
  2. While major station names (`KATRA`) and coach classes (`BUFFET CAR`, `AC THREE TIER`) are cleanly extracted, 5-digit train numbers on boards are insufficiently resolved at 768w under motion blur.
  3. Strict adherence to our core directive prevents us from relaxing candidate thresholds or fabricating 5-digit numbers from ground truth.

### Recommended Next Steps for Operator Review
1. To achieve full 5-digit pair extraction, configure shadow ingestion to pull from pristine raw full-resolution recordings (`_raw_record_loop`) before the 768w review-clip downscaling step, OR
2. Formally approve destination-name-first matching (`KATRA` $\rightarrow$ Timetable Matcher $\rightarrow$ Sarvodaya/Swaraj candidate set) for 768w MP4 passages.
