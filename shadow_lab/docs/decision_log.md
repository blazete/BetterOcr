# ShadowOCR Decision Log

## 2026-10-14 - DL-001: MP4-First Input Architecture
- **Decision**: Shadow OCR will ingest exclusively completed, copied MP4 passage recordings staged in shadow-owned storage.
- **Alternatives Rejected**: Direct RTSP streaming, live-frame consumer thread, modifying ARDS VideoRecorder callback.
- **Rationale**: Isolates shadow processing entirely from live capture threads; prevents CPU/GPU starvation on production ARDS; eliminates camera bridge risks.

## 2026-10-14 - DL-002: Ground Truth Strict Separation
- **Decision**: Ground truth files/dictionaries will never be imported or accessible to the inference pipeline or `coach_pair_resolver`. Ground truth is read solely by downstream evaluation benchmarks.
- **Alternatives Rejected**: Reusing `run_final_validation_batch.py` candidate filtering (`if c in gt_set`).
- **Rationale**: Real deployed installations cannot leak future or external ground truth into consensus ranking.

## 2026-10-14 - DL-003: Strict Pair Confirmation Thresholds
- **Decision**: `PAIR_CONFIRMED` strictly requires that BOTH candidate service numbers have $\ge 3$ distinct verified physical coaches. If one candidate has $\ge 3$ coaches and the second has fewer, status is `PAIR_PROVISIONAL` (with candidate marked `SINGLE_CANDIDATE`).
- **Alternatives Rejected**: Relaxing to 1 supporting coach for the second number.
- **Rationale**: Preserves truth in measurement; avoids false pair claims under asymmetric glare or dirty coach boards.
