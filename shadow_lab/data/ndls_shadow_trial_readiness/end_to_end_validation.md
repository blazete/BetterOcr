# End-to-End Validation Report — Synthetic Passage Full Workflow

**Test File:** [`shadow_lab/tests/unit/test_v1_task4_e2e_synthetic_mp4.py`](file:///Users/shreyanshmalviya/Desktop/BetterOcr/shadow_lab/tests/unit/test_v1_task4_e2e_synthetic_mp4.py)  
**Passage ID:** `pas_synthetic_ndls_trial_001`  
**Execution Date:** 2026-10-26  
**Status:** `PASSED` (17.53s)  

---

## 1. Test Setup & Isolation

1. **Authorized Replay Workspace:**
   - Dedicated temporary workspace (`tmp_path / "operator_workspace"`).
   - Inbox root: `workspace / "inbox"`.
   - Allowed roots: `[str(inbox_dir.resolve())]`.
   - Replay workspace root: `workspace.resolve(strict=True)`.
2. **Mock Production ARDS Isolation:**
   - Dedicated mock directory (`tmp_path / "ards_production"`).
   - Pre-run file list recorded.
   - Post-run verification confirmed zero modifications to ARDS tree.
3. **Synthetic Recording Generation:**
   - Filename: `pas_synthetic_ndls_trial_001.mp4`.
   - Content: 20 synthetic video frames (320x240, 5.0 fps) depicting train coach blue livery with a yellow destination board stripe containing `NDLS SPECIAL` and coach serial `243561`.
   - Finalization marker: `pas_synthetic_ndls_trial_001.done`.

---

## 2. Step-by-Step Execution Verification

| Step | Operation | Result | Invariant Verified |
|---|---|---|---|
| **1. Discovery** | Inbox scanned for finalized recordings | Discovered 1 recording (`pas_synthetic_ndls_trial_001.mp4`) | File must be finalized; symlinks rejected |
| **2. Staging** | Copied to shadow storage `temp_mp4` | Staged successfully; `job.status == "STAGED"` | Staged SHA-256 matches discovery SHA-256; Original MP4 untouched |
| **3. Inference** | PaddleOCR PP-OCRv6_medium crop recognition | Completed in 12.5s; 0 budget overrun | Strict inference ceiling: calls <= 25 (actual calls <= 25) |
| **4. Persistence** | Emitted decision JSON and raw JSONL | Decision and raw observation files saved in shadow storage | Schema `shadow_identity_v1`; durable evidence retained |
| **5. Direction Gate** | Direction evidence evaluation | `active_direction_verified: false`, `active_service_number: null` | Direction never assumed without hardware sensor |
| **6. Timestamp Gate** | Wall-clock timestamp check | `passage_start: null`, `passage_end: null` | Missing timestamps remain null; no synthetic times invented |
| **7. Identity Gate** | Train registry and family matching | `identity_status: "UNKNOWN"`, `service_family_candidates: []` | Unknown trains never assigned invented identities |
| **8. Idempotency** | Subsequent worker cycle executed | 0 discovered, 0 processed | Completed jobs are never re-inspected |
| **9. Cleanup** | Temporary MP4 retention expiration (0.1s) | Temporary staged MP4 unlinked; Original MP4 remains in inbox | Zero data loss; source recording preserved |
| **10. Restart Recovery** | Worker restarted from existing storage | Job reconstructed with status `COMPLETED` | Crash recovery restores durable state |

---

## 3. Conclusions

Full end-to-end processing has been verified using actual OCR inference on newly generated synthetic video frames. All isolation, safety, and persistence invariants hold.
