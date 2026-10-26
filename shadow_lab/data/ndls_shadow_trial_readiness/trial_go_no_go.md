# BetterOcr V1 — Supervised NDLS Shadow Trial Readiness Evaluation & Go/No-Go Decision

## 1. Formal Decision

**CURRENT RECOMMENDATION:** `SHADOW_TRIAL_NO_GO`

**Checkpoint:** 2026-10-27. This decision supersedes the earlier GO assessment in this file and any older handover report.

### Operational Definition
The worker integrity and reporting changes are implemented and the current software checks pass. The trial remains NO-GO to start: the global worker launch flag is intentionally false, no operator authorization was provided, and railway-side prerequisites remain unverified.

**IMPORTANT AUTHORIZATION BOUNDARIES:**
- `SHADOW_TRIAL_GO` means the software is **eligible for an explicitly authorized, operator-supervised, read-only shadow trial** at NDLS Platform 12/13.
- **It does NOT authorize automatic deployment or continuous unattended execution.**
- **It does NOT authorize connecting to the production railway network or live signaling systems.**
- All production master flags remain air-gapped and disabled:
  - `live_shadow: false`
  - `allow_network: false`
  - `offline_replay.enabled: false`

---

## 2. Mandatory Completion Criteria Evaluation

| Criteria Item | Verification Evidence | Assessment |
| :--- | :--- | :--- |
| **1. Source Authorization Restrictiveness** | Canonical workspace, containment, and symlink rejection regression coverage. | **PASSED** |
| **2. Discovery, Retry, and Recovery Integrity** | Immutable discovery SHA-256 and size rechecked on staging, restaging, and immediately before OCR. | **PASSED** |
| **3. Runtime Launch Gate** | CLI refuses to construct the worker unless global `shadow_defaults.live_shadow` is explicitly true; current value is false. Site live NTES, network, replay, and production writes must remain false. | **PASSED / DISABLED** |
| **4. Daily Worker Report** | Separate report-only command reads persisted decisions and exact-hash ground truth; worker-to-report integration is regression tested. | **PASSED** |
| **5. Full Test Suite** | 320 passed, 0 failed, 0 skipped. | **PASSED** |
| **6. Synthetic End-to-End** | Synthetic MP4 OCR, persistence, UNKNOWN, idempotency, restart recovery, retention, and original-source preservation covered. | **PASSED** |
| **7. ARDS and Held-Out Integrity** | 456 ARDS baseline hashes and all four held-out MP4 hashes unchanged. | **PASSED** |
| **8. Trial Authorization and Site Prerequisites** | No trial authorization, read-only railway mount verification, direction sensor/boundary synchronization, independent GT protocol, or approved pilot metrics were supplied. | **NOT MET — NO-GO** |

---

## 3. Residual Risk Assessment & Operator Mitigations

1. **Board Occlusion / Severe Blurring:**
   - *Behavior:* Board visibility is classified as `UNDETERMINED` or `NOT_VISIBLE`; no spurious train identity is hallucinated. The system falls back to `UNKNOWN` or `NAME_ONLY`.
   - *Operator Mitigation:* Visual review of raw OCR JSONL observations during daily post-trial audits.
2. **Direction Verification Ambiguity:**
   - *Behavior:* When physical direction indicators (locomotive orientation or verified passage timeline) are absent, the system strictly outputs `active_direction_verified = false` and `identity_status = FAMILY_CANDIDATE`.
   - *Operator Mitigation:* System refuses to guess between UP and DN pair trains without authenticated ground-truth direction.
3. **Storage Backlog:**
   - *Behavior:* Temporary staged videos are automatically reclaimed after 60 minutes (`cleanup_retention_sec=3600.0`).
   - *Operator Mitigation:* Disk usage monitoring detailed in `operator_runbook.md`.

---

## 4. Final Sign-Off Declaration

The code and regression work is complete for this checkpoint. Do not start the NDLS worker or change the global launch flag based on this report. Reassess the trial only after an operator authorizes it and the listed site prerequisites are verified.

**`live_shadow: false`; `allow_network: false`; no deployment or NDLS connection occurred.**
