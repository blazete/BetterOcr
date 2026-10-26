# NDLS Supervised Shadow-Trial Deployment Configuration

## 1. Overview & Policy Constraints

This document defines the proposed configuration baseline for a future supervised NDLS (New Delhi Railway Station) Platform 12/13 shadow trial. It is a policy reference; the MP4 worker CLI does not load this document. It reads `shadow_defaults.yaml` and the selected site YAML for launch safety checks.

**CRITICAL SAFETY POLICIES:**
1. **Air-Gapped Default:** All deployment flags are set to `false`/`disabled` by default. Under no circumstances should the system connect to live production railway systems without explicit operator authorization.
2. **Read-Only Ingestion:** The input recording directory must be mounted as read-only.
3. **Shadow-Owned Outputs:** All temporary video copies, state databases, raw observations, and decision JSONs are stored exclusively inside designated shadow directories.
4. **Frozen Budget:** OCR sampling and inference budget is strictly capped at $\le 25$ calls per passage.

---

## 2. Policy Targets and Current Enforcement

`shadow_lab/config/shadow_defaults.yaml` and the NDLS site YAML contain policy values, but the MP4 worker does not load them. The current worker CLI has no `--config` option and does not gate processing on `live_shadow`.

| Parameter | Policy target | Current worker behavior |
| :--- | :--- | :--- |
| Global `shadow_defaults.live_shadow` | `false` | Required to be explicitly `true` before the CLI will construct or run the worker; shipped value remains `false`. |
| Global and site `allow_network` | `false` | CLI refuses startup if either is not explicitly `false`; the site `shadow.live_shadow` must also remain false. |
| `offline_replay.enabled` | `false` | CLI refuses startup if enabled; worker also requires finalization signals. |
| `require_finalization_signal` | `true` | Enforced by default; not configurable through CLI. |
| `screening_fps` | `5.0` | Worker default; not configurable through CLI. |
| `max_ocr_budget` | `25` | Worker default and enforced; not configurable through CLI. |
| `max_concurrency` | `1` | CLI supports `--concurrency`; default is `2`. |
| `cleanup_retention_sec` | `3600.0` | Successful copy default is 1 hour; failed copy default is 24 hours. Neither is configurable through CLI. |
| `size_stability_interval_sec` | `2.0` | Worker default is 1 second; not configurable through CLI. |

**Readiness gate:** the CLI is blocked by the shipped global `live_shadow: false`. Do not change it or run against railway-mounted recordings without explicit operator authorization. When authorized, keep site-level `shadow.live_shadow`, networking, offline replay, and production writes false. The daily report is a separate report-only command and does not start inference.

---

## 3. Directory Layout & Mount Boundaries

For a supervised trial at NDLS Platform 12/13, the operator configures dedicated paths as follows:

```
/opt/railway/shadow_trial/
├── workspace/                       <-- operator_replay_workspace_root (trusted)
│   ├── inbox/                       <-- inbox_dir (READ-ONLY mount from camera storage)
│   │   ├── pas_ndls_20261026_01.mp4
│   │   └── pas_ndls_20261026_01.done
│   └── historical_archive/         <-- (Optional) offline replay recordings
├── shadow_storage/                  <-- storage_dir (READ-WRITE, shadow-owned)
│   ├── temp_mp4/                    <-- Temporary staged copies (TTL governed)
│   └── worker_jobs.json             <-- Immutable state machine ledger
└── shadow_output/                   <-- output_dir (READ-WRITE, shadow-owned)
    ├── ndls_pf12_13_side_a_..._decision.json
    └── ndls_pf12_13_side_a_..._raw.jsonl
```

### Path Authorization Constraints

When `offline_replay.enabled == true`:
- `operator_replay_workspace_root` must be set to `/opt/railway/shadow_trial/workspace` (must exist, must be absolute).
- `allowed_roots` must contain `["/opt/railway/shadow_trial/workspace/inbox", "/opt/railway/shadow_trial/workspace/historical_archive"]`.
- Any candidate file whose resolved canonical path falls outside `operator_replay_workspace_root` is immediately rejected.
- Any symlink pointing inside or outside the workspace is rejected.

---

## 4. Hardware & Runtime Prerequisites

- **Host OS:** Linux (Ubuntu 22.04 LTS / Debian 12 / RHEL 9) or macOS (Darwin 23+).
- **Python Runtime:** Python 3.10, 3.11, or 3.12 with virtual environment.
- **Compute:** 4 CPU cores minimum (8 cores recommended). GPU/MPS acceleration optional; CPU inference executes within 25–35 seconds per passage.
- **Storage:** 50 GB free NVMe disk space allocated for `shadow_storage/` and `shadow_output/`.
- **Memory:** 8 GB RAM minimum (peak memory footprint per worker thread is $\sim 1.2$ GB).

---

## 5. Pre-Deployment Validation Checklist

Before a future launch, after the software blockers above are resolved:

- [ ] Confirm `live_shadow: false` and `allow_network: false` in active config.
- [ ] Confirm `inbox_dir` is mounted `ro` (read-only) at the OS/filesystem level.
- [ ] Confirm `storage_dir` and `output_dir` are strictly on a separate partition or isolated directory tree outside production ARDS.
- [ ] Run `python shadow_lab/scripts/verify_originals_unchanged.py` to confirm 456 ARDS files are untouched.
- [ ] Run synthetic end-to-end dry run (`test_v1_task4_e2e_synthetic_mp4.py`) in isolated scratch folder.
