# Supervised NDLS Shadow Trial — Operator Runbook

## 1. Scope & Operator Responsibility

This runbook guides railway system operators through executing a **supervised, read-only shadow trial** of BetterOcr V1 at New Delhi Railway Station (NDLS) Platform 12/13.

### Core Operating Principles
- **Air-Gapped & Read-Only:** The shadow system does not interact with live signaling, ARDS databases, or production train management networks.
- **Human-in-the-Loop Supervision:** The worker must run under direct operator monitoring during designated observation windows.
- **Strict Evidence Preservation:** Never delete input recordings or raw observation outputs during incident investigation.

---

## 2. Pre-Flight Verification & System Validation

Before launching any worker cycles, execute the pre-flight verification checklist:

### Step 2.1: Verify Cryptographic Integrity of Production Code
Run the ARDS file integrity checker from the repository root:
```bash
python shadow_lab/scripts/verify_originals_unchanged.py
```
**Expected Output:**
```
[PASS] All 456 original ARDS files verified unchanged (zero modifications).
```
*If any file fails hash validation, DO NOT PROCEED. Immediately report code tampering.*

### Step 2.2: Confirm Read-Only Filesystem Mounts
Ensure the recording intake directory is mounted read-only at the OS layer:
```bash
# Verify mount flags include 'ro'
mount | grep /path/to/recordings_inbox

# Test write rejection (must fail with Permission Denied)
touch /path/to/recordings_inbox/.test_write 2>/dev/null || echo "PASS: Read-only confirmed"
```

### Step 2.3: Verify Output & Storage Isolation
Verify that `shadow_storage` and `shadow_output` are writeable and have the operator-approved capacity (50 GB is the current planning target):
```bash
df -h /path/to/shadow_storage /path/to/shadow_output
```

---

## 3. Current Worker Configuration Support

The MP4 worker does **not** load a worker-specific YAML configuration. Its supported arguments are `--watch-dir`, `--storage-dir`, `--output-dir`, `--site-id`, `--camera-id`, `--timetable`, `--ntes-dir`, `--ground-truth-dir`, `--concurrency`, `--poll-interval`, and `--once` (see `python -m shadow_identity.shadow_ocr.worker --help`). It does not support `--config`, `--single-cycle`, or `--poll-interval-sec`. For the launch safety check, it reads the existing global defaults and selected site YAML.

The worker CLI checks the existing global `shadow_defaults.live_shadow` setting before constructing the worker. It refuses to start unless that value is explicitly `true`; the shipped value remains `false`. It also requires global networking and offline replay to be false, plus site-level `shadow.live_shadow`, `ntes.allow_network`, and `production_write_allowed` to be false. `--once` is a processing cycle, **not** a dry run. Do not change the global setting or point the worker at railway recordings without explicit operator authorization.

The required policy values remain: site `ndls_pf12_13`, camera `side_a`, concurrency `1`, OCR budget at most `25`, finalized source required, and shadow-owned storage/output. The CLI currently enforces finalization and the OCR limit but does not expose all settings as arguments or load them from a configuration file.

## 4. Launch Status

There is no approved NDLS launch command in this checkout. The global launch gate remains false, and the worker must not be run against railway-mounted storage. For software verification, use only the isolated synthetic end-to-end test. The report-only command in section 7 can be used on persisted local test data without enabling the worker.

---

## 5. Backlog & Resource Monitoring

These commands describe local state for a future authorized trial. They do not mean the worker is ready to run against NDLS.

### 5.1 Backlog Monitoring
Inspect pending vs completed jobs in `shadow_storage/worker_jobs.json`:
```bash
python -c '
import json
with open("/opt/railway/shadow_trial/shadow_storage/worker_jobs.json") as f:
    jobs = json.load(f)
statuses = {}
for j in jobs.values():
    s = j.get("status", "UNKNOWN")
    statuses[s] = statuses.get(s, 0) + 1
print("Current Worker Status:", statuses)
'
```

### 5.2 Disk Usage Monitoring
```bash
# Check size of temporary staged video buffer
du -sh /opt/railway/shadow_trial/shadow_storage/temp_mp4

# Check size of output evidence
du -sh /opt/railway/shadow_trial/shadow_output
```

### 5.3 Pilot Measurement Targets
These are proposed pilot targets, not values verified by the current automated suite:
1. Average wall-clock processing time per train passage is **$\le 35$ seconds**.
2. Staged MP4 copies in `temp_mp4` are deleted after 1 hour (retained count should match recent arrival rate).
3. Memory consumption per worker process remains stable under **1.5 GB RSS**.

---

## 6. Failure Inspection & Evidence Preservation

If a job fails or an anomaly is flagged:

### 6.1 Inspect Job Error State
Query the exact error reason from `worker_jobs.json`:
```bash
jq '.[] | select(.status=="FAILED") | {job_id, error_message, discovered_sha256}' \
    /opt/railway/shadow_trial/shadow_storage/worker_jobs.json
```

**Common Safe Rejection Codes:**
- `SOURCE_CHANGED_SINCE_DISCOVERY`: File was modified by recording writer while staging. The job was rejected cleanly before OCR inference.
- `SOURCE_SIZE_CHANGED_SINCE_DISCOVERY`: Source or staged byte count differs from discovery.
- `SOURCE_NOT_FINALIZED_AT_RESTAGING`: Finalization signal is missing when a staged copy must be recreated.
- `SOURCE_AUTHORIZATION_FAILED` / `STAGED_SOURCE_UNSAFE`: Source is outside the configured inbox or a staged input is unsafe.

Successful staged copies are retained for 60 minutes. Permanently failed staged copies are retained for 24 hours. These are current worker defaults, not YAML-loaded settings.

### 6.2 Preserving Evidence for Audit
To preserve raw OCR evidence and decision telemetry for engineering analysis without interrupting worker execution:
```bash
# Package outputs into an audit archive
tar -czf /tmp/shadow_trial_audit_$(date +%Y%m%d_%H%M%S).tar.gz \
    /opt/railway/shadow_trial/shadow_output/*_decision.json \
    /opt/railway/shadow_trial/shadow_output/*_raw.jsonl
```
*Note: Never modify or delete input MP4s.*

---

## 7. Daily Reporting Status

The existing `daily-report` command reads passage bundles under `shadow_lab/data/passages/`; it does not read continuous MP4 worker decisions. Use the separate report-only command after worker output has been persisted:

```bash
PYTHONPATH=shadow_lab/src python shadow_lab/scripts/generate_worker_daily_report.py \
    --date 2026-10-26 \
    --worker-state /opt/railway/shadow_trial/shadow_storage/worker_jobs.json \
    --ground-truth-dir /opt/railway/shadow_trial/ground_truth \
    --reports-dir /opt/railway/shadow_trial/shadow_reports \
    --site-id ndls_pf12_13
```

This command only reads worker decisions and the separate evaluator-owned ground-truth store; it does not run OCR. It groups passages by recorded local passage time, falling back to worker discovery/completion time only when passage time is missing. Ground truth is matched by exact source SHA-256. The worker does not invoke this command automatically.

## 8. Controlled Shutdown

To terminate the supervised shadow trial and return the system to the completely disabled state:

### Step 7.1: Send Graceful Stop Signal
If running in the foreground, press `Ctrl+C`.
If running as a background service:
```bash
kill -SIGINT <worker_pid>
```
The worker will finish any in-flight passage OCR inference, write the decision JSON, persist the job state, and exit cleanly.

### Step 7.2: Verify State Cleanliness
Inspect staged video lifecycle records and allow the normal retention reaper to remove expired files. Do not advance the cleanup clock manually; doing so can delete staged evidence before its configured retention expires.
```bash
PYTHONPATH=shadow_lab/src python -c '
from shadow_identity.shadow_ocr.temp_mp4_lifecycle import TempMp4LifecycleManager
mgr = TempMp4LifecycleManager(temp_dir="/opt/railway/shadow_trial/shadow_storage/temp_mp4")
cleaned = mgr.run_cleanup_due()
print(f"Purged {len(cleaned)} expired staged copies.")
'
```

### Step 7.3: Confirm Disabled Policy Values
Confirm the policy files still contain the disabled values below. The current MP4 worker does not load or enforce them:
```yaml
live_shadow: false
allow_network: false
offline_replay:
  enabled: false
```
Stopping the process ends collection. Keep global `shadow_defaults.live_shadow: false`; the CLI then refuses to start. Keep site-level `shadow.live_shadow: false` and all `allow_network` values false so an authorized worker run cannot activate the NTES adapter or network access.
