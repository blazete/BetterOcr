# NDLS Shadow Trial — Rollback & Emergency Abort Procedure

## 1. Purpose & Guarantees

This document establishes the immediate abort procedure for the NDLS Platform 12/13 BetterOcr shadow trial. 

Because BetterOcr V1 is completely decoupled from ARDS production signaling, camera capture, and train dispatching:
- **Executing a rollback produces ZERO impact on railway operations or camera recording.**
- **No production railway server reboot or service restart is required.**
- **Input recording files remain untouched throughout the rollback.**

---

## 2. Immediate Abort Triggers

An operator must execute an immediate rollback if ANY of the following conditions occur:

1. **Production Directory Access:** Any file access error or unintended write attempt detected outside designated shadow output paths.
2. **CPU / Memory Saturation:** Host CPU utilization exceeds 90% continuously for $> 5$ minutes, or worker memory RSS exceeds 3.0 GB.
3. **Recording Backlog Escalation:** Worker lag behind live passage finalization exceeds 15 minutes.
4. **Filesystem Warning:** Free disk space on the shadow volume drops below 10 GB.
5. **Unauthorized Network Activity:** Any attempted network connection from the worker process (indicated by OS socket alert or firewall log).
6. **Explicit Directive:** Order from the Chief Controller or Lead Signal Engineer to cease trial.

---

## 3. Step-by-Step Emergency Abort Procedure

### Step 1: Immediate Process Termination
Terminate all active ShadowWorker processes immediately:
```bash
# Graceful stop (allow 10 seconds to finish in-flight OCR)
pkill -SIGINT -f "shadow_identity.shadow_ocr.worker"

# Force terminate if worker does not exit within 15 seconds
pkill -SIGKILL -f "shadow_identity.shadow_ocr.worker"
```

### Step 2: Unmount or Revoke Read-Only Ingestion Mounts
If the recording directory was mounted specifically for the trial, unmount it:
```bash
sudo umount -l /opt/railway/shadow_trial/workspace/inbox 2>/dev/null || true
```

### Step 3: Purge Staged Temporary Video Files
Clear any quarantined temporary video slices to immediately reclaim disk space:
```bash
rm -rf /opt/railway/shadow_trial/shadow_storage/temp_mp4/*
```
*Note: This command only deletes shadow copies in `temp_mp4/`. Do NOT delete the `worker_jobs.json` state file or output directories.*

### Step 4: Verify Master Disable State
Ensure that configuration files have `live_shadow` set to `false`:
```bash
sed -i 's/live_shadow: true/live_shadow: false/g' shadow_lab/config/*.yaml 2>/dev/null || true
sed -i 's/allow_network: true/allow_network: false/g' shadow_lab/config/*.yaml 2>/dev/null || true
```

---

## 4. Post-Rollback Auditing & Triage

Following process termination:

1. **Verify ARDS Zero-Touch Invariant:**
   ```bash
   python shadow_lab/scripts/verify_originals_unchanged.py
   ```
   Confirm all 456 production files report `[PASS]`.

2. **Collect Post-Mortem Diagnostic Bundle:**
   ```bash
   tar -czf /tmp/shadow_rollback_bundle_$(date +%Y%m%d_%H%M%S).tar.gz \
       /opt/railway/shadow_trial/shadow_storage/worker_jobs.json \
       /opt/railway/shadow_trial/shadow_output/
   ```

3. **Log Abort Event:**
   Record timestamp, trigger event, operator name, and diagnostic archive location in the operational log.
