# Source Integrity Audit — Discovery-to-Staging Identity Verification

**Component:** `shadow_lab/src/shadow_identity/shadow_ocr/worker.py` (`ShadowWorker`, `WorkerJob`, `DiscoveryRecord`)  
**Audit Date:** 2026-10-26  
**Status:** `VERIFIED_SECURE`  

---

## 1. Problem Statement & Threat Model

In autonomous video ingestion systems, a recording file is first discovered and validated during an inbox scan. A finite time window elapses between discovery and staging into shadow-owned working storage. If an external process or network mount replaces, appends, or truncates the source file during this window, the worker might process bytes that do not match the discovered identity.

The system must guarantee that the file discovered is bit-for-bit the exact file that gets processed.

---

## 2. Implemented Algorithm & Integrity Controls

### Discovery Identity Recording
During discovery (`discover_new_recordings`), an immutable `DiscoveryRecord` is constructed and cached:
```python
@dataclass(frozen=True)
class DiscoveryRecord:
    path: Path
    source_sha256: str
    source_size: int
    discovered_at_epoch: float
    passage_id: str
    site_id: str
    camera_id: str
```
The full SHA-256 hash and byte size are computed and bound to the resolved path.

### Immediate Pre-Staging Revalidation
Immediately prior to copying, `stage_job` re-verifies:
1. The path is not a symbolic link (`mp4_path.is_symlink() == False`).
2. The file is accessible and finalized (`is_file_finalized() == True`).

### Shadow Storage Staging & Comparison Gate
The file is copied exclusively into shadow-owned temporary storage (`storage_dir / "temp_mp4"`). Immediately following copy completion:
```python
staged_sha256 = compute_file_sha256(staged_file)
staged_size = staged_file.stat().st_size

discovered_sha256 = disc_rec.source_sha256
discovered_size = disc_rec.source_size

if staged_sha256 != discovered_sha256:
    staged_file.unlink(missing_ok=True)
    reject_job("SOURCE_CHANGED_SINCE_DISCOVERY")

if staged_size != discovered_size:
    staged_file.unlink(missing_ok=True)
    reject_job("SOURCE_SIZE_CHANGED_SINCE_DISCOVERY")
```

### Safety & Isolation Guarantees
1. **Immediate Purge:** Any staged file that fails the comparison gate is immediately deleted from disk.
2. **Inference Exclusion:** The job status is permanently marked `"FAILED"`. Because `claim_job_for_processing` strictly requires status `"STAGED"` or `"RETRY_WAIT"`, modified files NEVER enter OCR inference.
3. **Immutable Job Metadata:** Both `discovered_sha256` and `discovered_size_bytes` are stored in the durable `WorkerJob` record.
4. **Collision Resistance:** Job keys are generated as:
   ```
   {site_id}_{camera_id}_{passage_id}_{discovered_sha256}
   ```
   If a new recording arrives for the same passage name with a different hash, it generates a distinct, non-colliding `job_key` without overwriting the previous passage decision.
5. **Read-Only Invariant:** Original recording files in `inbox_dir` are never opened for writing or unlinked during cleanup.

---

## 3. Regression Test Coverage

The following tests in [`shadow_lab/tests/unit/test_v1_discovery_staging_integrity.py`](file:///Users/shreyanshmalviya/Desktop/BetterOcr/shadow_lab/tests/unit/test_v1_discovery_staging_integrity.py) verify these behaviors:

1. `test_staging_rejects_source_hash_changed_since_discovery`: Deliberately mutates video content between discovery and staging; confirms job fails with `"SOURCE_CHANGED_SINCE_DISCOVERY"` and processing is blocked.
2. `test_staging_rejects_source_size_changed_since_discovery`: Appends bytes between discovery and staging; confirms job fails with `"SOURCE_SIZE_CHANGED_SINCE_DISCOVERY"`.
3. `test_staging_rejects_symlink_replacement`: Swaps source file with a symlink before staging; confirms immediate rejection.
4. `test_discovery_hash_preserved_in_metadata`: Confirms `discovered_sha256` and `discovered_size_bytes` are stored in `WorkerJob`.
5. `test_different_hash_for_same_passage_distinct_event`: Confirms different content hashes for the same passage name produce distinct job keys and preserve prior decisions without overwriting.
