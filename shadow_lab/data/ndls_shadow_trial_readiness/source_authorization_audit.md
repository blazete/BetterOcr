# Source Authorization Audit — Hardened Offline-Replay Invariants

**Component:** `shadow_lab/src/shadow_identity/shadow_ocr/worker.py` (`ShadowWorker`)  
**Audit Date:** 2026-10-26  
**Status:** `VERIFIED_SECURE`  

---

## 1. Problem Statement & Threat Model

In previous revisions, the offline replay configuration could accept caller-supplied `allowed_roots` without enforcing that those roots were strictly contained within a designated, trusted operator-controlled replay workspace root. This left an exposure where arbitrary directories on the host filesystem could be designated as replay intake paths, potentially bypassing production finalization rules.

---

## 2. Implemented Security Controls

### Rule 1 — Explicit Trusted Replay Workspace Root
An operator-controlled workspace root must be explicitly configured via `operator_replay_workspace_root` or `offline_replay_config["trusted_workspace_root"]`.

### Rule 2 — Replay Mode Must Be Explicitly Enabled
Offline replay is strictly gated by `offline_replay.enabled == true`. By default, offline replay is disabled (`enabled: false`), in which case unmarked recordings are immediately rejected.

### Rule 3 — Strict Canonicalization via `Path.resolve(strict=True)`
All workspace roots, allowed intake roots, and candidate file paths are resolved to their strict canonical absolute paths on disk:
- Missing paths raise `ValueError` during initialization.
- Non-existent files or directories fail validation immediately.

### Rule 4 — Strict Containment via `Path.is_relative_to()`
Every entry in `allowed_roots` must satisfy:
```python
canonical_root.is_relative_to(self.operator_replay_workspace_root)
```
If an operator attempts to configure an allowed root outside the trusted workspace root, initialization raises `ValueError` immediately.

### Rule 5 — Rejection of Sibling Paths (No Textual Prefix Sharing)
Paths sharing a textual prefix (e.g. `/workspace/approved_sibling` vs `/workspace/approved`) are strictly rejected. `is_relative_to` evaluates hierarchical filesystem path segments, preventing textual prefix spoofing.

### Rule 6 — Absolute Symlink Rejection
Intake files that are symbolic links (`path.is_symlink() == True`) are rejected unconditionally at intake discovery and staging, preventing symlink traversal attacks.

### Rule 7 — Mandatory Production Finalization Outside Replay
When `offline_replay.enabled == false`, source discovery strictly requires authoritative finalization markers (`.done` files or metadata sidecars with `finalized: true` or `status: "FINALIZED"`).

### Rule 8 — Elimination of Substring Checks
Arbitrary substring checks (such as `"shadow_lab" in path` or `"/tmp" in path`) are completely eliminated from the codebase. All boundary decisions use pure, canonical path predicates.

---

## 3. Regression Test Coverage

The following tests in [`shadow_lab/tests/unit/test_v1_replay_authorization_hardening.py`](file:///Users/shreyanshmalviya/Desktop/BetterOcr/shadow_lab/tests/unit/test_v1_replay_authorization_hardening.py) formally verify these invariants:

1. `test_replay_enabled_missing_workspace_root_rejected`: Confirms `ValueError` when workspace root is omitted.
2. `test_replay_enabled_empty_workspace_root_rejected`: Confirms `ValueError` on empty string or whitespace workspace root.
3. `test_replay_enabled_nonexistent_workspace_root_rejected`: Confirms `ValueError` on non-existent directory.
4. `test_replay_allowed_root_outside_workspace_rejected`: Confirms `ValueError` when allowed root escapes workspace.
5. `test_replay_sibling_directory_textual_prefix_rejected`: Confirms rejection of sibling directories sharing textual prefix.
6. `test_replay_symlink_rejected`: Confirms symlink intake files are rejected.
7. `test_replay_disabled_by_default_rejects`: Confirms unmarked files are rejected when replay is disabled.
8. `test_replay_valid_trusted_root_allowed`: Confirms legitimate files inside trusted workspace root are accepted.
