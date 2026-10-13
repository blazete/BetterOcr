# Shadow Lab Operational Runbook

This runbook guides operators and engineers on running preflight checks, replaying passage fixtures, validating schedules, and generating side-by-side benchmark reports without modifying production ARDS.

---

## 1. Environment & Setup

Ensure the virtual environment is activated and `PYTHONPATH` includes `shadow_lab/src`:

```bash
cd /Users/shreyanshmalviya/Desktop/BetterOcr
export PYTHONPATH=shadow_lab/src
PYTHON="train-identity-lab/.venv/bin/python"
```

---

## 2. Command Line Workflows

### 2.1 Preflight Health Check
Validates that workspace directories exist, configurations adhere to JSON schema, production write guards are active (`production_write_allowed: false`), network access is disabled (`allow_network: false`), and schedule checksums match references.

```bash
# Check all registered sites
$PYTHON -m shadow_identity.cli preflight

# Check a specific site
$PYTHON -m shadow_identity.cli preflight --site-id ndls_pf12_13
$PYTHON -m shadow_identity.cli preflight --site-id testx_pf1_2
```

---

### 2.2 Timetable Validation
Verifies decimal time parsing (`5.10` $\rightarrow$ `05:10`), leading zero preservation (`02563`), operating days, and flags any missing arrival/departure times.

```bash
$PYTHON -m shadow_identity.cli validate-schedule --site-id ndls_pf12_13
$PYTHON -m shadow_identity.cli validate-schedule --site-id testx_pf1_2
```

---

### 2.3 Replaying Passage Fixtures
Replays a passage fixture through the deterministic evaluator and prints the complete evidence trace:

```bash
# 1. Daylight passage with bidirectional number board (Video 8 / Malwa Express)
$PYTHON -m shadow_identity.cli replay --site-id ndls_pf12_13 --fixtures shadow_lab/tests/fixtures/ndls_day.json

# 2. Night passage with non-numeric board (Video 6 / Taj Express)
$PYTHON -m shadow_identity.cli replay --site-id ndls_pf12_13 --fixtures shadow_lab/tests/fixtures/ndls_name_only.json

# 3. Synthetic station at identical clock time (TESTX station isolation)
$PYTHON -m shadow_identity.cli replay --site-id testx_pf1_2 --fixtures shadow_lab/tests/fixtures/siteB_same_time.json
```

---

### 2.4 Verifying Production Integrity (Zero-Write Audit)
Cryptographically verifies that all 456 files in `ARDS-main/` remain identical to baseline SHA256 hashes:

```bash
$PYTHON -m shadow_identity.cli verify-no-writes --baseline-manifest shadow_lab/workspace/fixtures/original_hashes.json
# Or run the script directly:
$PYTHON shadow_lab/scripts/verify_originals_unchanged.py
```

---

### 2.5 Generating Weekly Benchmark Comparison
Replays all benchmark fixtures, joins against production decisions and independent ground truth, and writes:
- `shadow_lab/workspace/reports/week_comparison.csv`
- `shadow_lab/workspace/reports/week_mismatches.csv`
- `shadow_lab/workspace/reports/week_summary.md`

```bash
$PYTHON -m shadow_identity.cli compare --week-start 2026-10-12 --all-sites
```

---

### 2.6 Running the Full Test Suite
```bash
$PYTHON -m pytest shadow_lab/tests/ -v
```
