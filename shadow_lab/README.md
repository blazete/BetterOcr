# Multi-Site Train Identity Shadow Lab

**Document Status:** Implementation complete for a separate, additive, read-only shadow evaluation system.  
**Constraint Guarantee:** Zero changes to production ARDS (`ARDS-main/` verified unchanged across all 456 files).

---

## 1. Quick Start

Run the automated test suite and preflight verification:

```bash
# 1. Run all 61 automated unit, integration, and isolation tests
PYTHONPATH=shadow_lab/src train-identity-lab/.venv/bin/python -m pytest shadow_lab/tests/ -v

# 2. Run system preflight health check across all sites
PYTHONPATH=shadow_lab/src train-identity-lab/.venv/bin/python -m shadow_identity.cli preflight

# 3. Verify zero modifications to original ARDS production files
train-identity-lab/.venv/bin/python shadow_lab/scripts/verify_originals_unchanged.py

# 4. Replay benchmark passages
PYTHONPATH=shadow_lab/src train-identity-lab/.venv/bin/python -m shadow_identity.cli replay --site-id ndls_pf12_13 --fixtures shadow_lab/tests/fixtures/ndls_day.json

# 5. Generate weekly side-by-side comparison report
PYTHONPATH=shadow_lab/src train-identity-lab/.venv/bin/python -m shadow_identity.cli compare --week-start 2026-10-12 --all-sites
```

---

## 2. Directory Structure

```text
shadow_lab/
├── README.md                      # This document
├── pyproject.toml                 # Package definition & dependencies
├── config/
│   ├── schema.json                # JSON schema enforcing site isolation & read-only constraints
│   └── shadow_defaults.yaml       # Global shadow defaults (replay_only, allow_network: false)
├── sites/
│   ├── ndls_pf12_13/              # New Delhi PF 12/13 profile
│   │   ├── site.yaml              # Site configuration
│   │   ├── aliases.yaml           # Station aliases and train name synonyms
│   │   └── schedules/
│   │       └── ndls_validated.csv # Validated timetable (SHA256: dd21cdce...)
│   └── testx_pf1_2/               # Synthetic second station (TESTX) for cross-site isolation tests
│       ├── site.yaml
│       ├── aliases.yaml
│       └── schedules/
│           └── testx_validated.csv# Synthetic timetable (SHA256: 707a0bb5...)
├── src/shadow_identity/           # Core library implementation
│   ├── cli.py                     # Unified CLI interface
│   ├── config_loader.py           # SiteRegistry with mandatory site_id & isolation guards
│   ├── models.py                  # Canonical dataclasses & schemas
│   ├── reason_codes.py            # Complete enumeration of explainable reason codes
│   ├── text_normalization.py      # 5-digit extraction & negative-class filtering
│   ├── coach_consensus.py         # Multi-frame aggregator with >=3 distinct coach rule
│   ├── timetable_importer.py      # Decimal time parsing & leading zero preservation
│   ├── timetable_matcher.py       # Site-specific timetable candidate matching
│   ├── candidate_resolver.py      # Pure deterministic decision function (resolve_shadow)
│   ├── ntes_archive.py            # Offline replay engine for recorded NTES JSONL snapshots
│   ├── ocr_adapter.py             # Adapter for historical per-frame OCR CSVs
│   └── comparison.py              # Side-by-side comparison & taxonomy engine
├── tests/
│   ├── unit/                      # 50 unit tests across 5 modules
│   ├── isolation/                 # 10 zero-production-write integrity tests
│   ├── integration/               # 2-station simultaneous replay tests
│   └── fixtures/                  # Benchmark passage fixtures (ndls_day, ndls_name_only, siteB)
├── scripts/
│   ├── preflight_check.py         # Workspace health and config preflight script
│   ├── verify_originals_unchanged.py # Verifies SHA256 hashes of all 456 ARDS-main files
│   ├── import_timetable.py        # Imports timetable CSVs
│   ├── import_ocr_exports.py      # Converts OCR per-frame CSVs into shadow fixtures
│   ├── import_ntes_audits.py       # Loads NTES JSONL audits
│   ├── replay_passages.py         # CLI replay wrapper
│   └── compare_week.py            # Weekly benchmark comparison wrapper
├── docs/
│   ├── current_ards_behavior_inventory.md # Baseline production audit & Video 8 discrepancy resolution
│   ├── architecture.md            # Multi-site hierarchy and causality stages
│   ├── decision_rules.md          # 2-number, 4/6-number, name-only pathways, and reason codes
│   ├── test_plan.md               # 61-test matrix specification
│   └── runbook.md                 # Operator execution runbook
└── workspace/                     # Dedicated writable output directory (gitignored)
    ├── fixtures/                  # Baseline manifest (original_hashes.json)
    ├── reports/                   # week_comparison.csv, week_mismatches.csv, week_summary.md
    ├── evidence/                  # Cropped frames, raw detections, and audit bundles
    └── logs/                      # Run logs
```

---

## 3. Key Findings & Engineering Invariants

1. **Production Read-Only Invariant:** Under no circumstances are files in `ARDS-main/` modified, overwritten, or called with write side-effects. All 456 production files are hashed and verified.
2. **Video 8 Historical Discrepancy Resolved:** Video 8 is strictly classified as `PAIR_PROVISIONAL` because candidate `12919` had only 2 distinct coaches while `12920` had 5. Under the frozen shadow rule, both numbers in a pair must reach $\ge 3$ distinct coaches for `PAIR_CONFIRMED`.
3. **No Silent Fallback:** Every request requires `--site-id`. If an unconfigured site is requested, the system aborts immediately with an explicit error. NDLS is never assumed as a default.
4. **Time Order & Causality:** Pre-passage decisions (T0, T1) never peek at future NTES snapshots. Post-passage (+180s) verification (T2) is tracked separately with explicit latency disclosure.
