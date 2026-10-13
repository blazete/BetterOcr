# BetterOcr

Multi-Site Train Identity Shadow Lab — an offline, read-only shadow evaluation system for railway train identification using OCR, timetable matching, and NTES verification.

## Quick Start

```bash
# Install dependencies
pip install pyyaml

# Run all 61 automated tests
PYTHONPATH=shadow_lab/src python -m pytest shadow_lab/tests/ -v

# Run preflight health check
PYTHONPATH=shadow_lab/src python -m shadow_identity.cli preflight

# Replay a benchmark passage
PYTHONPATH=shadow_lab/src python -m shadow_identity.cli replay \
  --site-id ndls_pf12_13 \
  --fixtures shadow_lab/tests/fixtures/ndls_day.json
```

See [`shadow_lab/README.md`](shadow_lab/README.md) for full documentation.
