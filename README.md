# BetterOcr

Multi-Site Train Identity Shadow Lab — an offline, read-only shadow evaluation system for railway train identification using OCR, timetable matching, and NTES verification.

## Quick Start

```bash
# Install dependencies
pip install pyyaml

# Run all 95 automated tests (unit, integration, isolation, delayed-worker bridge)
PYTHONPATH=shadow_lab/src python -m pytest shadow_lab/tests/ -v

# Run preflight health check (verifies ARDS NTES scraper availability and zero-write invariants)
PYTHONPATH=shadow_lab/src python -m shadow_identity.cli preflight

# Run delayed NTES verification worker loop (processes only due tasks at +180s)
PYTHONPATH=shadow_lab/src python -m shadow_identity.cli run-delayed-worker --service --poll-interval 5.0

# Replay a benchmark passage
PYTHONPATH=shadow_lab/src python -m shadow_identity.cli replay \
  --site-id ndls_pf12_13 \
  --fixtures shadow_lab/tests/fixtures/ndls_day.json
```

See [`shadow_lab/README.md`](shadow_lab/README.md) for full architecture and documentation.

