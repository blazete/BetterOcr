# Multi-Site Train Identity Shadow Lab — Test Plan

## 1. Overview & Verification Strategy

The test suite validates the shadow evaluator offline without network requests, production database access, or side effects on ARDS-main.

### Automated Test Matrix (61 Automated Tests)
1. **Config & Site Isolation (Tests 1–10):** `shadow_lab/tests/unit/test_site_isolation.py`
   - Mandatory `site_id` enforcement.
   - Rejection of unknown sites with strict prohibition of silent NDLS fallback.
   - Complete partition between Station A and Station B schedules/NTES snapshots.
   - Island platform ambiguity preservation.
2. **Timetable Import & Normalization (Tests 11–20):** `shadow_lab/tests/unit/test_timetable_importer.py`
   - Railway decimal time parsing (`5.10` $\rightarrow$ `05:10`, `6.05` $\rightarrow$ `06:05`, `22.0` $\rightarrow$ `22:00`).
   - Leading zero preservation (`02563`, `02570`).
   - Day rules (`Daily`, `Except Tuesday`, `Only Sunday`).
   - Rake-linking extraction without mutating active service number.
3. **OCR & Coach Consensus (Tests 21–30):** `shadow_lab/tests/unit/test_coach_consensus.py`
   - Rejection of coach serials (`217152/C`) and berth stencils (`S7`, `43-48`).
   - 50 frames of 1 coach = 1 vote.
   - Strict $\ge 3$ distinct coach threshold enforcement.
   - Video 8 historical discrepancy resolution (`PAIR_PROVISIONAL`).
4. **NTES & Timing Causality (Tests 31–40):** `shadow_lab/tests/unit/test_ntes_timing.py`
   - Strict causality: zero leakage of future NTES snapshots into T0/T1 decisions.
   - `+180s` timer measured strictly from `passage_end`.
   - Post-check train absence is not treated as negative proof.
   - Distinction between fresh fetch, cached snapshot, and stale fallback.
5. **Decision Logic & Abstention (Tests 41–50):** `shadow_lab/tests/unit/test_decision_logic.py`
   - Number-first disambiguation.
   - Name-first 4/6-number board resolution.
   - Non-numeric fallback for unnumbered boards.
   - Visual-NTES conflict detection.
   - Safe abstention on co-present trains without guessing.
6. **Isolation & Zero Production Writes (Tests 51–60):** `shadow_lab/tests/isolation/test_no_production_writes.py`
   - Cryptographic verification of all 456 production files against baseline SHA256 hashes.
   - Monkeypatched open calls raising on any attempted write to `ARDS-main/`.
   - Deterministic replay idempotency.
   - Separation of agreement from ground truth (`AGREE_UNVERIFIED`).
7. **Two-Station Integration Replay:** `shadow_lab/tests/integration/test_two_station_replay.py`
   - Simultaneous passages at NDLS PF12/13 and TESTX PF1/2 at identical clock times (22:00) with zero cross-site leakage.

---

## 2. Execution Command

```bash
# Run complete test suite in offline mode
PYTHONPATH=shadow_lab/src train-identity-lab/.venv/bin/python -m pytest shadow_lab/tests/ -v
```
