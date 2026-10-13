# Multi-Site Train Identity Shadow Lab Architecture

## 1. System Overview & Core Philosophy

The Multi-Site Train Identity Shadow Lab operates as a strictly additive, read-only shadow evaluation system running concurrently with production Automated Rolling-stock Defect Detection Systems (ARDS / MVIS).

```text
CAMERA PASSAGE (t0, site_id, camera_id, possible platforms)
             |
             +--> CURRENT ARDS PATH (untouched production)
             |      coach OCR + schedule + NTES + roster + resolver
             |      -> existing committed train assignment (e.g. INSP_XXX)
             |
             +--> SEPARATE SHADOW PATH (isolated shadow_lab)
                    service-number OCR + train-name OCR + endpoints
                    -> distinct physical coach consensus (>= 3 coaches)
                    -> site-specific timetable candidate matching
                    -> time-ordered NTES snapshot evidence (T0 -> T1 -> T2)
                    -> shadow-only identity / safe abstention / reason codes
                                  |
                            SIDE-BY-SIDE EVALUATION
                                  |
                        INDEPENDENT GROUND TRUTH
```

---

## 2. Multi-Station Hierarchy of Identity

No station configuration, timetable, or platform universe is ever shared across sites:

```text
installation_id (e.g. ndls_ards_01, testx_ards_01)
  └── site_id (e.g. ndls_pf12_13, testx_pf1_2)
        ├── station.code (e.g. NDLS, TESTX)
        ├── station.name (e.g. NEW DELHI)
        ├── possible_platforms (e.g. ["12", "13"])
        ├── schedule (site-specific versioned CSV with SHA256 checksum)
        └── ntes (replay_only / shared snapshot archive)
```

### Strict Non-Negotiable Rules
1. **Mandatory `--site-id`:** Every CLI invocation and internal query requires an explicit `site_id`.
2. **Zero Silent Fallback:** If an unrecognized site or unconfigured station is requested, execution immediately aborts with `ConfigError`. It is strictly prohibited to fall back silently to New Delhi (`NDLS`).
3. **Island Platform Ambiguity:** Side-view cameras monitoring an island platform (e.g., PF 12/13) cannot observe the physical track occupancy. The system explicitly records `possible_platforms: ["12", "13"]` and `observed_platform: null`.
4. **Zero Production Writes:** The shadow lab operates strictly inside `shadow_lab/workspace/`. All files under `ARDS-main/` remain read-only and verified by SHA256 baseline hashes.

---

## 3. Decision Pipeline & Causality Stages

Each train passage progresses through three evaluation stages:

- **Stage T0 (Visual + Schedule):** Immediate estimation available at physical passage completion (`passage_end`). Uses multi-coach visual consensus and site timetable.
- **Stage T1 (Pre-NTES):** Incorporates the nearest recorded NTES snapshot fetched at or before `passage_end`. Strictly causal; future snapshots are prohibited.
- **Stage T2 (Post-NTES +180s):** Evaluates follow-up verification using an NTES snapshot fetched approximately 3 minutes (`+180s`) after `passage_end`. Discloses the added latency explicitly.

---

## 4. Output Artifacts & Taxonomy

Evaluations generate:
- `week_comparison.csv`: All 34 standard columns including T0, T1, T2 status, reason codes, production decisions, and ground truth.
- `week_mismatches.csv`: Filtered queue containing all passages where production and shadow disagreed or review is required.
- `week_summary.md`: Markdown report detailing coverage, accuracy, and latency breakdown.
