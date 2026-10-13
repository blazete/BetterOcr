# Multi-Site Train Identity Shadow Lab — Weekly Comparison Report

**Evaluation Week**: `2026-10-12` to `2026-10-19` &nbsp;|&nbsp; **Total Passages**: `21` &nbsp;|&nbsp; **Verified Ground Truth**: `3`

---

## 1. Overall Head-to-Head Benchmark Metrics

| Evaluation Metric | Count | Percentage |
| :--- | :--- | :--- |
| Total Candidate Passages | **21** | 100.0% |
| Ground-Truth Reviewed Passages | **3** | 14.3% |
| Production Correct (vs Ground Truth) | **0** | 0.0% |
| Shadow T0 Correct (Visual + Schedule) | **1** | 33.3% |
| Shadow T1 Correct (+ Near NTES) | **3** | 100.0% |
| Shadow T2 Correct (+ Post NTES) | **3** | 100.0% |

## 2. Comparison Category Taxonomy Breakdown

| Taxonomy Category | Count | Meaning |
| :--- | :--- | :--- |
| `AGREE_UNVERIFIED` | **2** | 9.5% of total |
| `BOTH_ABSTAIN` | **9** | 42.9% of total |
| `DISAGREE_NO_GROUND_TRUTH` | **7** | 33.3% of total |
| `PRODUCTION_ABSTAINS_SHADOW_CORRECT` | **2** | 9.5% of total |
| `PRODUCTION_WRONG_SHADOW_CORRECT` | **1** | 4.8% of total |

## 3. Per-Site Isolation & Performance Breakdown (§22)

| Site ID | Total Passages | Match Rate | Conflict Rate |
| :--- | :--- | :--- | :--- |
| `ndls_pf12_13` | **20** | 10.0% | 5.0% |
| `testx_pf1_2` | **1** | 0.0% | 0.0% |

## 4. Evidence Source Contribution Analysis (§21)

| Evidence Source | Passages Utilizing | Contribution Role |
| :--- | :--- | :--- |
| Numeric OCR (Stable Pair) | **10** | Primary 5-digit number candidate extraction |
| Train Name OCR | **6** | Non-numeric fallback & 4/6-number rake disambiguation |
| Route Endpoints | **1** | Corroborating directional terminal evidence |
| Site Timetable Match | **20** | Station schedule window and operating day validation |
| Near-Passage NTES (T1) | **13** | Live platform and movement corroboration |
| Post-Passage NTES (T2) | **4** | +180s departure and arrival persistence check |

## 5. Failure Reason Taxonomy (§23)

| Failure Reason | Occurrences | Root Cause |
| :--- | :--- | :--- |
| `MULTIPLE_LIVE_CANDIDATES` | **1** | Audited failure category |
| `NTES_UNAVAILABLE` | **4** | Audited failure category |
| `OCR_NUMBER_FAILURE` | **2** | Audited failure category |
| `TIMETABLE_NTES_CONFLICT` | **2** | Audited failure category |
| `VISUAL_NTES_CONFLICT` | **2** | Audited failure category |

## 6. Representative Passage Walkthrough (§30)

### Passage 1: `2026-10-13T06-54-40+05-30_ndls_pf12_13_P001` (ndls_pf12_13)
- **Time & Platform**: `2026-10-13T06:54:40+05:30` | Platform `13`
- **Visual Evidence**: Numbers `[]` | Name `[{'name': 'TAJ EXPRESS', 'distinct_coaches': 3, 'total_observations': 3, 'mean_confidence': 0.96}]`
- **Decision Pipeline**: T0=`12280` → T1=`12280` → T2=`12280` (**PROVISIONAL**)
- **Production vs Shadow**: Production=`12004` (LOW) | Shadow=`12280` | Result=`CONFLICT`
- **Ground Truth**: `12280` (operator_manual)

### Passage 2: `PAS_20261013_1720_NDLS_001` (ndls_pf12_13)
- **Time & Platform**: `2026-10-13T17:20:00+05:30` | Platform `['12', '13']`
- **Visual Evidence**: Numbers `['12919', '12920']` | Name `[{'name': 'MALWA EXPRESS', 'distinct_coaches': 3, 'total_observations': 3, 'mean_confidence': 0.96}]`
- **Decision Pipeline**: T0=`12314` → T1=`12920` → T2=`12920` (**PROVISIONAL**)
- **Production vs Shadow**: Production=`None` (LOW) | Shadow=`12920` | Result=`SHADOW_ONLY`
- **Ground Truth**: `12920` (human_expert)

### Passage 3: `PAS_20261013_2155_NDLS_006` (ndls_pf12_13)
- **Time & Platform**: `2026-10-13T21:55:00+05:30` | Platform `['12', '13']`
- **Visual Evidence**: Numbers `[]` | Name `[{'name': 'TAJ EXPRESS', 'distinct_coaches': 4, 'total_observations': 4, 'mean_confidence': 0.955}]`
- **Decision Pipeline**: T0=`None` → T1=`12280` → T2=`12280` (**PROVISIONAL**)
- **Production vs Shadow**: Production=`None` (LOW) | Shadow=`12280` | Result=`SHADOW_ONLY`
- **Ground Truth**: `12280` (synthetic_platform_benchmark)

### Passage 4: `PAS_CONFINED_001` (ndls_pf12_13)
- **Time & Platform**: `2026-10-13T17:20:00+05:30` | Platform `12`
- **Visual Evidence**: Numbers `[]` | Name `[]`
- **Decision Pipeline**: T0=`None` → T1=`None` → T2=`None` (**UNKNOWN**)
- **Production vs Shadow**: Production=`12920` (HIGH) | Shadow=`None` | Result=`PRODUCTION_ONLY`
- **Ground Truth**: `Unreviewed` (Pending)

### Passage 5: `PAS_ISOLATION_01` (testx_pf1_2)
- **Time & Platform**: `2026-10-13T17:20:00+05:30` | Platform `['1', '2']`
- **Visual Evidence**: Numbers `[]` | Name `[]`
- **Decision Pipeline**: T0=`None` → T1=`None` → T2=`None` (**UNKNOWN**)
- **Production vs Shadow**: Production=`None` (LOW) | Shadow=`None` | Result=`BOTH_UNKNOWN`
- **Ground Truth**: `Unreviewed` (Pending)

### Passage 6: `PAS_SAFE_001` (ndls_pf12_13)
- **Time & Platform**: `2026-10-13T17:20:00+05:30` | Platform `12`
- **Visual Evidence**: Numbers `['12920']` | Name `[]`
- **Decision Pipeline**: T0=`12920` → T1=`12920` → T2=`12920` (**PROVISIONAL**)
- **Production vs Shadow**: Production=`12920` (HIGH) | Shadow=`12920` | Result=`MATCH`
- **Ground Truth**: `Unreviewed` (Pending)

### Passage 7: `PAS_SCENARIO_01` (ndls_pf12_13)
- **Time & Platform**: `2026-10-13T17:20:00+05:30` | Platform `12`
- **Visual Evidence**: Numbers `['12919', '12920']` | Name `[]`
- **Decision Pipeline**: T0=`None` → T1=`12920` → T2=`12920` (**CONFIRMED**)
- **Production vs Shadow**: Production=`12920` (HIGH) | Shadow=`12920` | Result=`MATCH`
- **Ground Truth**: `Unreviewed` (Pending)

### Passage 8: `PAS_SCENARIO_02` (ndls_pf12_13)
- **Time & Platform**: `2026-10-13T21:55:00+05:30` | Platform `['12', '13']`
- **Visual Evidence**: Numbers `[]` | Name `[{'name': 'TAJ EXPRESS', 'distinct_coaches': 2, 'total_observations': 2, 'mean_confidence': 0.95}]`
- **Decision Pipeline**: T0=`None` → T1=`12280` → T2=`12280` (**PROVISIONAL**)
- **Production vs Shadow**: Production=`None` (LOW) | Shadow=`12280` | Result=`SHADOW_ONLY`
- **Ground Truth**: `Unreviewed` (Pending)

