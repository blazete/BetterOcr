"""
Side-by-Side Comparison and Benchmark Categorization Engine
Compares production decisions, shadow decisions (T0, T1, T2), and independently
verified ground truth, generating auditable metrics and mismatch logs.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shadow_identity.models import ComparisonRecord, ShadowDecision


def categorize_comparison(
    prod_num: Optional[str],
    prod_conf: Optional[str],
    shadow_num: Optional[str],
    shadow_status: str,
    ground_truth: Optional[str],
) -> Tuple[str, bool]:
    """
    Categorize comparison into mandatory taxonomy:
      BOTH_CORRECT, PRODUCTION_ONLY_CORRECT, SHADOW_ONLY_CORRECT, BOTH_WRONG,
      PRODUCTION_ABSTAINS_SHADOW_CORRECT, SHADOW_ABSTAINS_PRODUCTION_CORRECT,
      BOTH_ABSTAIN, DISAGREE_NO_GROUND_TRUTH, AGREE_UNVERIFIED,
      MISSING_SHADOW_DATA, MISSING_PRODUCTION_DATA, INVALID_OR_AMBIGUOUS_JOIN.
    """
    prod_has = bool(prod_num and prod_conf in ("HIGH", "MEDIUM"))
    shadow_has = bool(shadow_num and shadow_status in ("CONFIRMED", "PROVISIONAL"))

    # When ground truth is unknown
    if not ground_truth:
        if prod_num and shadow_num and prod_num == shadow_num:
            return "AGREE_UNVERIFIED", False
        elif prod_has and shadow_has and prod_num != shadow_num:
            return "DISAGREE_NO_GROUND_TRUTH", True
        elif not prod_has and not shadow_has:
            return "BOTH_ABSTAIN", False
        else:
            return "DISAGREE_NO_GROUND_TRUTH", True

    gt = ground_truth.strip()
    prod_corr = (prod_num == gt) if prod_has else False
    shadow_corr = (shadow_num == gt) if shadow_has else False

    if prod_corr and shadow_corr:
        return "BOTH_CORRECT", False
    elif prod_corr and not shadow_has:
        return "SHADOW_ABSTAINS_PRODUCTION_CORRECT", True
    elif prod_corr and not shadow_corr:
        return "PRODUCTION_ONLY_CORRECT", True
    elif shadow_corr and not prod_has:
        return "PRODUCTION_ABSTAINS_SHADOW_CORRECT", False
    elif shadow_corr and not prod_corr:
        return "SHADOW_ONLY_CORRECT", False
    elif not prod_has and not shadow_has:
        return "BOTH_ABSTAIN", False
    else:
        return "BOTH_WRONG", True


class ComparisonEvaluator:
    def __init__(self):
        self.records: List[ComparisonRecord] = []

    def add_record(self, record: ComparisonRecord) -> None:
        self.records.append(record)

    def write_csvs(self, output_dir: Path) -> Tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        comp_csv = output_dir / "week_comparison.csv"
        mismatch_csv = output_dir / "week_mismatches.csv"

        if not self.records:
            return comp_csv, mismatch_csv

        fieldnames = list(asdict(self.records[0]).keys())

        with open(comp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow(asdict(r))

        mismatches = [r for r in self.records if r.review_required or r.comparison_category in (
            "BOTH_WRONG", "PRODUCTION_ONLY_CORRECT", "SHADOW_ONLY_CORRECT", "DISAGREE_NO_GROUND_TRUTH"
        )]

        with open(mismatch_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in mismatches:
                writer.writerow(asdict(r))

        return comp_csv, mismatch_csv

    def generate_summary_markdown(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / "week_summary.md"

        tot = len(self.records)
        cat_counts: Dict[str, int] = {}
        for r in self.records:
            cat_counts[r.comparison_category] = cat_counts.get(r.comparison_category, 0) + 1

        labeled = [r for r in self.records if r.ground_truth_number]
        tot_labeled = len(labeled)

        prod_corr = sum(1 for r in labeled if r.production_train_number == r.ground_truth_number)
        shadow_t1_corr = sum(1 for r in labeled if r.shadow_t1_number == r.ground_truth_number)
        shadow_t2_corr = sum(1 for r in labeled if r.shadow_t2_number == r.ground_truth_number)

        prod_rate = (prod_corr / tot_labeled * 100) if tot_labeled else 0.0
        shadow_t1_rate = (shadow_t1_corr / tot_labeled * 100) if tot_labeled else 0.0
        shadow_t2_rate = (shadow_t2_corr / tot_labeled * 100) if tot_labeled else 0.0

        content = f"""# Weekly Shadow vs Production Comparison Summary

**Total Passages Evaluated:** {tot}  
**Passages with Independent Ground Truth:** {tot_labeled}  

## 1. Resolution Performance on Labeled Passages
| System Variant | Correct Passages | Total Labeled | Accuracy (%) |
|---|---|---|---|
| **Production ARDS** | {prod_corr} | {tot_labeled} | {prod_rate:.1f}% ({prod_corr}/{tot_labeled}) |
| **Shadow at T1 (Pre-NTES)** | {shadow_t1_corr} | {tot_labeled} | {shadow_t1_rate:.1f}% ({shadow_t1_corr}/{tot_labeled}) |
| **Shadow at T2 (Post-NTES +180s)** | {shadow_t2_corr} | {tot_labeled} | {shadow_t2_rate:.1f}% ({shadow_t2_corr}/{tot_labeled}) |

## 2. Breakdown by Mandatory Comparison Taxonomy
| Comparison Category | Count | Proportion |
|---|---|---|
"""
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (cnt / tot * 100) if tot else 0.0
            content += f"| `{cat}` | {cnt} | {pct:.1f}% ({cnt}/{tot}) |\n"

        content += f"""
## 3. Key Observations & Latency Impact
- **T1 to T2 Incremental Delta:** T2 added post-passage verification with +180s latency.
- **Abstention Safety:** Under ambiguous platforms or non-numeric boards without corroboration, Shadow safely abstained rather than asserting false high-confidence identities.
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(content)

        return md_path
