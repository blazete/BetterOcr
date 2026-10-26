"""Generate a report from persisted ShadowOCR worker decisions without running inference."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from shadow_identity.continuous_evaluation import generate_worker_daily_review_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Operational date in YYYY-MM-DD form")
    parser.add_argument("--worker-state", required=True, help="Path to worker_jobs.json")
    parser.add_argument("--ground-truth-dir", required=True, help="Separate evaluator-owned GT store")
    parser.add_argument("--reports-dir", required=True, help="Shadow-owned report output directory")
    parser.add_argument("--site-id", default=None, help="Optional site filter")
    parser.add_argument("--timezone", default="Asia/Kolkata", help="Timezone used for operational-day grouping")
    args = parser.parse_args()

    try:
        md_path, json_path = generate_worker_daily_review_report(
            report_date=args.date,
            worker_state_path=Path(args.worker_state),
            ground_truth_dir=Path(args.ground_truth_dir),
            output_dir=Path(args.reports_dir),
            site_id=args.site_id,
            timezone_name=args.timezone,
        )
    except (ValueError, OSError) as exc:
        print(f"[FAIL] Could not generate daily worker report: {exc}", file=sys.stderr)
        return 2

    print(f"[PASS] Daily worker report: {md_path}")
    print(f"[PASS] Daily worker report data: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
