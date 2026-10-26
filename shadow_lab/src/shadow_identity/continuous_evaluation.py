"""
Continuous Evaluation and Ground Truth Isolation Module (Package D)
Implements:
D1 — Separately managed, evaluator-owned Ground Truth Store.
     Keyed by site_id + passage_id + source_sha256.
     Enforces strict isolation from inference, auditable corrections, and pair rejection.
D2 — Precise metric computation (eligible, emitted, coverage, precision, recall, INSUFFICIENT_DATA).
D3 — Daily review report generator (markdown & json in local shadow-owned storage).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from zoneinfo import ZoneInfo


@dataclass
class GroundTruthRecord:
    site_id: str
    passage_id: str
    source_sha256: str
    verified_active_train_number: Optional[str] = None
    verified_train_name: Optional[str] = None
    verified_service_family: List[str] = field(default_factory=list)
    verification_source: str = "INDEPENDENT_OPERATOR"
    verified_platform: Optional[str] = None
    verified_direction: Optional[str] = None
    label_quality: str = "VERIFIED_ACTIVE"  # "VERIFIED_ACTIVE" | "UNVERIFIED_PAIR" | "UNLABELED"
    verified_by: Optional[str] = None
    verified_at_utc: Optional[str] = None

    def __post_init__(self):
        # Enforce Rule D1: Bidirectional pair notation is NOT verified active GT
        if self.verified_active_train_number:
            num = self.verified_active_train_number.strip()
            if "/" in num or "-" in num:
                self.verified_active_train_number = None
                self.label_quality = "UNVERIFIED_PAIR"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroundTruthRecord:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class GroundTruthStore:
    """
    Evaluator-owned isolated store for passage ground truth.
    Key structure: site_id + passage_id + source_sha256.
    """

    def __init__(self, store_dir: Union[str, Path]):
        self.store_dir = Path(store_dir).resolve()
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, site_id: str, passage_id: str, source_sha256: str) -> str:
        s_site = site_id.replace("/", "_")
        s_pas = passage_id.replace("/", "_")
        s_sha = source_sha256[:64]
        return f"{s_site}_{s_pas}_{s_sha}"

    def save_record(self, record: GroundTruthRecord) -> Path:
        key = self._make_key(record.site_id, record.passage_id, record.source_sha256)
        path = self.store_dir / f"{key}_gt.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)
        return path

    def get_record(self, site_id: str, passage_id: str, source_sha256: str) -> Optional[GroundTruthRecord]:
        key = self._make_key(site_id, passage_id, source_sha256)
        path = self.store_dir / f"{key}_gt.json"
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return GroundTruthRecord.from_dict(data)
        except Exception:
            return None

    def list_records(self) -> List[GroundTruthRecord]:
        records: List[GroundTruthRecord] = []
        for p in sorted(self.store_dir.glob("*_gt.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    records.append(GroundTruthRecord.from_dict(json.load(f)))
            except Exception:
                continue
        return records


def compute_v1_evaluation_metrics(
    results: List[Dict[str, Any]],
    gt_records: List[GroundTruthRecord],
) -> Dict[str, Any]:
    """
    D2: Formal evaluation metrics computation per Part 6 specification.
    """
    total_passages = len(results)
    processed_passages = sum(1 for r in results if r.get("identity_status") != "FAILED")
    failed_passages = sum(1 for r in results if r.get("identity_status") == "FAILED")

    gt_map: Dict[str, GroundTruthRecord] = {}
    for gt in gt_records:
        k_full = f"{gt.site_id}_{gt.passage_id}_{gt.source_sha256}"
        gt_map[k_full] = gt

    eligible_passages: List[Tuple[Dict[str, Any], GroundTruthRecord]] = []
    for r in results:
        pid = r.get("passage_id", "")
        sid = r.get("site_id", "")
        sha = r.get("input_video_sha256") or r.get("mp4_sha256", "")
        gt = gt_map.get(f"{sid}_{pid}_{sha}")
        if gt and gt.label_quality == "VERIFIED_ACTIVE" and gt.verified_active_train_number:
            eligible_passages.append((r, gt))

    verified_gt_passages = len(eligible_passages)

    emitted: List[Tuple[Dict[str, Any], GroundTruthRecord]] = [
        (r, gt) for r, gt in eligible_passages if r.get("active_service_number") is not None
    ]

    correctly_predicted = sum(
        1 for r, gt in emitted if str(r.get("active_service_number")).strip() == str(gt.verified_active_train_number).strip()
    )

    incorrect_active_predictions = len(emitted) - correctly_predicted

    # Check for unsupported verified decisions
    unsupported_verified_decisions = sum(
        1 for r in results
        if (
            (r.get("identity_status") == "VERIFIED_ACTIVE" or r.get("resolver_status") == "CONFIRMED")
            and not r.get("active_direction_verified", False)
        )
    )

    # Coverage, Precision, Recall
    if eligible_passages:
        active_prediction_coverage = round(len(emitted) / len(eligible_passages), 4)
        active_prediction_recall = round(correctly_predicted / len(eligible_passages), 4)
    else:
        active_prediction_coverage = "INSUFFICIENT_DATA"
        active_prediction_recall = "INSUFFICIENT_DATA"

    if emitted:
        active_prediction_precision = round(correctly_predicted / len(emitted), 4)
    else:
        active_prediction_precision = "INSUFFICIENT_DATA"

    # Family Recall
    family_eligible = [
        (r, gt) for r, gt in eligible_passages if gt.verified_service_family
    ]
    if family_eligible:
        family_correct = sum(
            1 for r, gt in family_eligible
            if any(cand in gt.verified_service_family for cand in r.get("service_family_candidates", []))
        )
        family_recall = round(family_correct / len(family_eligible), 4)
    else:
        family_recall = "INSUFFICIENT_DATA"

    # Abstention Rate
    abstained = sum(1 for r in results if r.get("active_service_number") is None)
    abstention_rate = round(abstained / total_passages, 4) if total_passages else 0.0

    # Latencies
    durations = sorted([float(r.get("duration_sec", 0.0)) for r in results if r.get("duration_sec") is not None])
    p50_latency = None
    p95_latency = None
    if durations:
        idx_p50 = int(len(durations) * 0.5)
        idx_p95 = min(len(durations) - 1, int(len(durations) * 0.95))
        p50_latency = round(durations[idx_p50], 3)
        p95_latency = round(durations[idx_p95], 3)

    # OCR calls
    calls = [int(r.get("ocr_calls") or r.get("processing_metrics", {}).get("ocr_calls_executed", 0)) for r in results]
    average_ocr_calls = round(sum(calls) / total_passages, 2) if total_passages else 0.0

    return {
        "total_passages": total_passages,
        "processed_passages": processed_passages,
        "failed_passages": failed_passages,
        "verified_gt_passages": verified_gt_passages,
        "family_recall": family_recall,
        "active_prediction_coverage": active_prediction_coverage,
        "active_prediction_precision": active_prediction_precision,
        "active_prediction_recall": active_prediction_recall,
        "incorrect_active_predictions": incorrect_active_predictions,
        "unsupported_verified_decisions": unsupported_verified_decisions,
        "abstention_rate": abstention_rate,
        "p50_latency_sec": p50_latency,
        "p95_latency_sec": p95_latency,
        "peak_memory_mb": None,
        "average_ocr_calls": average_ocr_calls,
    }


def generate_daily_review_report(
    date_str: str,
    results: List[Dict[str, Any]],
    gt_records: List[GroundTruthRecord],
    output_dir: Union[str, Path],
) -> Tuple[Path, Path]:
    """
    D3: Read-only daily review report generator.
    Stores report locally inside shadow-owned output directory.
    """
    out_p = Path(output_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)

    metrics = compute_v1_evaluation_metrics(results, gt_records)

    verified_active_count = sum(1 for r in results if r.get("identity_status") == "VERIFIED_ACTIVE")
    family_candidate_count = sum(1 for r in results if r.get("identity_status") == "FAMILY_CANDIDATE")
    unknown_count = sum(1 for r in results if r.get("identity_status") == "UNKNOWN")
    contradictory_count = sum(1 for r in results if r.get("identity_status") == "CONFLICT" or r.get("contradictions"))

    metrics["verified_active_count"] = verified_active_count
    metrics["family_candidate_count"] = family_candidate_count
    metrics["unknown_count"] = unknown_count
    metrics["contradictory_cases"] = contradictory_count

    md_path = out_p / f"daily_review_{date_str}.md"
    json_path = out_p / f"daily_review_{date_str}.json"

    doc = {
        "report_date": date_str,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "passages_count": len(results),
        "passages": results,
        "unresolved_blockers": [
            "Hardware directional sensor authentic link verification",
            "Single-coach stencil verification against hardware axle boundaries",
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    md_content = f"""# Daily ShadowOCR Review Report — {date_str}

**Generated at:** `{doc['generated_at_utc']}`  
**Site:** `ndls_pf12_13`  
**Execution Environment:** Production-Independent ShadowOCR V1  

---

## 1. Passage Processing Summary
- **Total Observed Passages:** {metrics['total_passages']}
- **Successfully Processed:** {metrics['processed_passages']}
- **Failed / Pending:** {metrics['failed_passages']}

## 2. Identity Status Breakdown
- **Verified Active Identities:** {verified_active_count}
- **Family-Only Candidates:** {family_candidate_count}
- **Unknown Passages:** {unknown_count}
- **Contradictory Evidence Cases:** {contradictory_count}
- **Abstention Rate:** {metrics['abstention_rate']}

## 3. Evaluation Against Ground Truth
- **Passages with Independent Active GT:** {metrics['verified_gt_passages']}
- **Active Prediction Coverage:** {metrics['active_prediction_coverage']}
- **Active Prediction Precision:** {metrics['active_prediction_precision']}
- **Active Prediction Recall:** {metrics['active_prediction_recall']}
- **Family Recall:** {metrics['family_recall']}
- **Wrong Active Predictions:** {metrics['incorrect_active_predictions']}
- **Unsupported Verified Decisions:** {metrics['unsupported_verified_decisions']}

## 4. Performance & Operational Budget
- **P50 Latency:** {metrics['p50_latency_sec']}s
- **P95 Latency:** {metrics['p95_latency_sec']}s
- **Average OCR Calls:** {metrics['average_ocr_calls']} (Max Budget: 25)

## 5. Top Unresolved Blockers
"""
    for b in doc["unresolved_blockers"]:
        md_content += f"- {b}\n"
    md_content += "\n## 6. Passage Results\n\n| Passage | Source SHA-256 | Status | Active Number | Direction Verified | Worker Status | Error |\n| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for result in results:
        md_content += (
            f"| `{result.get('passage_id', '')}` | `{result.get('input_video_sha256', '')}` | "
            f"{result.get('identity_status', 'UNKNOWN')} | `{result.get('active_service_number') or ''}` | "
            f"{result.get('active_direction_verified', False)} | {result.get('worker_status', '')} | "
            f"{result.get('error_message', '')} |\n"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return md_path, json_path


def generate_worker_daily_review_report(
    report_date: str,
    worker_state_path: Union[str, Path],
    ground_truth_dir: Union[str, Path],
    output_dir: Union[str, Path],
    site_id: Optional[str] = None,
    timezone_name: str = "Asia/Kolkata",
) -> Tuple[Path, Path]:
    """Build the local daily report from persisted worker decisions and separate GT records."""
    report_day = date.fromisoformat(report_date)
    zone = ZoneInfo(timezone_name)
    state_path = Path(worker_state_path)
    jobs = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    results: List[Dict[str, Any]] = []

    for job in jobs.values():
        if site_id and job.get("site_id") != site_id:
            continue
        decision_path = Path(job["output_json_path"]) if job.get("output_json_path") else None
        decision: Dict[str, Any] = {}
        if decision_path and decision_path.is_file():
            try:
                decision = json.loads(decision_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                decision = {}

        metadata = decision.get("passage_metadata", {})
        passage_time = metadata.get("passage_start") or metadata.get("passage_end")
        try:
            if passage_time:
                parsed_time = datetime.fromisoformat(str(passage_time).replace("Z", "+00:00"))
                if parsed_time.tzinfo is None:
                    parsed_time = parsed_time.replace(tzinfo=zone)
                job_day = parsed_time.astimezone(zone).date()
            else:
                epoch = job.get("completed_at_epoch") or job.get("discovered_at_epoch")
                if epoch is None:
                    continue
                job_day = datetime.fromtimestamp(float(epoch), zone).date()
        except (ValueError, TypeError, OSError, OverflowError):
            continue
        if job_day != report_day:
            continue

        if not decision:
            results.append({
                "job_id": job.get("job_id"),
                "passage_id": job.get("passage_id"),
                "site_id": job.get("site_id"),
                "input_video_sha256": job.get("discovered_sha256") or job.get("mp4_sha256"),
                "identity_status": "FAILED",
                "worker_status": job.get("status"),
                "error_message": job.get("error_message"),
            })
            continue

        worker_sha = job.get("discovered_sha256") or job.get("mp4_sha256")
        decision_sha = decision.get("input_video_sha256") or metadata.get("video_sha256")
        if worker_sha and worker_sha != decision_sha:
            results.append({
                "job_id": job.get("job_id"),
                "passage_id": job.get("passage_id"),
                "site_id": job.get("site_id"),
                "input_video_sha256": worker_sha,
                "identity_status": "FAILED",
                "worker_status": job.get("status"),
                "error_message": "DECISION_SOURCE_HASH_MISMATCH",
            })
            continue

        resolver = decision.get("resolver_decision", {})
        performance = decision.get("performance_latency", {})
        visual = decision.get("visual_evidence", {})
        results.append({
            "job_id": job.get("job_id"),
            "passage_id": decision.get("passage_id") or job.get("passage_id"),
            "site_id": decision.get("site_id") or job.get("site_id"),
            "input_video_sha256": decision_sha or worker_sha,
            "active_service_number": decision.get("active_service_number"),
            "active_direction_verified": decision.get("active_direction_verified", False),
            "identity_status": decision.get("identity_status", "UNKNOWN"),
            "resolver_status": resolver.get("shadow_status"),
            "reason_codes": decision.get("reason_codes", []),
            "contradictions": decision.get("contradictions", []),
            "duration_sec": job.get("duration_sec") or performance.get("total_wall_clock_s"),
            "ocr_calls": performance.get("ocr_calls_executed", visual.get("ocr_calls_executed", 0)),
            "worker_status": job.get("status"),
        })

    gt_path = Path(ground_truth_dir)
    gt_records = GroundTruthStore(gt_path).list_records() if gt_path.is_dir() else []
    return generate_daily_review_report(report_date, results, gt_records, output_dir)
