"""
Regression tests for BetterOcr / ShadowOCR V1 Package D:
Continuous Evaluation & Ground Truth Isolation.

Requirements:
D1 — Isolated Ground Truth Store (keyed by site_id + passage_id + source_sha256).
     Bidirectional pair is not verified active GT. No leaking to inference.
D2 — Evaluation Metrics:
     eligible, emitted, coverage, precision, recall, family_recall, abstention_rate.
     INSUFFICIENT_DATA when eligible or emitted is empty.
     Zero tolerance for false confirmations / unsupported verified decisions.
D3 — Daily Review Report:
     Read-only local report generation, no network transmission.
"""

from pathlib import Path
import json
import pytest

from shadow_identity.continuous_evaluation import (
    GroundTruthRecord,
    GroundTruthStore,
    compute_v1_evaluation_metrics,
    generate_daily_review_report,
    generate_worker_daily_review_report,
)


def test_d1_ground_truth_isolation_and_keying(tmp_path):
    """D1: Ground truth is stored separately, keyed by site_id + passage_id + source_sha256."""
    gt_dir = tmp_path / "gt_store"
    store = GroundTruthStore(gt_dir)

    record = GroundTruthRecord(
        site_id="ndls_pf12_13",
        passage_id="pas_001",
        source_sha256="abc123sha256",
        verified_active_train_number="12417",
        verified_train_name="PRAYAGRAJ EXP",
        verified_service_family=["12417", "12418"],
        verification_source="STATION_LOG_AUDITED",
        verified_platform="12",
        verified_direction="UP",
        label_quality="VERIFIED_ACTIVE",
        verified_by="chief_controller",
        verified_at_utc="2026-10-26T10:00:00Z",
    )
    store.save_record(record)

    loaded = store.get_record(site_id="ndls_pf12_13", passage_id="pas_001", source_sha256="abc123sha256")
    assert loaded is not None
    assert loaded.verified_active_train_number == "12417"

    # Bidirectional pair without direction verification cannot be active GT
    pair_record = GroundTruthRecord(
        site_id="ndls_pf12_13",
        passage_id="pas_002",
        source_sha256="def456sha256",
        verified_active_train_number=None,
        verified_service_family=["12559", "12560"],
        label_quality="UNVERIFIED_PAIR",
    )
    store.save_record(pair_record)
    loaded_pair = store.get_record("ndls_pf12_13", "pas_002", "def456sha256")
    assert loaded_pair.verified_active_train_number is None


def test_d2_evaluation_metrics_definitions():
    """D2: Metric definitions, coverage, precision, recall, and INSUFFICIENT_DATA handling."""
    # Scenario 1: No GT exists -> Must NOT report 0% or 100% accuracy, report INSUFFICIENT_DATA
    results_no_gt = [
        {
            "job_id": "job_01",
            "passage_id": "pas_01",
            "site_id": "ndls_pf12_13",
            "input_video_sha256": "sha1",
            "active_service_number": "12417",
            "active_direction_verified": True,
            "identity_status": "VERIFIED_ACTIVE",
            "duration_sec": 1.2,
            "ocr_calls": 20,
        }
    ]
    metrics_empty = compute_v1_evaluation_metrics(results=results_no_gt, gt_records=[])
    assert metrics_empty["verified_gt_passages"] == 0
    assert metrics_empty["active_prediction_coverage"] == "INSUFFICIENT_DATA"
    assert metrics_empty["active_prediction_precision"] == "INSUFFICIENT_DATA"
    assert metrics_empty["active_prediction_recall"] == "INSUFFICIENT_DATA"

    # Scenario 2: Eligible passages with predictions
    gt_records = [
        GroundTruthRecord(
            site_id="ndls_pf12_13",
            passage_id="pas_01",
            source_sha256="sha1",
            verified_active_train_number="12417",
            label_quality="VERIFIED_ACTIVE",
        ),
        GroundTruthRecord(
            site_id="ndls_pf12_13",
            passage_id="pas_02",
            source_sha256="sha2",
            verified_active_train_number="12560",
            label_quality="VERIFIED_ACTIVE",
        ),
    ]
    results = [
        {
            "passage_id": "pas_01",
            "site_id": "ndls_pf12_13",
            "input_video_sha256": "sha1",
            "active_service_number": "12417",  # Correct
            "active_direction_verified": True,
            "identity_status": "VERIFIED_ACTIVE",
            "duration_sec": 1.0,
            "ocr_calls": 22,
        },
        {
            "passage_id": "pas_02",
            "site_id": "ndls_pf12_13",
            "input_video_sha256": "sha2",
            "active_service_number": None,  # Abstained
            "active_direction_verified": False,
            "identity_status": "FAMILY_CANDIDATE",
            "duration_sec": 1.5,
            "ocr_calls": 24,
        },
    ]
    metrics = compute_v1_evaluation_metrics(results=results, gt_records=gt_records)
    assert metrics["total_passages"] == 2
    assert metrics["verified_gt_passages"] == 2
    # 1 emitted out of 2 eligible -> coverage = 0.5
    assert metrics["active_prediction_coverage"] == 0.5
    # 1 correct out of 1 emitted -> precision = 1.0
    assert metrics["active_prediction_precision"] == 1.0
    # 1 correct out of 2 eligible -> recall = 0.5
    assert metrics["active_prediction_recall"] == 0.5
    assert metrics["incorrect_active_predictions"] == 0
    assert metrics["unsupported_verified_decisions"] == 0
    assert metrics["abstention_rate"] == 0.5


def test_d3_daily_review_report_generation(tmp_path):
    """D3: Generate local daily review report without any network calls."""
    out_dir = tmp_path / "reports"
    results = [
        {
            "passage_id": "pas_01",
            "site_id": "ndls_pf12_13",
            "input_video_sha256": "sha1",
            "active_service_number": "12417",
            "active_direction_verified": True,
            "identity_status": "VERIFIED_ACTIVE",
            "duration_sec": 1.1,
            "ocr_calls": 18,
            "contradictions": [],
        },
        {
            "passage_id": "pas_02",
            "site_id": "ndls_pf12_13",
            "input_video_sha256": "sha2",
            "active_service_number": None,
            "active_direction_verified": False,
            "identity_status": "CONFLICT",
            "duration_sec": 1.4,
            "ocr_calls": 21,
            "contradictions": ["DIRECTION_CONFLICT"],
        },
    ]
    gt_records = [
        GroundTruthRecord(
            site_id="ndls_pf12_13",
            passage_id="pas_01",
            source_sha256="sha1",
            verified_active_train_number="12417",
            label_quality="VERIFIED_ACTIVE",
        )
    ]

    rep_path, rep_json = generate_daily_review_report(
        date_str="2026-10-26",
        results=results,
        gt_records=gt_records,
        output_dir=out_dir,
    )
    assert rep_path.is_file()
    assert rep_json.is_file()
    content = rep_path.read_text(encoding="utf-8")
    assert "Daily ShadowOCR Review Report" in content
    assert "Verified Active Identities" in content
    assert "Contradictory Evidence Cases" in content

    data = json.loads(rep_json.read_text(encoding="utf-8"))
    assert data["report_date"] == "2026-10-26"
    assert data["metrics"]["total_passages"] == 2
    assert data["metrics"]["contradictory_cases"] == 1


def test_worker_daily_report_reads_persisted_unknown_and_exact_hash_gt_only(tmp_path):
    output = tmp_path / "worker_output"
    output.mkdir()
    decision_path = output / "job_decision.json"
    decision_path.write_text(json.dumps({
        "site_id": "ndls_pf12_13",
        "passage_id": "pas_01",
        "input_video_sha256": "source-hash-a",
        "identity_status": "UNKNOWN",
        "active_service_number": None,
        "active_direction_verified": False,
        "reason_codes": ["NO_VERIFIED_NUMBER"],
        "contradictions": [],
        "resolver_decision": {"shadow_status": "UNKNOWN"},
        "performance_latency": {"total_wall_clock_s": 3.5, "ocr_calls_executed": 7},
        "passage_metadata": {"passage_start": "2026-10-26T10:00:00+05:30"},
    }))
    state_path = tmp_path / "worker_jobs.json"
    state_path.write_text(json.dumps({"job-a": {
        "job_id": "job-a", "site_id": "ndls_pf12_13", "passage_id": "pas_01",
        "discovered_sha256": "source-hash-a", "status": "COMPLETED",
        "output_json_path": str(decision_path), "duration_sec": 3.5,
        "completed_at_epoch": 1792989000,
    }}))
    gt_store = GroundTruthStore(tmp_path / "ground_truth")
    gt_store.save_record(GroundTruthRecord(
        site_id="ndls_pf12_13", passage_id="pas_01", source_sha256="source-hash-old",
        verified_active_train_number="12445", label_quality="VERIFIED_ACTIVE",
    ))

    md_path, json_path = generate_worker_daily_review_report(
        report_date="2026-10-26", worker_state_path=state_path,
        ground_truth_dir=tmp_path / "ground_truth", output_dir=tmp_path / "reports",
    )
    report = json.loads(json_path.read_text())
    assert md_path.is_file()
    assert report["passages_count"] == 1
    assert report["passages"][0]["identity_status"] == "UNKNOWN"
    assert report["passages"][0]["ocr_calls"] == 7
    assert report["metrics"]["unknown_count"] == 1
    assert report["metrics"]["verified_gt_passages"] == 0
    assert report["metrics"]["active_prediction_coverage"] == "INSUFFICIENT_DATA"


def test_worker_daily_report_marks_decision_source_hash_mismatch_failed(tmp_path):
    decision_path = tmp_path / "job_decision.json"
    decision_path.write_text(json.dumps({
        "site_id": "ndls_pf12_13", "passage_id": "pas_02",
        "input_video_sha256": "wrong-source", "identity_status": "VERIFIED_ACTIVE",
        "active_service_number": "12445", "active_direction_verified": True,
        "passage_metadata": {"passage_start": "2026-10-26T10:00:00+05:30"},
    }))
    state_path = tmp_path / "worker_jobs.json"
    state_path.write_text(json.dumps({"job-b": {
        "job_id": "job-b", "site_id": "ndls_pf12_13", "passage_id": "pas_02",
        "discovered_sha256": "actual-source", "status": "COMPLETED",
        "output_json_path": str(decision_path), "completed_at_epoch": 1792989000,
    }}))
    _, json_path = generate_worker_daily_review_report(
        report_date="2026-10-26", worker_state_path=state_path,
        ground_truth_dir=tmp_path / "missing_gt", output_dir=tmp_path / "reports",
    )
    report = json.loads(json_path.read_text())
    assert report["passages"][0]["identity_status"] == "FAILED"
    assert report["passages"][0]["error_message"] == "DECISION_SOURCE_HASH_MISMATCH"
    assert report["passages"][0].get("active_service_number") is None
