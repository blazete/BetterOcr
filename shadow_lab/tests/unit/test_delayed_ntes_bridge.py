"""
Delayed NTES Operational Bridge & Service Tests (§1 - §8)

Proves:
1. A task does not execute before its due time (+180s).
2. It executes after +180 seconds.
3. Station A never fetches Station B (strict site isolation).
4. A failed NTES request leaves the passage valid and marks T2 unavailable.
5. The worker never writes outside shadow_lab.
6. The worker loop service periodically calls process_due() and respects due times.
"""

from __future__ import annotations

import builtins
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from shadow_identity.config_loader import ConfigError, SiteRegistry
from shadow_identity.delayed_ntes_worker import (
    DelayedNtesVerificationWorker,
    DelayedVerificationTask,
)
from shadow_identity.models import (
    NameCandidate,
    NtesSnapshot,
    NtesTrainEntry,
    NumberCandidate,
    PassageBundle,
    PassageInput,
    PassageVisualConsensus,
    VisualObservation,
    VisualOcrEvidence,
)
from shadow_identity.ntes_adapter import ArdsNtesAdapter
from shadow_lab.scripts.verify_originals_unchanged import verify_originals


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parents[3]


@pytest.fixture
def shadow_lab_root(repo_root):
    return repo_root / "shadow_lab"


@pytest.fixture(autouse=True)
def cleanup_test_artifacts(shadow_lab_root):
    snaps_before = set((shadow_lab_root / "data" / "ntes_snapshots").rglob("*.json"))
    yield
    # Clean up test-created passages and results
    for p in (shadow_lab_root / "data" / "passages").glob("P_*.json"):
        try: p.unlink()
        except OSError: pass
    for p in (shadow_lab_root / "results" / "passage_results").glob("P_*.json"):
        try: p.unlink()
        except OSError: pass
    snaps_after = set((shadow_lab_root / "data" / "ntes_snapshots").rglob("*.json"))
    for p in (snaps_after - snaps_before):
        try: p.unlink()
        except OSError: pass
    testx_dir = shadow_lab_root / "data" / "ntes_snapshots" / "testx_pf1_2"
    if testx_dir.exists() and not any(testx_dir.iterdir()):
        try: testx_dir.rmdir()
        except OSError: pass


@pytest.fixture
def sample_passage_bundle():
    """Create canonical passage bundle dict for testing with schedule-matched train 12560."""
    return {
        "site_id": "ndls_pf12_13",
        "passage_id": "P_BRIDGE_TEST_001",
        "passage": {
            "passage_id": "P_BRIDGE_TEST_001",
            "start_time": "2026-10-13T20:00:00+05:30",
            "end_time": "2026-10-13T20:01:00+05:30",
            "observed_platform": "12",
            "possible_platforms": ["12", "13"],
            "direction": "down",
        },
        "production": {
            "train_number": "12560",
            "confidence": "HIGH",
            "source": "production_ards",
        },
        "visual": {
            "service_numbers": [
                {"number": "12560", "distinct_coaches": 3, "total_observations": 3, "mean_confidence": 0.95}
            ],
            "stable_pair": ["12559", "12560"],
            "train_name_candidates": [
                {"name": "SHIV GANGA EXP", "distinct_coaches": 2, "total_observations": 2}
            ],
            "endpoint_candidates": ["NDLS", "BSBS"],
            "number_board_type": "PRINTED_VINYL",
            "numeric_status": "PAIR_CONFIRMED",
            "coach_evidence": [],
        },
        "timetable": {
            "version": "2026-10-13-validated-v1",
            "candidate_services": [
                {"train_number": "12560", "train_name": "NDLS-BSBS (SHIV GANGA EXP.)", "score": 2.5}
            ],
        },
        "ntes": {
            "pre_or_near_passage_snapshot": None,
            "post_passage_snapshot": None,
        },
        "shadow_result": {
            "T0": {"train_number": "12560", "status": "CONFIRMED", "reason_codes": []},
            "T1": {"train_number": "12560", "status": "CONFIRMED", "reason_codes": []},
            "T2": {"train_number": None, "status": "PENDING", "reason_codes": []},
        },
        "ground_truth": {
            "train_number": "12560",
            "train_name": "SHIV GANGA EXP",
            "reviewer": "test_suite",
        },
        "comparison": {
            "category": "BOTH_CORRECT",
            "production_vs_shadow": "MATCH",
            "review_required": False,
            "failure_reason": None,
        },
    }


def test_task_does_not_execute_before_due_time(shadow_lab_root, sample_passage_bundle, tmp_path):
    """
    Test §8.1: A task does not execute before its due time.
    Scheduled verify at = passage_end + 180s.
    Running process_due() at t + 60s must leave task in PENDING status.
    """
    worker = DelayedNtesVerificationWorker(
        shadow_lab_root=shadow_lab_root,
        queue_file=tmp_path / "queue.json",
    )

    passage_id = "P_DUE_CHECK_01"
    sample_passage_bundle["passage_id"] = passage_id
    sample_passage_bundle["passage"]["passage_id"] = passage_id

    bundle_file = worker.passages_dir / f"{passage_id}.json"
    with open(bundle_file, "w", encoding="utf-8") as f:
        json.dump(sample_passage_bundle, f, indent=2)

    end_time_dt = datetime(2026, 10, 13, 20, 1, 0)
    end_time_str = end_time_dt.isoformat()

    # Enqueue task with 180s delay
    task = worker.enqueue_passage(
        passage_id=passage_id,
        site_id="ndls_pf12_13",
        passage_end_time=end_time_str,
        delay_seconds=180.0,
    )

    sched_dt = datetime.fromisoformat(task.scheduled_verify_at)
    assert sched_dt == end_time_dt + timedelta(seconds=180.0)

    # Simulate clock at +60s (NOT yet due)
    clock_before_due = end_time_dt + timedelta(seconds=60.0)
    processed = worker.process_due(now_dt=clock_before_due)

    # Task must NOT have executed
    assert len(processed) == 0
    assert worker._tasks[passage_id].status == "PENDING"
    assert worker._tasks[passage_id].processed_at is None


def test_task_executes_after_plus_180_seconds(shadow_lab_root, sample_passage_bundle, tmp_path):
    """
    Test §8.2: A task executes after +180 seconds.
    Running process_due() at t + 185s executes T2 and updates status to PROCESSED.
    """
    worker = DelayedNtesVerificationWorker(
        shadow_lab_root=shadow_lab_root,
        queue_file=tmp_path / "queue.json",
    )

    passage_id = "P_EXEC_180_01"
    sample_passage_bundle["passage_id"] = passage_id
    sample_passage_bundle["passage"]["passage_id"] = passage_id

    # Store passage bundle
    bundle_file = worker.passages_dir / f"{passage_id}.json"
    with open(bundle_file, "w", encoding="utf-8") as f:
        json.dump(sample_passage_bundle, f, indent=2)

    end_time_dt = datetime(2026, 10, 13, 20, 1, 0)
    task = worker.enqueue_passage(
        passage_id=passage_id,
        site_id="ndls_pf12_13",
        passage_end_time=end_time_dt.isoformat(),
        delay_seconds=180.0,
    )

    # Clock at +185s (due time has arrived)
    clock_after_due = end_time_dt + timedelta(seconds=185.0)
    processed = worker.process_due(now_dt=clock_after_due)

    assert len(processed) == 1
    assert processed[0].passage_id == passage_id
    assert worker._tasks[passage_id].status == "PROCESSED"
    assert worker._tasks[passage_id].processed_at is not None
    assert worker._tasks[passage_id].t2_status is not None

    # Bundle in passages directory must be updated with T2 result
    with open(bundle_file, "r", encoding="utf-8") as f:
        saved_dict = json.load(f)
    assert saved_dict["shadow_result"]["T2"]["status"] in ("CONFIRMED", "PROVISIONAL")
    assert saved_dict["shadow_result"]["T2"]["train_number"] == "12560"


def test_station_a_never_fetches_station_b(shadow_lab_root, repo_root):
    """
    Test §8.3: Station A never fetches Station B.
    ndls_pf12_13 strictly uses NDLS config with PF 12/13.
    testx_pf1_2 strictly uses TESTX config with PF 1/2.
    Cache instances and query parameters are strictly isolated per site_id.
    """
    adapter = ArdsNtesAdapter(
        shadow_lab_root=shadow_lab_root,
        ards_root=repo_root / "ARDS-main",
    )

    # 1. Inspect Station A (NDLS)
    cache_a = adapter._get_or_create_cache("ndls_pf12_13")
    assert "NDLS" in cache_a._station
    assert "TESTX" not in cache_a._station
    assert cache_a._platforms == ["12", "13"]

    # 2. Inspect Station B (TESTX)
    cache_b = adapter._get_or_create_cache("testx_pf1_2")
    assert "TESTX" in cache_b._station
    assert "NDLS" not in cache_b._station
    assert cache_b._platforms == ["1", "2"]

    # 3. Cache instances are separate
    assert cache_a is not cache_b

    # 4. Unknown site raises ConfigError; silent fallback to NDLS is strictly forbidden
    with pytest.raises(ConfigError) as exc_info:
        adapter.get_snapshot("unknown_random_site")
    assert "Silent fallback to NDLS is strictly prohibited" in str(exc_info.value)


def test_failed_ntes_request_leaves_passage_valid_and_marks_t2_unavailable(
    shadow_lab_root, sample_passage_bundle, tmp_path
):
    """
    Test §8.4: A failed NTES request leaves the passage valid and marks T2 unavailable.
    The passage is NOT rejected or discarded.
    T2 post_fetch_kind is set to UNAVAILABLE.
    Shadow evaluation completes and comparison taxonomy remains intact.
    """
    # Custom provider that simulates failed NTES query
    def failing_snapshot_provider(site_id: str, passage_end_time: str) -> NtesSnapshot:
        return NtesSnapshot(
            site_id=site_id,
            station_code="NDLS",
            requested_at=datetime.now().isoformat(),
            fetched_at=datetime.now().isoformat(),
            window_hr=2,
            available=False,
            from_cache=False,
            stale_fallback=False,
            source="ards_adapter_mock_failure",
            trains=[],
            raw_hash="",
        )

    worker = DelayedNtesVerificationWorker(
        shadow_lab_root=shadow_lab_root,
        snapshot_provider=failing_snapshot_provider,
        queue_file=tmp_path / "queue.json",
    )

    passage_id = "P_FAIL_NTES_01"
    sample_passage_bundle["passage_id"] = passage_id
    sample_passage_bundle["passage"]["passage_id"] = passage_id

    bundle_file = worker.passages_dir / f"{passage_id}.json"
    with open(bundle_file, "w", encoding="utf-8") as f:
        json.dump(sample_passage_bundle, f, indent=2)

    res = worker.process_passage(passage_id)

    # 1. Passage remains completely valid
    assert res is not None
    assert res.passage_id == passage_id

    # 2. Stage T2 post-fetch kind is marked UNAVAILABLE
    t2_result = res.shadow_result.get("T2", {})
    assert t2_result.get("post_fetch_kind") == "UNAVAILABLE"

    # 3. Decision status remains valid based on visual & timetable evidence
    assert t2_result.get("status") in ("CONFIRMED", "PROVISIONAL")
    assert t2_result.get("train_number") == "12560"

    # 4. Comparison category is properly formed
    assert res.comparison.get("category") == "BOTH_CORRECT"
    assert res.comparison.get("production_vs_shadow") == "MATCH"


def test_worker_never_writes_outside_shadow_lab(
    shadow_lab_root, repo_root, monkeypatch, sample_passage_bundle, tmp_path
):
    """
    Test §8.5: The worker never writes outside shadow_lab.
    Installs a strict filesystem guard: any write to ARDS-main or outside shadow_lab
    immediately raises a PermissionError.
    Proves all snapshots and bundles are written strictly under shadow_lab.
    """
    # Guard: intercept all open() calls in write modes
    original_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(m in mode for m in ("w", "a", "+", "x")):
            path_str = str(Path(file).resolve())
            shadow_str = str(shadow_lab_root.resolve())
            ards_str = str((repo_root / "ARDS-main").resolve())
            tmp_str = str(tmp_path.resolve())

            if ards_str in path_str:
                raise PermissionError(f"Attempted write into ARDS production tree: {path_str}")
            if not (path_str.startswith(shadow_str) or path_str.startswith(tmp_str)):
                raise PermissionError(f"Attempted write outside shadow_lab directory: {path_str}")

        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    worker = DelayedNtesVerificationWorker(
        shadow_lab_root=shadow_lab_root,
        queue_file=tmp_path / "queue.json",
    )

    passage_id = "P_ZERO_WRITE_01"
    sample_passage_bundle["passage_id"] = passage_id
    sample_passage_bundle["passage"]["passage_id"] = passage_id

    bundle_file = worker.passages_dir / f"{passage_id}.json"
    with open(bundle_file, "w", encoding="utf-8") as f:
        json.dump(sample_passage_bundle, f, indent=2)

    # Save snapshot locally via worker
    mock_snap = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at=datetime.now().isoformat(),
        fetched_at=datetime.now().isoformat(),
        available=True,
        trains=[NtesTrainEntry(train_number="12560", train_name="SHIV GANGA EXP", platform="12")],
    )

    saved_snap_path = worker.save_snapshot_locally("ndls_pf12_13", mock_snap)

    # Verify snapshot was stored inside shadow_lab/data/ntes_snapshots/ndls_pf12_13/
    assert str(shadow_lab_root / "data" / "ntes_snapshots" / "ndls_pf12_13") in str(saved_snap_path)
    assert saved_snap_path.exists()

    # Process passage under guard
    res = worker.process_passage(passage_id, snapshot_override=mock_snap)
    assert res is not None

    # Check baseline ARDS integrity: all 456 files still match original baseline hashes
    assert verify_originals() is True


def test_worker_service_loop_only_processes_due_tasks(
    shadow_lab_root, sample_passage_bundle, tmp_path
):
    """
    Test §8.6: The worker loop service periodically calls process_due() and only
    processes tasks whose scheduled_verify_at has passed.
    """
    worker = DelayedNtesVerificationWorker(
        shadow_lab_root=shadow_lab_root,
        queue_file=tmp_path / "queue.json",
    )

    passage_id = "P_SERVICE_LOOP_01"
    sample_passage_bundle["passage_id"] = passage_id
    sample_passage_bundle["passage"]["passage_id"] = passage_id

    bundle_file = worker.passages_dir / f"{passage_id}.json"
    with open(bundle_file, "w", encoding="utf-8") as f:
        json.dump(sample_passage_bundle, f, indent=2)

    # Enqueue with future due time (+180s)
    future_time = datetime.now() + timedelta(seconds=180)
    worker.enqueue_passage(
        passage_id=passage_id,
        site_id="ndls_pf12_13",
        passage_end_time=future_time.isoformat(),
        delay_seconds=180.0,
    )

    # Run 1 iteration of loop: task must NOT be processed
    processed_count = worker.run_worker_loop(poll_interval_s=0.01, max_iterations=1)
    assert processed_count == 0
    assert worker._tasks[passage_id].status == "PENDING"

    # Now backdate scheduled_verify_at so it is due
    worker._tasks[passage_id].scheduled_verify_at = (datetime.now() - timedelta(seconds=10)).isoformat()
    worker._save_queue()

    # Run 1 iteration of loop: task must now be processed
    processed_count = worker.run_worker_loop(poll_interval_s=0.01, max_iterations=1)
    assert processed_count == 1
    assert worker._tasks[passage_id].status == "PROCESSED"
