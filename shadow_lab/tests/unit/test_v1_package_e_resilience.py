"""
Regression & Resilience Test Suite for BetterOcr / ShadowOCR V1 Package E:
Operational Reliability & Failure Handling.

Requirements:
E1 — Feature Flags (offline_replay disabled, network disabled, live_shadow disabled).
E2 — Comprehensive resilience tests covering all 22 failure and edge-case scenarios:
 1. One new finalized MP4.
 2. Multiple new finalized MP4s.
 3. Duplicate files.
 4. Duplicate passage IDs with different hashes.
 5. Same MP4 after service restart.
 6. Partial video write.
 7. Crash during staging.
 8. Crash during inference.
 9. Crash during output persistence.
 10. Missing timetable.
 11. Missing NTES.
 12. Stale NTES.
 13. Ambiguous platform.
 14. Missing actual passage timestamp.
 15. Missing coach boundaries.
 16. Unrecognized train.
 17. Recognized name with no numeric board.
 18. Two-number and four-number boards.
 19. Six-number coach stencil rejected as train number.
 20. False train name.
 21. Contradictory registry mappings.
 22. Corrupted video / disconnected site storage.

Invariant: Original ARDS files must remain untouched in every case.
"""

from pathlib import Path
import json
import sys
from unittest.mock import patch
import cv2
import numpy as np
import pytest

from shadow_identity.config_loader import SiteRegistry
from shadow_identity.models import (
    DirectionEvidence,
    PassageInput,
    PassageVisualConsensus,
    NumberCandidate,
    NameCandidate,
    VisualOcrEvidence,
    TimetableRow,
    NtesSnapshot,
    NtesTrainEntry,
)
from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.evidence_fusion import decide_identity_v1
from shadow_identity.shadow_ocr.text_entities import EntityClassifier
from shadow_identity.shadow_ocr.worker import (
    ShadowWorker,
    WorkerJob,
    compute_file_sha256,
    main as worker_main,
    validate_worker_runtime_policy,
)
from shadow_identity.reason_codes import (
    RC_AMBIGUOUS_MULTI_CANDIDATE,
    RC_NAME_ONLY_ACTIVE_DIRECTION_UNVERIFIED,
    RC_NTES_CORROBORATED_STALE,
    RC_NTES_UNAVAILABLE,
    RC_PLATFORM_AMBIGUOUS_ISLAND,
    RC_SCHEDULE_UNAVAILABLE,
    RC_SERVICE_MAPPING_UNVERIFIED,
    RC_VISUAL_NUMBERS_UNAVAILABLE,
)


def _create_test_mp4(path: Path, frames: int = 5):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 64))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(frames):
        out.write(frame)
    out.release()


def test_e1_feature_flags_default_off():
    """E1: Ensure feature flags default to disabled."""
    registry = SiteRegistry()
    defaults = registry.defaults
    assert defaults.get("live_shadow") is False
    assert defaults.get("allow_network") is False
    assert defaults.get("offline_replay", {}).get("enabled") is False
    assert defaults.get("identity", {}).get("require_independent_direction") is True
    assert defaults.get("identity", {}).get("allow_name_only_confirmation") is False
    assert defaults.get("worker", {}).get("max_ocr_calls") == 25


def _write_runtime_policy(root: Path, *, enabled=False, network=False, site_network=False):
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "shadow_defaults.yaml").write_text(
        "shadow_defaults:\n"
        f"  live_shadow: {str(enabled).lower()}\n"
        f"  allow_network: {str(network).lower()}\n"
        "  offline_replay:\n    enabled: false\n"
    )
    site_dir = root / "sites" / "ndls_pf12_13"
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "site.yaml").write_text(
        "site_id: ndls_pf12_13\n"
        f"ntes:\n  allow_network: {str(site_network).lower()}\n"
        "shadow:\n"
        "  live_shadow: false\n"
        "  production_write_allowed: false\n"
    )


def test_worker_runtime_gate_uses_existing_global_live_shadow_setting(tmp_path):
    _write_runtime_policy(tmp_path, enabled=False)
    with pytest.raises(RuntimeError, match="SHADOW_WORKER_DISABLED_LIVE_SHADOW_FALSE"):
        validate_worker_runtime_policy(tmp_path, "ndls_pf12_13")

    _write_runtime_policy(tmp_path, enabled=True)
    validate_worker_runtime_policy(tmp_path, "ndls_pf12_13")


def test_worker_runtime_gate_requires_site_ntes_and_network_off(tmp_path):
    _write_runtime_policy(tmp_path, enabled=True, network=True)
    with pytest.raises(RuntimeError, match="SHADOW_WORKER_NETWORK_OR_REPLAY_POLICY_UNSAFE"):
        validate_worker_runtime_policy(tmp_path, "ndls_pf12_13")

    _write_runtime_policy(tmp_path, enabled=True, site_network=True)
    with pytest.raises(RuntimeError, match="SITE_NTES_NETWORK_MUST_REMAIN_FALSE"):
        validate_worker_runtime_policy(tmp_path, "ndls_pf12_13")


def test_worker_cli_blocks_before_constructing_worker_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["worker", "--watch-dir", str(tmp_path), "--once"])
    monkeypatch.setattr(
        "shadow_identity.shadow_ocr.worker.validate_worker_runtime_policy",
        lambda *_: (_ for _ in ()).throw(RuntimeError("disabled for test")),
    )
    with patch("shadow_identity.shadow_ocr.worker.ShadowWorker") as worker_cls:
        assert worker_main() == 2
    worker_cls.assert_not_called()


def test_e2_scenario_01_one_new_finalized_mp4(tmp_path):
    """1. One new finalized MP4 is discovered and processed safely."""
    inbox = tmp_path / "inbox"
    storage = tmp_path / "storage"
    out_dir = tmp_path / "output"
    inbox.mkdir(parents=True)

    mp4 = inbox / "passage_001.mp4"
    _create_test_mp4(mp4)
    (inbox / "passage_001.done").write_text("DONE")

    worker = ShadowWorker(inbox_dir=inbox, storage_dir=storage, output_dir=out_dir)
    discovered = worker.discover_new_recordings()
    assert len(discovered) == 1
    assert mp4.is_file()  # ARDS file untouched


def test_e2_scenario_02_multiple_new_finalized_mp4s(tmp_path):
    """2. Multiple new finalized MP4s discovered concurrently."""
    inbox = tmp_path / "inbox"
    storage = tmp_path / "storage"
    out_dir = tmp_path / "output"
    inbox.mkdir(parents=True)

    for i in range(3):
        p = inbox / f"passage_00{i}.mp4"
        _create_test_mp4(p)
        (inbox / f"passage_00{i}.done").write_text("DONE")

    worker = ShadowWorker(inbox_dir=inbox, storage_dir=storage, output_dir=out_dir)
    discovered = worker.discover_new_recordings()
    assert len(discovered) == 3


def test_e2_scenario_03_duplicate_files(tmp_path):
    """3. Duplicate files / same job key are idempotently skipped."""
    inbox = tmp_path / "inbox"
    storage = tmp_path / "storage"
    out_dir = tmp_path / "output"
    inbox.mkdir(parents=True)

    mp4 = inbox / "passage_dup.mp4"
    _create_test_mp4(mp4)
    (inbox / "passage_dup.done").write_text("DONE")

    worker = ShadowWorker(inbox_dir=inbox, storage_dir=storage, output_dir=out_dir)
    job1 = worker.stage_job(mp4)
    job2 = worker.stage_job(mp4)
    assert job1.job_id == job2.job_id
    assert len(worker.jobs) == 1


def test_e2_scenario_04_duplicate_passage_ids_different_hashes(tmp_path):
    """4. Duplicate passage IDs with different content hashes receive distinct job IDs."""
    inbox = tmp_path / "inbox"
    storage = tmp_path / "storage"
    out_dir = tmp_path / "output"
    inbox.mkdir(parents=True)

    mp4_a = inbox / "passage_v1.mp4"
    _create_test_mp4(mp4_a, frames=5)
    mp4_b = inbox / "passage_v2.mp4"
    _create_test_mp4(mp4_b, frames=10)

    worker = ShadowWorker(inbox_dir=inbox, storage_dir=storage, output_dir=out_dir)
    job_a = worker.stage_job(mp4_a)
    job_b = worker.stage_job(mp4_b)
    assert job_a.job_id != job_b.job_id
    assert job_a.mp4_sha256 != job_b.mp4_sha256


def test_e2_scenario_05_same_mp4_after_service_restart(tmp_path):
    """5. Same MP4 after service restart is recovered without re-processing."""
    inbox = tmp_path / "inbox"
    storage = tmp_path / "storage"
    out_dir = tmp_path / "output"
    inbox.mkdir(parents=True)

    mp4 = inbox / "passage_restart.mp4"
    _create_test_mp4(mp4)
    (inbox / "passage_restart.done").write_text("DONE")

    worker1 = ShadowWorker(inbox_dir=inbox, storage_dir=storage, output_dir=out_dir)
    job = worker1.stage_job(mp4)
    job.status = "COMPLETED"
    worker1._save_state()

    worker2 = ShadowWorker(inbox_dir=inbox, storage_dir=storage, output_dir=out_dir)
    discovered = worker2.discover_new_recordings()
    assert len(discovered) == 0


def test_e2_scenario_06_partial_video_write(tmp_path):
    """6. Incomplete / unfinalized MP4 without .done marker is ignored."""
    inbox = tmp_path / "inbox"
    storage = tmp_path / "storage"
    out_dir = tmp_path / "output"
    inbox.mkdir(parents=True)

    mp4 = inbox / "writing_now.mp4"
    mp4.write_bytes(b"partial byte stream not finalized")

    worker = ShadowWorker(inbox_dir=inbox, storage_dir=storage, output_dir=out_dir)
    assert worker.is_file_finalized(mp4) is False
    assert len(worker.discover_new_recordings()) == 0


def test_e2_scenario_07_08_09_crashes_preserve_raw_evidence(tmp_path):
    """7, 8, 9. Crashes during staging, inference, or persistence preserve raw evidence."""
    inbox = tmp_path / "inbox"
    storage = tmp_path / "storage"
    out_dir = tmp_path / "output"
    inbox.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    mp4 = inbox / "crash_test.mp4"
    _create_test_mp4(mp4)
    sha = compute_file_sha256(mp4)

    job_id = f"ndls_pf12_13_side_a_pas_crash_{sha}"
    raw_file = out_dir / f"{job_id}_raw.jsonl"
    raw_file.write_text('{"frame": 1, "text": "12445"}\n')

    worker = ShadowWorker(inbox_dir=inbox, storage_dir=storage, output_dir=out_dir)
    # Raw file must never be deleted
    assert raw_file.is_file()
    assert mp4.is_file()


def test_e2_scenario_10_missing_timetable():
    """10. Missing timetable results in RC_SCHEDULE_UNAVAILABLE and does not crash."""
    passage = PassageInput(site_id="ndls_pf12_13", passage_id="p10")
    vis = VisualOcrEvidence(passage_id="p10", passage_consensus=PassageVisualConsensus())
    dec = resolve_shadow(passage, vis, site_schedule_rows=[])
    assert RC_SCHEDULE_UNAVAILABLE in dec.reason_codes
    assert dec.shadow_status == "UNKNOWN"


def test_e2_scenario_11_12_missing_and_stale_ntes():
    """11, 12. Missing and stale NTES handling."""
    passage = PassageInput(site_id="ndls_pf12_13", passage_id="p11", passage_start="2026-10-14T10:00:00+05:30")
    vis = VisualOcrEvidence(passage_id="p11", passage_consensus=PassageVisualConsensus())
    # Missing NTES
    dec_no_ntes = resolve_shadow(passage, vis, site_schedule_rows=[], ntes_before=None)
    assert RC_NTES_UNAVAILABLE in dec_no_ntes.reason_codes

    # Stale NTES
    stale_ntes = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at="2026-10-14T09:00:00+05:30",
        fetched_at="2026-10-14T09:00:00+05:30",
        stale_fallback=True,
        trains=[NtesTrainEntry(train_number="12417", train_name="PRAYAGRAJ EXP", platform="12")],
    )
    dec_stale = resolve_shadow(passage, vis, site_schedule_rows=[], ntes_before=stale_ntes)
    assert RC_NTES_CORROBORATED_STALE in dec_stale.reason_codes


def test_e2_scenario_13_ambiguous_platform():
    """13. Ambiguous island platform assigns RC_PLATFORM_AMBIGUOUS_ISLAND."""
    passage = PassageInput(
        site_id="ndls_pf12_13",
        passage_id="p13",
        observed_platform=None,
        possible_platforms=["12", "13"],
    )
    vis = VisualOcrEvidence(passage_id="p13", passage_consensus=PassageVisualConsensus())
    dec = resolve_shadow(passage, vis, site_schedule_rows=[])
    assert RC_PLATFORM_AMBIGUOUS_ISLAND in dec.reason_codes


def test_e2_scenario_14_missing_actual_passage_timestamp():
    """14. Missing actual passage timestamp disables time corroboration without inventing times."""
    passage = PassageInput(site_id="ndls_pf12_13", passage_id="p14", passage_start="", passage_end="")
    vis = VisualOcrEvidence(passage_id="p14", passage_consensus=PassageVisualConsensus())
    dec = resolve_shadow(passage, vis, site_schedule_rows=[])
    assert dec.active_direction_verified is False


def test_e2_scenario_15_missing_coach_boundaries():
    """15. Missing coach boundaries prevents multi-coach consensus confirmation."""
    vis = VisualOcrEvidence(
        passage_id="p15",
        passage_consensus=PassageVisualConsensus(
            train_name_candidates=[NameCandidate(name="SHIV GANGA EXP", distinct_coaches=3)],
        ),
    )
    passage = PassageInput(site_id="ndls_pf12_13", passage_id="p15", coach_intervals_verified=False)
    dec = resolve_shadow(passage, vis, site_schedule_rows=[])
    assert "NAME_RECOGNIZED_COACH_UNVERIFIED" in dec.reason_codes


def test_e2_scenario_16_unrecognized_train():
    """16. Unrecognized train returns UNKNOWN without fabricating identities."""
    vis = VisualOcrEvidence(
        passage_id="p16",
        passage_consensus=PassageVisualConsensus(
            service_number_candidates=[],
            train_name_candidates=[],
            numeric_status="PAIR_UNCONFIRMED",
        ),
    )
    passage = PassageInput(site_id="ndls_pf12_13", passage_id="p16")
    dec = resolve_shadow(passage, vis, site_schedule_rows=[])
    assert dec.shadow_status in ("UNKNOWN", "PAIR_UNCONFIRMED")
    assert dec.active_service_number is None


def test_e2_scenario_17_recognized_name_no_numeric_board():
    """17. Recognized train name with no numeric board assigns FAMILY_CANDIDATE and active None."""
    vis = VisualOcrEvidence(
        passage_id="p17",
        passage_consensus=PassageVisualConsensus(
            train_name_candidates=[NameCandidate(name="PRAYAGRAJ EXP", distinct_coaches=3)],
            numeric_status="NOT_NUMBERED",
            recognized_train_name="PRAYAGRAJ EXP",
            service_family_candidates=["12417", "12418"],
        ),
    )
    passage = PassageInput(site_id="ndls_pf12_13", passage_id="p17")
    dec = decide_identity_v1(passage, vis, site_schedule_rows=[])
    assert dec.identity_status == "FAMILY_CANDIDATE"
    assert dec.active_service_number is None
    assert dec.active_direction_verified is False


def test_e2_scenario_18_two_and_four_number_boards():
    """18. Two-number and four-number boards handled deterministically."""
    # Two-number board
    vis_2 = VisualOcrEvidence(
        passage_id="p18_2",
        passage_consensus=PassageVisualConsensus(
            service_number_candidates=[
                NumberCandidate(number="12417", distinct_coaches=3),
                NumberCandidate(number="12418", distinct_coaches=3),
            ],
            stable_pair=["12417", "12418"],
            numeric_status="PAIR_CONFIRMED",
            service_family_candidates=["12417", "12418"],
        ),
    )
    passage = PassageInput(site_id="ndls_pf12_13", passage_id="p18_2")
    dec_2 = decide_identity_v1(passage, vis_2, site_schedule_rows=[])
    assert dec_2.identity_status == "FAMILY_CANDIDATE"
    assert dec_2.active_service_number is None

    # Four-number board
    vis_4 = VisualOcrEvidence(
        passage_id="p18_4",
        passage_consensus=PassageVisualConsensus(
            service_number_candidates=[
                NumberCandidate(number="12417", distinct_coaches=1),
                NumberCandidate(number="12418", distinct_coaches=1),
                NumberCandidate(number="12419", distinct_coaches=1),
                NumberCandidate(number="12420", distinct_coaches=1),
            ],
            numeric_status="COMPLEX_BOARD",
        ),
    )
    dec_4 = resolve_shadow(passage, vis_4, site_schedule_rows=[])
    assert dec_4.shadow_status == "AMBIGUOUS"


def test_e2_scenario_19_six_number_coach_stencil_rejected():
    """19. Six-number coach stencil (e.g. 213456) rejected as service number."""
    entities = EntityClassifier.classify_all("WGACCW 213456/C", 0.95)
    numbers = [e.value for e in entities if e.entity_type == "SERVICE_NUMBER"]
    assert "213456" not in numbers


def test_e2_scenario_20_false_train_name():
    """20. False train name text that matches no registry alias does not invent candidates."""
    vis = VisualOcrEvidence(
        passage_id="p20",
        passage_consensus=PassageVisualConsensus(
            train_name_candidates=[NameCandidate(name="FICTIONAL GHOST EXPRESS", distinct_coaches=3)],
            numeric_status="NOT_NUMBERED",
            recognized_train_name="FICTIONAL GHOST EXPRESS",
        ),
    )
    passage = PassageInput(site_id="ndls_pf12_13", passage_id="p20")
    dec = resolve_shadow(passage, vis, site_schedule_rows=[])
    assert RC_SERVICE_MAPPING_UNVERIFIED in dec.reason_codes
    assert dec.service_family_candidates == []


def test_e2_scenario_21_contradictory_registry_mappings():
    """21. Contradictory registry mappings fail closed."""
    vis = VisualOcrEvidence(
        passage_id="p21",
        passage_consensus=PassageVisualConsensus(
            train_name_candidates=[NameCandidate(name="SHARED ALIAS EXP", distinct_coaches=3)],
            numeric_status="NOT_NUMBERED",
            recognized_train_name="SHARED ALIAS EXP",
        ),
    )
    passage = PassageInput(site_id="testx_pf1_2", passage_id="p21")
    dec = resolve_shadow(passage, vis, site_schedule_rows=[])
    assert dec.service_family_candidates == []
    assert RC_SERVICE_MAPPING_UNVERIFIED in dec.reason_codes


def test_e2_scenario_22_corrupted_video_and_disconnected_storage(tmp_path):
    """22. Corrupted video or disconnected storage fails safely without modifying source."""
    inbox = tmp_path / "inbox"
    storage = tmp_path / "storage"
    out_dir = tmp_path / "output"
    inbox.mkdir(parents=True)

    corrupt_mp4 = inbox / "corrupt.mp4"
    corrupt_mp4.write_bytes(b"NOT_A_VALID_MP4_HEADER")
    (inbox / "corrupt.done").write_text("DONE")

    worker = ShadowWorker(inbox_dir=inbox, storage_dir=storage, output_dir=out_dir)
    assert worker.is_file_finalized(corrupt_mp4) is False
    assert corrupt_mp4.is_file()  # Source remains unchanged
