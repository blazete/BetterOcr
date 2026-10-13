"""Integration tests for Two-Station Replay and End-to-End Pipeline.
Validates NDLS vs TESTX synthetic station with identical clock times and zero cross-site leakage.
"""

from pathlib import Path
import pytest

from shadow_identity.candidate_resolver import resolve_shadow
from shadow_identity.config_loader import SiteRegistry
from shadow_identity.models import (
    NameCandidate,
    NtesSnapshot,
    NtesTrainEntry,
    NumberCandidate,
    PassageInput,
    PassageVisualConsensus,
    VisualOcrEvidence,
)
from shadow_identity.timetable_importer import TimetableImporter


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parents[3]


@pytest.fixture
def registry(repo_root):
    return SiteRegistry(shadow_lab_root=repo_root / "shadow_lab")


def test_two_station_synthetic_isolation(registry, repo_root):
    """
    Two-station synthetic test: NDLS PF12/13 vs TESTX PF1/2.
    Both have trains scheduled around 22:00.
    NDLS at 22:00 is train 12230 LUCKNOW MAIL.
    TESTX at 22:00 is train 12230 REGIONAL OVERLAP TEST EXP.
    Ensures zero cross-site leakage, site-specific candidate sets, and correct isolation.
    """
    # 1. Load NDLS timetable
    ndls_cfg = registry.get_site("ndls_pf12_13")
    ndls_sched = TimetableImporter.import_schedule(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        csv_path=str(repo_root / ndls_cfg["schedule"]["path"]),
        version=ndls_cfg["schedule"]["version"],
    )

    # 2. Load TESTX timetable
    testx_cfg = registry.get_site("testx_pf1_2")
    testx_sched = TimetableImporter.import_schedule(
        site_id="testx_pf1_2",
        station_code="TESTX",
        csv_path=str(repo_root / testx_cfg["schedule"]["path"]),
        version=testx_cfg["schedule"]["version"],
    )

    # Passage at 22:00
    passage_ndls = PassageInput(
        site_id="ndls_pf12_13",
        passage_id="P_NDLS_2200",
        passage_start="2026-10-13T22:00:00+05:30",
        passage_end="2026-10-13T22:02:00+05:30",
        possible_platforms=["12", "13"],
    )
    passage_testx = PassageInput(
        site_id="testx_pf1_2",
        passage_id="P_TESTX_2200",
        passage_start="2026-10-13T22:00:00+05:30",
        passage_end="2026-10-13T22:02:00+05:30",
        possible_platforms=["1", "2"],
    )

    # Visual evidence: train name "LUCKNOW MAIL"
    evidence_ndls = VisualOcrEvidence(
        passage_id="P_NDLS_2200",
        passage_consensus=PassageVisualConsensus(
            train_name_candidates=[NameCandidate(name="LUCKNOW MAIL", distinct_coaches=4)],
            numeric_status="NOT_NUMBERED",
        ),
    )

    # Visual evidence for TESTX: train name "REGIONAL OVERLAP TEST EXP"
    evidence_testx = VisualOcrEvidence(
        passage_id="P_TESTX_2200",
        passage_consensus=PassageVisualConsensus(
            train_name_candidates=[NameCandidate(name="REGIONAL OVERLAP TEST EXP", distinct_coaches=4)],
            numeric_status="NOT_NUMBERED",
        ),
    )

    # Resolver at NDLS
    dec_ndls = resolve_shadow(passage_ndls, evidence_ndls, ndls_sched.rows)
    assert dec_ndls.site_id == "ndls_pf12_13"
    assert dec_ndls.shadow_final_number == "12230"
    assert "LUCKNOW MAIL" in dec_ndls.candidate_services[0].train_name

    # Resolver at TESTX
    dec_testx = resolve_shadow(passage_testx, evidence_testx, testx_sched.rows)
    assert dec_testx.site_id == "testx_pf1_2"
    assert dec_testx.shadow_final_number == "12230"
    assert dec_testx.candidate_services[0].train_name == "REGIONAL OVERLAP TEST EXP"

    # Ensure cross-site leak does not occur: TESTX with NDLS train name cannot resolve
    dec_leak_check = resolve_shadow(passage_testx, evidence_ndls, testx_sched.rows)
    assert dec_leak_check.shadow_final_number is None or dec_leak_check.shadow_status == "UNKNOWN"
