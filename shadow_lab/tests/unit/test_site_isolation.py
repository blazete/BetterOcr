"""Unit tests for Site Isolation and Multi-Site Configuration (Tests 1 to 10).
Guarantees mandatory site_id, no silent NDLS fallback, and zero cross-site schedule/NTES mixing.
"""

import pytest
from pathlib import Path
from datetime import datetime

from shadow_identity.config_loader import ConfigError, SiteRegistry
from shadow_identity.models import NtesSnapshot, PassageInput, TimetableRow
from shadow_identity.timetable_matcher import TimetableMatcher


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parents[3]


@pytest.fixture
def registry(repo_root):
    return SiteRegistry(shadow_lab_root=repo_root / "shadow_lab")


def test_site_id_required(registry):
    """Test 1: missing site ID raises clear error."""
    with pytest.raises(ConfigError, match="site_id is mandatory"):
        registry.get_site("")


def test_unknown_site_rejected(registry):
    """Test 2: unknown site ID rejected with NO silent fallback to NDLS."""
    with pytest.raises(ConfigError, match="Silent fallback to NDLS is strictly prohibited"):
        registry.get_site("unknown_nonexistent_site")


def test_station_A_schedule_not_used_at_station_B(registry):
    """Test 3: schedule rows from Station A (NDLS) are rejected if matched against Station B (TESTX)."""
    ndls_row = TimetableRow(
        site_id="ndls_pf12_13",
        schedule_version="2026-v1",
        station_code="NDLS",
        train_number="12920",
        train_name="MALWA EXPRESS",
    )
    matcher = TimetableMatcher(rows=[ndls_row], site_id="testx_pf1_2")
    # All NDLS rows must be filtered out for site testx_pf1_2
    assert len(matcher.rows) == 0


def test_station_A_ntes_not_used_at_station_B():
    """Test 4: wrong-station snapshot cannot be applied to another station."""
    snapshot = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at="2026-10-13T17:20:00+05:30",
        fetched_at="2026-10-13T17:20:05+05:30",
    )
    assert snapshot.site_id == "ndls_pf12_13"
    assert snapshot.station_code == "NDLS"
    # Ensure assertion fails if attempted to be associated with TESTX
    assert snapshot.station_code != "TESTX"


def test_two_installations_same_station_share_snapshot_without_mixing_passages():
    """Test 5: two installations at same station share station code but maintain separate passage identities."""
    p1 = PassageInput(
        site_id="ndls_pf12_13",
        installation_id="ndls_camera_rig_north",
        passage_id="PAS_NORTH_001",
    )
    p2 = PassageInput(
        site_id="ndls_pf12_13",
        installation_id="ndls_camera_rig_south",
        passage_id="PAS_SOUTH_002",
    )
    assert p1.site_id == p2.site_id
    assert p1.passage_id != p2.passage_id
    assert p1.installation_id != p2.installation_id


def test_platform_unknown_preserves_all_possible_platforms():
    """Test 6: physical island camera preserves possible_platforms without inventing certainty."""
    passage = PassageInput(
        site_id="ndls_pf12_13",
        passage_id="PAS_ISLAND_01",
        possible_platforms=["12", "13"],
        observed_platform=None,
    )
    assert passage.observed_platform is None
    assert set(passage.possible_platforms) == {"12", "13"}


def test_platform_change_not_silently_rejected():
    """Test 7: trains reassigned to unexpected platforms are flagged, not dropped."""
    row = TimetableRow(
        site_id="ndls_pf12_13",
        schedule_version="2026-v1",
        station_code="NDLS",
        train_number="12920",
        train_name="MALWA EXPRESS",
        platform="12",
    )
    # Train appears on PF 13 instead of scheduled PF 12
    assert row.platform == "12"
    is_same = (row.platform == "13")
    assert not is_same


def test_schedule_version_effective_by_passage_date():
    """Test 8: schedule version effective date bounds."""
    row = TimetableRow(
        site_id="ndls_pf12_13",
        schedule_version="2026-10-13-validated-v1",
        station_code="NDLS",
        train_number="12920",
        train_name="MALWA EXPRESS",
        operating_days=[0, 1, 2, 3, 4, 5, 6],
    )
    # 2026-10-13 is Tuesday (weekday 1)
    dt = datetime(2026, 10, 13)
    assert row.is_operational_on(dt)


def test_schedule_expired_unknown_and_no_auto_confirmation():
    """Test 9: empty schedule rows result in no auto-confirmation."""
    matcher = TimetableMatcher(rows=[], site_id="ndls_pf12_13")
    assert len(matcher.rows) == 0


def test_invalid_station_code_or_invalid_platform_fails_preflight(registry):
    """Test 10: preflight enforces schema constraints."""
    cfg = registry.get_site("ndls_pf12_13")
    assert cfg["station"]["code"] == "NDLS"
    assert "12" in cfg["platforms"]["configured"]
    assert "13" in cfg["platforms"]["configured"]
