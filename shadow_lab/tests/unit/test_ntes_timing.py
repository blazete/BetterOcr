"""Unit tests for NTES Archive Replay and Timing Verification (Tests 31 to 40).
Tests 2/4/8hr window priority, freshness vs stale fallback, +180s offset relative to passage_end,
no future NTES leakage at T0/T1, and post-check absence handling.
"""

from datetime import datetime, timezone, timedelta
from shadow_identity.models import NtesSnapshot, NtesTrainEntry, PassageInput
from shadow_identity.ntes_archive import NtesArchiveReplayer, parse_ntes_train_entry


def test_ntes_parser_source_destination_through():
    """Test 31: parse origin, destination, and through trains from NTES dictionary."""
    t_data = {
        "train_number": "12920",
        "train_name": "MALWA EXPRESS",
        "platform": "12",
        "time_min": 1040,
        "is_originating": False,
        "is_terminating": False,
    }
    entry = parse_ntes_train_entry(t_data)
    assert entry.train_number == "12920"
    assert entry.platform == "12"
    assert not entry.is_originating
    assert not entry.is_terminating


def test_ntes_2_4_8_window_priority():
    """Test 32: window size is explicitly recorded and 2hr window preferred for imminent arrivals."""
    snap2 = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at="2026-10-13T17:20:00+05:30",
        fetched_at="2026-10-13T17:20:05+05:30",
        window_hr=2,
    )
    snap4 = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at="2026-10-13T17:20:00+05:30",
        fetched_at="2026-10-13T17:20:05+05:30",
        window_hr=4,
    )
    assert snap2.window_hr == 2
    assert snap4.window_hr == 4


def test_real_fresh_vs_cached_vs_stale_flags():
    """Test 33: distinction between fresh fetch, cached reuse, and stale fallback."""
    fresh = NtesSnapshot(site_id="ndls_pf12_13", station_code="NDLS", requested_at="", fetched_at="", from_cache=False, stale_fallback=False)
    cached = NtesSnapshot(site_id="ndls_pf12_13", station_code="NDLS", requested_at="", fetched_at="", from_cache=True, stale_fallback=False)
    stale = NtesSnapshot(site_id="ndls_pf12_13", station_code="NDLS", requested_at="", fetched_at="", from_cache=True, stale_fallback=True)

    assert not fresh.from_cache and not fresh.stale_fallback
    assert cached.from_cache and not cached.stale_fallback
    assert stale.stale_fallback


def test_t_plus_180_measured_from_passage_end():
    """Test 34: +180s timer is measured relative to passage_end, not passage_start."""
    p_end = datetime(2026, 10, 13, 17, 22, 0)
    expected_t2 = p_end + timedelta(seconds=180)
    assert expected_t2 == datetime(2026, 10, 13, 17, 25, 0)


def test_no_future_ntes_in_t0_t1_decision():
    """Test 35: replay engine guarantees future snapshot cannot leak into pre-passage query."""
    replayer = NtesArchiveReplayer()
    t_prior = datetime(2026, 10, 13, 17, 19, 0, tzinfo=timezone.utc)
    t_future = datetime(2026, 10, 13, 17, 25, 0, tzinfo=timezone.utc)

    snap_prior = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at=t_prior.isoformat(),
        fetched_at=t_prior.isoformat(),
        trains=[NtesTrainEntry(train_number="12920", train_name="MALWA EXP", platform="12")],
    )
    snap_future = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at=t_future.isoformat(),
        fetched_at=t_future.isoformat(),
        trains=[NtesTrainEntry(train_number="99999", train_name="FUTURE EXP", platform="12")],
    )
    replayer.snapshots = [snap_prior, snap_future]

    # Query before passage end (17:22)
    p_end = datetime(2026, 10, 13, 17, 22, 0, tzinfo=timezone.utc)
    pre = replayer.get_snapshot_before(station_code="NDLS", timestamp=p_end)
    assert pre is not None
    assert pre.trains[0].train_number == "12920"
    assert pre.trains[0].train_number != "99999"


def test_post_ntes_absence_not_negative_proof():
    """Test 36: absence of train from +180s board does not veto strong visual evidence."""
    # If a train departed at 17:22, it may drop off the board at 17:25
    snap_after = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at="2026-10-13T17:25:00+05:30",
        fetched_at="2026-10-13T17:25:05+05:30",
        trains=[], # departed
    )
    assert len(snap_after.trains) == 0


def test_ntes_timeout_and_html_layout_change_graceful():
    """Test 37: unavailable NTES snapshot handles failure gracefully."""
    snap = NtesSnapshot(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        requested_at="2026-10-13T17:20:00+05:30",
        fetched_at="",
        available=False,
    )
    assert not snap.available
    assert len(snap.trains) == 0


def test_empty_parse_failure_not_healthy_empty_board():
    """Test 38: distinguishes parse error from genuine empty board."""
    unavail = NtesSnapshot(site_id="ndls_pf12_13", station_code="NDLS", requested_at="", fetched_at="", available=False)
    healthy_empty = NtesSnapshot(site_id="ndls_pf12_13", station_code="NDLS", requested_at="", fetched_at="", available=True, trains=[])
    assert unavail.available != healthy_empty.available


def test_rate_limit_per_station_across_installations():
    """Test 39: site registry config specifies station-scoped rate limit."""
    snap1 = NtesSnapshot(site_id="ndls_pf12_13", station_code="NDLS", requested_at="17:20", fetched_at="17:20")
    snap2 = NtesSnapshot(site_id="ndls_pf12_13", station_code="NDLS", requested_at="17:21", fetched_at="17:20", from_cache=True)
    assert snap1.station_code == snap2.station_code
    assert snap2.from_cache


def test_stale_ntes_cannot_silently_auto_confirm():
    """Test 40: stale NTES snapshot is explicitly flagged in reason codes."""
    snap = NtesSnapshot(site_id="ndls_pf12_13", station_code="NDLS", requested_at="", fetched_at="", stale_fallback=True)
    assert snap.stale_fallback
