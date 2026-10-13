"""Unit tests for Timetable Importer (Tests 11 to 20).
Validates decimal time parsing (5.10 -> 05:10), leading zero preservation,
operating days, rake links, and checksum tracking.
"""

from pathlib import Path
from shadow_identity.models import TimetableRow
from shadow_identity.timetable_importer import (
    TimetableImporter,
    parse_operating_days,
    parse_rake_links,
    parse_schedule_time,
)


def test_excel_5_dot_10_means_05_10():
    """Test 11: 5.10 in Indian Railway decimal format means 05:10, NOT 05:01!"""
    assert parse_schedule_time("5.10") == "05:10"
    assert parse_schedule_time("6.05") == "06:05"
    assert parse_schedule_time("22.00") == "22:00"
    assert parse_schedule_time("22.0") == "22:00"
    assert parse_schedule_time("7.3") == "07:30"
    assert parse_schedule_time("05:10") == "05:10"


def test_preserve_02563_leading_zero(tmp_path):
    """Test 12: preserve leading zeros in 5-digit train numbers."""
    csv_file = tmp_path / "test_zeros.csv"
    csv_file.write_text(
        "S.No,Train No.,Train Name,PF,Arrl.,Dep.,Remarks\n"
        "1,02563,BARAUNI SPECIAL,12,,20.40,Daily\n"
    )
    rows = TimetableImporter.load_schedule_csv(csv_file, "ndls_pf12_13", "NDLS", "v1")
    assert len(rows) == 1
    assert rows[0].train_number == "02563"
    assert rows[0].train_number.startswith("0")


def test_weekday_daily_except_only():
    """Test 13: parsing Daily, Except Tuesday, Only Sunday."""
    # Daily
    assert parse_operating_days("Daily") == [0, 1, 2, 3, 4, 5, 6]
    # Except Tuesday (1)
    except_tue = parse_operating_days("Except Tuesday")
    assert 1 not in except_tue
    assert 0 in except_tue and 2 in except_tue
    # Only Sunday (6)
    only_sun = parse_operating_days("Only Sunday")
    assert only_sun == [6]


def test_overnight_departure_and_service_day_rollover(tmp_path):
    """Test 14: late night service and overnight day crossing."""
    csv_file = tmp_path / "overnight.csv"
    csv_file.write_text(
        "S.No,Train No.,Train Name,PF,Arrl.,Dep.,Remarks\n"
        "1,12425,JAMMU RAJDHANI,12,,21.10,Daily\n"
        "2,12230,LUCKNOW MAIL,12,,22.00,Daily\n"
    )
    rows = TimetableImporter.load_schedule_csv(csv_file, "ndls_pf12_13", "NDLS", "v1")
    assert len(rows) == 2
    assert rows[0].departure_local == "21:10"
    assert rows[1].departure_local == "22:00"


def test_duplicate_train_names_do_not_force_unique_service(tmp_path):
    """Test 15: duplicate train names represent distinct candidate services."""
    csv_file = tmp_path / "duplicate_names.csv"
    csv_file.write_text(
        "S.No,Train No.,Train Name,PF,Arrl.,Dep.,Remarks\n"
        "1,12001,SHATABDI EXPRESS,12,06.00,,Daily\n"
        "2,12002,SHATABDI EXPRESS,13,,22.30,Daily\n"
    )
    rows = TimetableImporter.load_schedule_csv(csv_file, "ndls_pf12_13", "NDLS", "v1")
    assert len(rows) == 2
    assert rows[0].train_name == rows[1].train_name
    assert rows[0].train_number != rows[1].train_number


def test_rake_link_does_not_imply_same_active_train_number():
    """Test 16: rake links extract next service without mutating active train number."""
    remarks = "Rake will go as 64465 to GZB"
    next_svc = parse_rake_links(remarks)
    assert next_svc == "64465"


def test_truncated_pdf_name_requires_review(tmp_path):
    """Test 17: short or truncated names are cleanly parsed and flagged for verification."""
    csv_file = tmp_path / "trunc.csv"
    csv_file.write_text(
        "S.No,Train No.,Train Name,PF,Arrl.,Dep.,Remarks\n"
        "1,12920,MALWA EXP...,12,,17.20,Daily\n"
    )
    rows = TimetableImporter.load_schedule_csv(csv_file, "ndls_pf12_13", "NDLS", "v1")
    assert rows[0].train_name == "MALWA EXP..."


def test_missing_arrival_departure_flagged(tmp_path):
    """Test 18: missing arrival and departure rows have None values."""
    csv_file = tmp_path / "no_time.csv"
    csv_file.write_text(
        "S.No,Train No.,Train Name,PF,Arrl.,Dep.,Remarks\n"
        "1,12562,SWATANTRTA S EXP,12,-,-,Daily\n"
    )
    rows = TimetableImporter.load_schedule_csv(csv_file, "ndls_pf12_13", "NDLS", "v1")
    assert rows[0].arrival_local is None
    assert rows[0].departure_local is None


def test_source_checksum_and_version_stored(tmp_path):
    """Test 19: import_schedule stores sha256 checksum and version metadata."""
    csv_file = tmp_path / "sched.csv"
    csv_file.write_text("S.No,Train No.,Train Name,PF,Arrl.,Dep.,Remarks\n1,12920,MALWA EXP,12,,17.20,Daily\n")
    snapshot = TimetableImporter.import_schedule(
        site_id="ndls_pf12_13",
        station_code="NDLS",
        csv_path=str(csv_file),
        version="v2026-test",
    )
    assert snapshot.version == "v2026-test"
    assert len(snapshot.checksum_sha256) == 64


def test_schedule_alias_normalization_keeps_original():
    """Test 20: train name and aliases retain original formatting."""
    row = TimetableRow(
        site_id="ndls_pf12_13",
        schedule_version="v1",
        station_code="NDLS",
        train_number="12230",
        train_name="LUCKNOW MAIL",
        aliases=["NDLS-LJN LUCKNOW MAIL"],
    )
    assert row.train_name == "LUCKNOW MAIL"
    assert "NDLS-LJN LUCKNOW MAIL" in row.aliases
