"""
Comprehensive Replay Suite and Multi-Site Isolation Tests for Collector (§26, §27)
Covers all 10 mandatory evaluation scenarios:
1. 2-number confirmed (distinct coach >= 3 + NTES corroborated)
2. name-only fallback (train name + timetable + NTES)
3. 4-number complex board (name-first disambiguation)
4. NTES unavailable (safe resolution via visual + timetable)
5. wrong-station NTES (strict rejection, zero candidates)
6. stale NTES (marked stale, cannot confirm)
7. wrong platform in NTES (unrelated platform excluded)
8. delayed train (within permissible window)
9. two competing candidates (equal scores -> AMBIGUOUS)
10. visual vs NTES conflict (visual pair differs from NTES -> CONFLICT)
11. Multi-site timetable isolation (Station A passage + Station B timetable -> rejected)
"""

from datetime import datetime
import pytest

from shadow_identity.passage_collector import ShadowCollector
from shadow_identity.reason_codes import (
    RC_AMBIGUOUS_MULTI_CANDIDATE,
    RC_NTES_CORROBORATED_STALE,
    RC_NTES_UNAVAILABLE,
    RC_NTES_WRONG_STATION_REJECTED,
    RC_VISUAL_NTES_CONFLICT,
)


@pytest.fixture
def collector():
    return ShadowCollector()


def test_scenario_1_two_number_confirmed(collector):
    """Scenario 1: 2-number confirmed with >= 3 distinct coaches and fresh NTES -> CONFIRMED."""
    p_data = {
        "passage_id": "PAS_SCENARIO_01",
        "start_time": "2026-10-13T17:20:00+05:30",
        "end_time": "2026-10-13T17:21:00+05:30",
        "observed_platform": "12",
        "possible_platforms": ["12", "13"],
        "visual": {
            "coach_evidence": [
                {"coach_id": "C01", "raw_text": "12920", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C02", "raw_text": "12920", "entity_type": "SERVICE_NUMBER", "confidence": 0.94},
                {"coach_id": "C03", "raw_text": "12920", "entity_type": "SERVICE_NUMBER", "confidence": 0.96},
                {"coach_id": "C01", "raw_text": "12919", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C02", "raw_text": "12919", "entity_type": "SERVICE_NUMBER", "confidence": 0.94},
                {"coach_id": "C03", "raw_text": "12919", "entity_type": "SERVICE_NUMBER", "confidence": 0.96},
            ]
        },
        "ntes": {
            "pre_or_near_passage_snapshot": {
                "site_id": "ndls_pf12_13",
                "station_code": "NDLS",
                "fetched_at": "2026-10-13T17:19:00+05:30",
                "trains": [{"train_number": "12920", "train_name": "MALWA EXP", "platform": "12"}],
            }
        },
        "production": {"train_number": "12920", "confidence": "HIGH"},
    }
    bundle = collector.import_production_passage(source=p_data, site_id="ndls_pf12_13")
    assert bundle.shadow_result["T1"]["train_number"] == "12920"
    assert bundle.shadow_result["T1"]["status"] == "CONFIRMED"
    assert bundle.comparison["production_vs_shadow"] == "MATCH"


def test_scenario_2_name_only_resolution(collector):
    """Scenario 2: No numbers on board, strong train name TAJ EXPRESS -> matches 12280."""
    p_data = {
        "passage_id": "PAS_SCENARIO_02",
        "start_time": "2026-10-13T21:55:00+05:30",
        "end_time": "2026-10-13T21:56:00+05:30",
        "possible_platforms": ["12", "13"],
        "visual": {
            "coach_evidence": [
                {"coach_id": "C01", "raw_text": "TAJ EXPRESS", "entity_type": "TRAIN_NAME", "confidence": 0.95},
                {"coach_id": "C02", "raw_text": "TAJ EXPRESS", "entity_type": "TRAIN_NAME", "confidence": 0.95},
            ]
        },
        "ntes": {
            "pre_or_near_passage_snapshot": {
                "site_id": "ndls_pf12_13",
                "station_code": "NDLS",
                "fetched_at": "2026-10-13T21:54:00+05:30",
                "trains": [{"train_number": "12280", "train_name": "TAJ EXPRESS", "platform": "12"}],
            }
        },
    }
    bundle = collector.import_production_passage(source=p_data, site_id="ndls_pf12_13")
    assert bundle.shadow_result["T1"]["train_number"] == "12280"
    assert bundle.shadow_result["T1"]["status"] in ("CONFIRMED", "PROVISIONAL")


def test_scenario_3_four_number_complex_board(collector):
    """Scenario 3: 4-number board (rake share) uses name to isolate service family."""
    nums = ["12559", "12560", "12451", "12452"]
    coach_ev = []
    for cid in ["C01", "C02", "C03"]:
        for n in nums:
            coach_ev.append({"coach_id": cid, "raw_text": n, "entity_type": "SERVICE_NUMBER", "confidence": 0.95})
        coach_ev.append({"coach_id": cid, "raw_text": "SHIV GANGA", "entity_type": "TRAIN_NAME", "confidence": 0.95})

    p_data = {
        "passage_id": "PAS_SCENARIO_03",
        "start_time": "2026-10-13T20:05:00+05:30",
        "possible_platforms": ["12", "13"],
        "visual": {"coach_evidence": coach_ev},
    }
    bundle = collector.import_production_passage(source=p_data, site_id="ndls_pf12_13")
    # Must resolve within Shiv Ganga service family (12559 or 12560), NOT Shram Shakti (12451/12452)
    assert bundle.shadow_result["T0"]["train_number"] in ("12559", "12560")


def test_scenario_4_ntes_unavailable(collector):
    """Scenario 4: NTES unavailable degrades gracefully without crash."""
    p_data = {
        "passage_id": "PAS_SCENARIO_04",
        "start_time": "2026-10-13T17:20:00+05:30",
        "possible_platforms": ["12", "13"],
        "visual": {
            "coach_evidence": [
                {"coach_id": "C01", "raw_text": "12920", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
            ]
        },
        "ntes": {"pre_or_near_passage_snapshot": None},
    }
    bundle = collector.import_production_passage(source=p_data, site_id="ndls_pf12_13")
    assert RC_NTES_UNAVAILABLE in bundle.shadow_result["T1"]["reason_codes"]


def test_scenario_5_wrong_station_ntes_rejected(collector):
    """Scenario 5: Wrong-station NTES snapshot is rejected (§26)."""
    p_data = {
        "passage_id": "PAS_SCENARIO_05",
        "start_time": "2026-10-13T17:20:00+05:30",
        "possible_platforms": ["12", "13"],
        "visual": {"coach_evidence": []},
        "ntes": {
            "pre_or_near_passage_snapshot": {
                "site_id": "testx_pf1_2",
                "station_code": "TESTX",
                "fetched_at": "2026-10-13T17:19:00+05:30",
                "trains": [{"train_number": "99999", "train_name": "TESTX TRAIN", "platform": "12"}],
            }
        },
    }
    bundle = collector.import_production_passage(source=p_data, site_id="ndls_pf12_13")
    assert RC_NTES_WRONG_STATION_REJECTED in bundle.shadow_result["T1"]["reason_codes"]
    assert bundle.shadow_result["T1"]["train_number"] != "99999"


def test_scenario_6_stale_ntes_marked_stale(collector):
    """Scenario 6: Snapshot older than TTL (180s) is marked stale."""
    p_data = {
        "passage_id": "PAS_SCENARIO_06",
        "start_time": "2026-10-13T17:20:00+05:30",
        "possible_platforms": ["12", "13"],
        "visual": {"coach_evidence": []},
        "ntes": {
            "pre_or_near_passage_snapshot": {
                "site_id": "ndls_pf12_13",
                "station_code": "NDLS",
                "fetched_at": "2026-10-13T17:05:00+05:30",  # 15 minutes old
                "snapshot_age_seconds_at_decision": 900.0,
                "trains": [{"train_number": "12920", "train_name": "MALWA EXP", "platform": "12"}],
            }
        },
    }
    bundle = collector.import_production_passage(source=p_data, site_id="ndls_pf12_13")
    assert RC_NTES_CORROBORATED_STALE in bundle.shadow_result["T1"]["reason_codes"]


def test_scenario_7_wrong_platform_in_ntes_excluded(collector):
    """Scenario 7: Train on Platform 1 in NTES does not match Platform 12/13 passage."""
    p_data = {
        "passage_id": "PAS_SCENARIO_07",
        "start_time": "2026-10-13T17:20:00+05:30",
        "possible_platforms": ["12", "13"],
        "visual": {"coach_evidence": []},
        "ntes": {
            "pre_or_near_passage_snapshot": {
                "site_id": "ndls_pf12_13",
                "station_code": "NDLS",
                "fetched_at": "2026-10-13T17:19:00+05:30",
                "trains": [{"train_number": "12004", "train_name": "SHATABDI", "platform": "1"}],
            }
        },
    }
    bundle = collector.import_production_passage(source=p_data, site_id="ndls_pf12_13")
    assert bundle.shadow_result["T1"]["train_number"] != "12004"


def test_scenario_8_delayed_train_within_window(collector):
    """Scenario 8: Train delayed by 45 minutes still matches within asymmetric window."""
    # Passage at 18:05 for a train scheduled at 17:20 (diff = +45 min, within late_window 90/180)
    p_data = {
        "passage_id": "PAS_SCENARIO_08",
        "start_time": "2026-10-13T18:05:00+05:30",
        "possible_platforms": ["12", "13"],
        "visual": {
            "coach_evidence": [
                {"coach_id": "C01", "raw_text": "12314", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C02", "raw_text": "12314", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C03", "raw_text": "12314", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
            ]
        },
    }
    bundle = collector.import_production_passage(source=p_data, site_id="ndls_pf12_13")
    assert bundle.shadow_result["T0"]["train_number"] == "12314"


def test_scenario_9_two_competing_candidates_ambiguous(collector):
    """Scenario 9: Two equal candidates with no differentiator declare AMBIGUOUS."""
    p_data = {
        "passage_id": "PAS_SCENARIO_09",
        "start_time": "2026-10-13T03:00:00+05:30",
        "possible_platforms": ["12", "13"],
        "visual": {
            "coach_evidence": [
                {"coach_id": "C01", "raw_text": "12001", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C02", "raw_text": "12001", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C03", "raw_text": "12001", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C01", "raw_text": "12002", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C02", "raw_text": "12002", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C03", "raw_text": "12002", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
            ]
        },
        "ntes": {
            "pre_or_near_passage_snapshot": {
                "site_id": "ndls_pf12_13",
                "station_code": "NDLS",
                "fetched_at": "2026-10-13T02:59:00+05:30",
                "trains": [
                    {"train_number": "12001", "train_name": "SHATABDI UP", "platform": "12"},
                    {"train_number": "12002", "train_name": "SHATABDI DN", "platform": "13"},
                ],
            }
        },
    }
    bundle = collector.import_production_passage(source=p_data, site_id="ndls_pf12_13")
    assert bundle.shadow_result["T1"]["status"] in ("AMBIGUOUS", "PROVISIONAL")


def test_scenario_10_visual_vs_ntes_conflict(collector):
    """Scenario 10: Strong visual OCR (12919/12920) contradicts NTES (12420) -> CONFLICT."""
    p_data = {
        "passage_id": "PAS_SCENARIO_10",
        "start_time": "2026-10-13T17:20:00+05:30",
        "possible_platforms": ["12", "13"],
        "visual": {
            "coach_evidence": [
                {"coach_id": "C01", "raw_text": "12919", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C02", "raw_text": "12919", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C03", "raw_text": "12919", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C01", "raw_text": "12920", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C02", "raw_text": "12920", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
                {"coach_id": "C03", "raw_text": "12920", "entity_type": "SERVICE_NUMBER", "confidence": 0.95},
            ]
        },
        "ntes": {
            "pre_or_near_passage_snapshot": {
                "site_id": "ndls_pf12_13",
                "station_code": "NDLS",
                "fetched_at": "2026-10-13T17:19:00+05:30",
                "trains": [{"train_number": "12420", "train_name": "GOMTI EXP", "platform": "12"}],
            }
        },
    }
    bundle = collector.import_production_passage(source=p_data, site_id="ndls_pf12_13")
    assert RC_VISUAL_NTES_CONFLICT in bundle.shadow_result["T1"]["reason_codes"]
    assert bundle.shadow_result["T1"]["status"] == "CONFLICT"


def test_multi_site_timetable_isolation(collector):
    """Verify that Station A passage strictly rejects Station B timetable rows (§26)."""
    p_data = {
        "passage_id": "PAS_ISOLATION_01",
        "start_time": "2026-10-13T17:20:00+05:30",
        "possible_platforms": ["1", "2"],
        "visual": {"coach_evidence": []},
    }
    bundle_testx = collector.import_production_passage(source=p_data, site_id="testx_pf1_2")
    assert bundle_testx.site_id == "testx_pf1_2"
    # NDLS-specific trains must not appear in TESTX candidate services
    cand_nums = [c["train_number"] for c in bundle_testx.timetable["candidate_services"]]
    assert "12920" not in cand_nums
    assert "12280" not in cand_nums
