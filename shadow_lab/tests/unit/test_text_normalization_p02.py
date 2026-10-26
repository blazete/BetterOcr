"""
Regression tests for Package P0.2: Stop false name matching.
"""
import pytest
from shadow_identity.text_normalization import (
    match_name_similarity,
    match_name_with_aliases,
)


def test_substring_shortcut_removed():
    # 'RAJ' vs 'RAJDHANI EXPRESS' must NOT match (0.0), previously was 1.0
    score = match_name_similarity("RAJ", "RAJDHANI EXPRESS")
    assert score == 0.0, f"Expected 0.0 for 'RAJ' vs 'RAJDHANI EXPRESS', got {score}"

    # Generic short tokens (<4 chars) like 'TAJ' alone without whitelist must return 0.0
    score_taj = match_name_similarity("TAJ", "TAJ EXPRESS")
    assert score_taj == 0.0, f"Expected 0.0 for 'TAJ' alone vs 'TAJ EXPRESS', got {score_taj}"


def test_meaningful_token_similarity_after_generic_stripping():
    # 'SACHKHAND EXP' vs 'SACHKHAND EXPRESS' => 1.0 after suffix normalization
    score = match_name_similarity("SACHKHAND EXP", "SACHKHAND EXPRESS")
    assert score == 1.0, f"Expected 1.0 for 'SACHKHAND EXP' vs 'SACHKHAND EXPRESS', got {score}"

    # 'LUCKNOW MAIL' vs 'LUCKNOW EXPRESS' -> core is {'LUCKNOW'} vs {'LUCKNOW'}
    # Wait, in the spec:
    # generic = {EXP, EXPRESS, SF, SUPERFAST, SPL, SPECIAL, MAIL}
    # Both have core {'LUCKNOW'}, intersection/union = 1.0
    # But for distinct train names with different cores:
    score_diff = match_name_similarity("SWARAJ EXPRESS", "SARVODAYA EXPRESS")
    assert score_diff == 0.0, f"Expected 0.0, got {score_diff}"


def test_whitelisted_site_aliases_match():
    site_aliases = {
        "aliases": {
            "TAJ EXPRESS": {
                "canonical_name": "TAJ EXPRESS",
                "service_numbers": ["12279", "12280"],
                "endpoints": ["NEW DELHI", "VGL JHANSI", "GWALIOR"],
                "variations": ["NDLS-VGLJ TAJ EXP", "TAJ EXP", "TAJ SUPERFAST"],
            }
        }
    }
    # 'TAJ EXP' matches 'TAJ EXPRESS' via site aliases
    score = match_name_with_aliases("TAJ EXP", "TAJ EXPRESS", site_aliases=site_aliases)
    assert score == 1.0, f"Expected 1.0 via site aliases, got {score}"
