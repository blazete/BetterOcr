"""
Text Normalization and Entity Extraction for Train Identity
Provides robust normalization for 5-digit service numbers, train names,
and route endpoints, while rejecting coach serials and negative classes.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

# Negative class patterns (coach serials, berth numbers, class stencils)
COACH_SERIAL_REGEX = re.compile(r"^\d{4,6}[A-Z]?(?:/[A-Z0-9]+)?$")
BERTH_STENCIL_REGEX = re.compile(r"^\d{1,2}\s*[-–toTO]\s*\d{1,2}$")
CLASS_STENCIL_REGEX = re.compile(r"^(?:S\d{1,2}|B\d{1,2}|A\d{1,2}|H\d{1,2}|C\d{1,2}|D\d{1,2}|GEN|GS|SL|EOG|SLR)(?:\s+\d{1,2}\s*[-–toTO]\s*\d{1,2})?$", re.IGNORECASE)

SUB_MAP = str.maketrans({
    "O": "0", "o": "0",
    "D": "0",
    "I": "1", "i": "1", "l": "1", "L": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "B": "8"
})


def is_negative_class(text: str) -> bool:
    """Check if token is a coach serial, berth range, or class designation."""
    t = text.strip()
    if "/" in t: # e.g. 217152/C
        return True
    if BERTH_STENCIL_REGEX.match(t):
        return True
    if CLASS_STENCIL_REGEX.match(t):
        return True
    # If it is 6 digits long, it is a modern LHB coach serial (e.g. 217152), not a 5-digit service number
    if len(t) == 6 and t.isdigit():
        return True
    return False


def extract_5digit_candidates(text: str) -> List[str]:
    """Extract valid 5-digit service numbers with letter-confusion normalization."""
    candidates: List[str] = []
    tokens = re.split(r"[\s,;|\-/]+", text.strip())
    for tok in tokens:
        if not tok or is_negative_class(tok):
            continue
        # Check direct 5-digit
        if len(tok) == 5 and tok.isdigit():
            if tok not in candidates:
                candidates.append(tok)
            continue
        # Normalized translation
        norm = tok.translate(SUB_MAP)
        for m in re.findall(r"\b\d{5}\b", norm):
            if m not in candidates and not is_negative_class(m):
                candidates.append(m)
    return candidates


# Aliases for compatibility
is_negative_class_text = is_negative_class
extract_service_numbers = extract_5digit_candidates


def normalize_train_name(text: str) -> str:
    """Normalize train name string for fuzzy and token matching."""
    cleaned = text.upper()
    # Replace non-alphanumeric with spaces
    cleaned = re.sub(r"[^A-Z0-9\s]", " ", cleaned)
    # Collapse whitespace
    tokens = cleaned.split()
    # Filter common noise tokens
    filtered = [tok for tok in tokens if tok not in ("INDIAN", "RAILWAYS", "BHARATIYA", "RAIL")]
    return " ".join(filtered)


KNOWN_STATIONS = {
    "NEW DELHI", "DELHI", "NDLS", "DLI", "OLD DELHI", "HAZRAT NIZAMUDDIN", "NZM",
    "ANAND VIHAR", "ANVT", "VGL JHANSI", "JHANSI", "VGLJ", "LUCKNOW", "LJN", "LKO",
    "KANPUR", "KANPUR CENTRAL", "CNB", "BANARAS", "BSBS", "VARANASI", "BSB",
    "HOWRAH", "HWH", "SEALDAH", "SDAH", "KOLKATA", "GWALIOR", "GWL", "AGRA", "AGC",
    "AMRITSAR", "ASR", "JAMMU", "JAMMU TAWI", "JAT", "PATNA", "PNBE", "GORAKHPUR", "GKP",
    "MUMBAI", "MUMBAI CENTRAL", "CSMT", "MMCT", "CHENNAI", "MAS", "GHAZIABAD", "GZB",
    "PANIPAT", "PNP", "KURUKSHETRA", "KKDE", "JAYNAGAR", "JYG", "DARBHANGA", "DBG",
    "GODDA", "GODA", "SILCHAR", "SCL", "GUWAHATI", "GHY", "KOTA", "SGAC", "TILAK BRIDGE", "TKJ", "BSC",
}


def is_endpoint_text(text: str, extra_endpoints: Optional[Set[str]] = None) -> bool:
    """
    Check if text is a route endpoint or origin-destination pair rather than a train name.
    E.g. 'NEW DELHI', 'VGL JHANSI', 'NEW DELHI - VGL JHANSI', 'NDLS-LJN', etc.
    """
    cleaned = text.strip().upper()
    if not cleaned:
        return False
    # Route separators
    if "-" in cleaned or "–" in cleaned or " TO " in cleaned:
        return True
    norm = normalize_train_name(cleaned)
    if norm in KNOWN_STATIONS:
        return True
    if extra_endpoints:
        norm_extras = {normalize_train_name(e) for e in extra_endpoints if e}
        if norm in norm_extras:
            return True
    return False


GENERIC_NAME_TOKENS = {"EXP", "EXPRESS", "SF", "SUPERFAST", "SPL", "SPECIAL", "MAIL"}


def meaningful_token_sets(q_norm: str, t_norm: str) -> Tuple[Set[str], Set[str]]:
    q_tokens = set(q_norm.split())
    t_tokens = set(t_norm.split())
    q_core = q_tokens - GENERIC_NAME_TOKENS
    t_core = t_tokens - GENERIC_NAME_TOKENS
    return q_core, t_core


def match_name_similarity(query_name: str, target_name: str) -> float:
    """Token-based Jaccard similarity between two train names with generic token stripping."""
    q_norm = normalize_train_name(query_name)
    t_norm = normalize_train_name(target_name)
    if not q_norm or not t_norm:
        return 0.0
    if q_norm == t_norm:
        return 1.0

    q_core, t_core = meaningful_token_sets(q_norm, t_norm)
    if not q_core or not t_core:
        return 0.0
    if any(len(tok) < 4 for tok in q_core):
        return 0.0  # short aliases can match ONLY if explicitly whitelisted in site aliases
    score = len(q_core & t_core) / len(q_core | t_core)
    return round(score, 4)


def match_name_with_aliases(
    query_name: str,
    target_name: str,
    target_aliases: Optional[List[str]] = None,
    site_aliases: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute best similarity across canonical target name, target aliases,
    and site-level alias registry.
    """
    best_score = match_name_similarity(query_name, target_name)
    if target_aliases:
        for alias in target_aliases:
            score = match_name_similarity(query_name, alias)
            if score > best_score:
                best_score = score

    if site_aliases:
        alias_dict = site_aliases.get("aliases", site_aliases)
        q_norm = normalize_train_name(query_name)
        t_norm = normalize_train_name(target_name)
        for canon, entry in alias_dict.items():
            if not isinstance(entry, dict):
                continue
            names_in_entry = [canon]
            names_in_entry.append(entry.get("canonical_name", ""))
            names_in_entry.extend(entry.get("variations", []))
            normalized_entry_names = [normalize_train_name(n) for n in names_in_entry if n]
            
            # If query matches this entry
            q_matches = any(match_name_similarity(q_norm, en) >= 0.8 for en in normalized_entry_names)
            t_matches = any(match_name_similarity(t_norm, en) >= 0.8 for en in normalized_entry_names)
            if q_matches and t_matches:
                return max(best_score, 1.0)
            if q_matches:
                for en in normalized_entry_names:
                    s = match_name_similarity(t_norm, en)
                    if s > best_score:
                        best_score = s

    return round(best_score, 4)
