"""
Text Normalization and Entity Extraction for Train Identity
Provides robust normalization for 5-digit service numbers, train names,
and route endpoints, while rejecting coach serials and negative classes.
"""

import re
from typing import List, Optional, Set, Tuple

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


def match_name_similarity(query_name: str, target_name: str) -> float:
    """Token-based Jaccard / Overlap similarity between two train names."""
    q_norm = normalize_train_name(query_name)
    t_norm = normalize_train_name(target_name)
    if not q_norm or not t_norm:
        return 0.0
    if q_norm == t_norm or q_norm in t_norm or t_norm in q_norm:
        return 1.0
    
    q_tokens = set(q_norm.split())
    t_tokens = set(t_norm.split())
    
    # Strip generic suffixes for token comparison
    generic = {"EXP", "EXPRESS", "SF", "SUPERFAST", "SPL", "SPECIAL", "MAIL"}
    q_core = q_tokens - generic
    t_core = t_tokens - generic
    
    if q_core and t_core:
        intersect = q_core.intersection(t_core)
        union = q_core.union(t_core)
        return len(intersect) / len(union)
    
    intersect = q_tokens.intersection(t_tokens)
    union = q_tokens.union(t_tokens)
    return len(intersect) / len(union) if union else 0.0
