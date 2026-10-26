"""
Phase 2 / Phase 7C: Railway OCR Entity Classification.
Strictly distinguishes visually observed raw tokens into typed railway entities:
- SERVICE_NUMBER: 5-digit service candidate
- TRAIN_NAME: Candidate train name verified against curated canonical registry
- ENDPOINT: Source / Destination station or junction
- COACH_SERIAL: Coach identifier (e.g. 217152/C)
- COACH_DESIGNATION: Seating class (S1, B2, A1, GS, etc.)
- OTHER_TEXT: Unclassified or rejected text with explicit rejection reason
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import unicodedata

# Common Indian Railway station endpoints / junctions
KNOWN_ENDPOINTS = {
    "NEW DELHI", "NDLS", "DELHI", "DLI", "HAZRAT NIZAMUDDIN", "NZM",
    "ANAND VIHAR", "ANVT", "OLD DELHI", "KATRA", "SVDK", "JAMMU TAWI", "JAT",
    "JAMNAGAR", "HAPA", "MUMBAI CENTRAL", "MMCT", "BANDRA TERMINUS", "BDTS",
    "HOWRAH", "HWH", "SEALDAH", "SDAH", "CHENNAI CENTRAL", "MAS",
    "BENGALURU", "SBC", "PUNE", "PUNE JN", "LUCKNOW", "LKO", "VARANASI", "BSB",
    "AMRITSAR", "ASR", "CHANDIGARH", "CDG", "DEHRADUN", "DDN",
    "VGL JHANSI", "JHANSI", "GWALIOR", "AGRA CANTT", "MATHURA", "KOTA",
    "AHMEDABAD", "ADI", "PATNA", "PNBE", "GORAKHPUR", "GKP",
    "SHRI MATA VAISHNO DEVI KATRA", "VAISHNO DEVI", "SRI GANGANAGAR", "SGNR",
    "KACHEGUDA", "SECUNDERABAD", "HYDERABAD", "NANDED", "NED"
}

# Negative stencils / class annotations on coach exterior
COACH_CLASS_DESIGNATIONS = {
    "GEN", "GS", "SL", "SLEEPER", "3A", "3E", "2A", "1A", "CC", "2S", "EC", "EA",
    "A1", "A2", "A3", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9",
    "C1", "C2", "C3", "D1", "D2", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8",
    "AC THREE TIER", "AC 3 TIER", "AC TWO TIER", "AC 2 TIER", "AC FIRST CLASS",
    "SECOND CLASS", "LADIES", "DIVYANGJAN", "LUGGAGE", "GUARD", "BRAKE", "RMS",
    "BUFFET CAR", "PANTRY CAR", "PANTRY", "BUFFET", "GENERAL"
}

# Interior, seating, safety signs that must be rejected from train name consideration
SEATING_SAFETY_PATTERNS = [
    re.compile(r"\bEMERGENCY\s+WINDOW\b", re.IGNORECASE),
    re.compile(r"\bTO\s+SEAT(?:/SLEEP)?\b", re.IGNORECASE),
    re.compile(r"\bTOSEAT\b", re.IGNORECASE),
    re.compile(r"^\d{1,2}\s*[-–]\s*\d{1,2}$"),
    re.compile(r"^\d{1,2}\s*[-–]\s*\d{1,2}\s+TO\s+SEAT", re.IGNORECASE),
    re.compile(r"^TO\s+SEAT\s+\d{1,2}\s*[-–]\s*\d{1,2}", re.IGNORECASE),
]

# Generic train name suffix tokens (non-informative for multi-token identity requirements)
GENERIC_NAME_TOKENS = {"EXP", "EXPRESS", "SF", "SUPERFAST", "SPL", "SPECIAL", "MAIL"}


@dataclass
class RegisteredTrainName:
    canonical_name: str
    aliases: List[str]
    allow_single_token: bool = False
    service_family_numbers: List[str] = field(default_factory=list)


# Curated Canonical Railway Train Name Registry
TRAIN_NAME_REGISTRY: Dict[str, RegisteredTrainName] = {
    "UTTAR SAMPARK KRANTI": RegisteredTrainName(
        canonical_name="UTTAR SAMPARK KRANTI",
        aliases=[
            "UTTAR SAMPARK KRANTI",
            "UTTAR SAMPRK KRANTI",
            "UTTAR S KRANTI",
            "UTTAR SAMPARK KRAN",
            "UTTAR SAMPARK",
        ],
        allow_single_token=False,
        service_family_numbers=["12445", "12446"],
    ),
    "BIHAR SAMPARK KRANTI": RegisteredTrainName(
        canonical_name="BIHAR SAMPARK KRANTI",
        aliases=["BIHAR SAMPARK KRANTI", "BIHAR S KRANTI"],
        allow_single_token=False,
        service_family_numbers=["12565", "12566"],
    ),
    "SACHKHAND EXPRESS": RegisteredTrainName(
        canonical_name="SACHKHAND EXPRESS",
        aliases=["SACHKHAND EXPRESS", "SACHKHAND EXP", "SACHKHAND"],
        allow_single_token=True,
        service_family_numbers=["12715", "12716"],
    ),
    "LUCKNOW MAIL": RegisteredTrainName(
        canonical_name="LUCKNOW MAIL",
        aliases=["LUCKNOW MAIL", "LKO MAIL"],
        allow_single_token=False,
        service_family_numbers=["12229", "12230"],
    ),
    "TAJ EXPRESS": RegisteredTrainName(
        canonical_name="TAJ EXPRESS",
        aliases=["TAJ EXPRESS", "TAJ EXP"],
        allow_single_token=False,
        service_family_numbers=["12279", "12280"],
    ),
    "SHIV GANGA EXPRESS": RegisteredTrainName(
        canonical_name="SHIV GANGA EXPRESS",
        aliases=["SHIV GANGA EXPRESS", "SHIV GANGA EXP", "SHIVGANGA EXP", "SHIV GANGA"],
        allow_single_token=False,
        service_family_numbers=["12559", "12560"],
    ),
    "SHRAM SHAKTI EXPRESS": RegisteredTrainName(
        canonical_name="SHRAM SHAKTI EXPRESS",
        aliases=["SHRAM SHAKTI EXP", "SHRAM SHKTI EXP", "SHRAM SHAKTI"],
        allow_single_token=False,
        service_family_numbers=["12451", "12452"],
    ),
    "SWARAJ EXPRESS": RegisteredTrainName(
        canonical_name="SWARAJ EXPRESS",
        aliases=["SWARAJ EXPRESS", "SWARAJ EXP"],
        allow_single_token=False,
        service_family_numbers=["12471", "12472"],
    ),
    "SARVODAYA EXPRESS": RegisteredTrainName(
        canonical_name="SARVODAYA EXPRESS",
        aliases=["SARVODAYA EXPRESS", "SARVODAYA EXP"],
        allow_single_token=False,
        service_family_numbers=["12473", "12474"],
    ),
    "SHRI SHAKTI EXPRESS": RegisteredTrainName(
        canonical_name="SHRI SHAKTI EXPRESS",
        aliases=["SHRI SHAKTI EXPRESS", "SHRI SHAKTI AC EXPRESS", "SHREE SHAKTI EXPRESS"],
        allow_single_token=False,
        service_family_numbers=["22461", "22462"],
    ),
    "VAISHALI EXPRESS": RegisteredTrainName(
        canonical_name="VAISHALI EXPRESS",
        aliases=["VAISHALI EXPRESS", "VAISHALI EXP"],
        allow_single_token=False,
        service_family_numbers=["12553", "12554"],
    ),
    "PURUSHOTTAM EXPRESS": RegisteredTrainName(
        canonical_name="PURUSHOTTAM EXPRESS",
        aliases=["PURUSHOTTAM EXPRESS", "PURUSHOTTAM EXP"],
        allow_single_token=False,
        service_family_numbers=["12801", "12802"],
    ),
    "PRAYAGRAJ EXPRESS": RegisteredTrainName(
        canonical_name="PRAYAGRAJ EXPRESS",
        aliases=["PRAYAGRAJ EXPRESS", "PRAYAGRAJ EXP"],
        allow_single_token=False,
        service_family_numbers=["12417", "12418"],
    ),
    "GOA EXPRESS": RegisteredTrainName(
        canonical_name="GOA EXPRESS",
        aliases=["GOA EXPRESS", "GOA EXP"],
        allow_single_token=False,
        service_family_numbers=["12779", "12780"],
    ),
    "GOLDEN TEMPLE MAIL": RegisteredTrainName(
        canonical_name="GOLDEN TEMPLE MAIL",
        aliases=["GOLDEN TEMPLE MAIL"],
        allow_single_token=False,
        service_family_numbers=["12903", "12904"],
    ),
    "PASCHIM EXPRESS": RegisteredTrainName(
        canonical_name="PASCHIM EXPRESS",
        aliases=["PASCHIM EXPRESS"],
        allow_single_token=False,
        service_family_numbers=["12925", "12926"],
    ),
    "MALWA EXPRESS": RegisteredTrainName(
        canonical_name="MALWA EXPRESS",
        aliases=["MALWA EXPRESS", "MALWA EXP"],
        allow_single_token=False,
        service_family_numbers=["12919", "12920"],
    ),
    "JAMMU MAIL": RegisteredTrainName(
        canonical_name="JAMMU MAIL",
        aliases=["JAMMU MAIL"],
        allow_single_token=False,
        service_family_numbers=["14033", "14034"],
    ),
    "POORVA EXPRESS": RegisteredTrainName(
        canonical_name="POORVA EXPRESS",
        aliases=["POORVA EXPRESS", "POORVA EXP"],
        allow_single_token=False,
        service_family_numbers=["12303", "12304", "12381", "12382"],
    ),
    "GOMTI EXPRESS": RegisteredTrainName(
        canonical_name="GOMTI EXPRESS",
        aliases=["GOMTI EXPRESS", "GOMTI EXP"],
        allow_single_token=False,
        service_family_numbers=["12419", "12420"],
    ),
    "UTTARANCHAL EXPRESS": RegisteredTrainName(
        canonical_name="UTTARANCHAL EXPRESS",
        aliases=["UTTARANCHAL EXPRESS", "UTTARANCHAL EXP"],
        allow_single_token=False,
        service_family_numbers=["19565", "19566"],
    ),
    "SWATANTRATA SENANI EXPRESS": RegisteredTrainName(
        canonical_name="SWATANTRATA SENANI EXPRESS",
        aliases=["SWATANTRATA SENANI EXPRESS", "SWATANTRATA SENANI", "SWATNTRATA SENANI"],
        allow_single_token=False,
        service_family_numbers=["12561", "12562"],
    ),
    "SHAN E PUNJAB": RegisteredTrainName(
        canonical_name="SHAN E PUNJAB",
        aliases=["SHAN E PUNJAB"],
        allow_single_token=False,
        service_family_numbers=["12497", "12498"],
    ),
}

# Backward compatibility alias set
KNOWN_TRAIN_NAMES = {alias for entry in TRAIN_NAME_REGISTRY.values() for alias in entry.aliases}


def levenshtein_distance(s1: str, s2: str) -> int:
    """Standard Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insert = prev[j + 1] + 1
            delete = curr[j] + 1
            sub = prev[j] + (0 if c1 == c2 else 1)
            curr.append(min(insert, delete, sub))
        prev = curr
    return prev[-1]


@dataclass
class ClassifiedEntity:
    entity_type: str  # SERVICE_NUMBER | TRAIN_NAME | ENDPOINT | COACH_SERIAL | COACH_DESIGNATION | OTHER_TEXT
    raw_text: str
    normalized_text: str
    confidence: float
    canonical_name: Optional[str] = None
    candidate_names: List[str] = field(default_factory=list)
    is_ambiguous: bool = False
    rejection_reason: Optional[str] = None
    notes: str = ""


class EntityClassifier:
    """
    Phase 7C: Evidence-Preserving Railway OCR Entity Classifier.
    Strictly distinguishes visually observed raw tokens into typed railway entities
    without fabricating active service numbers or suppressing evidence.
    """

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Algorithm Step 1 & 2:
        Normalize OCR text using Unicode NFKC normalization, uppercase conversion,
        whitespace normalization, and punctuation removal, while preserving the original text.
        """
        if not text:
            return ""
        t = unicodedata.normalize("NFKC", text)
        t = t.strip().upper()
        # Collapse multiple spaces
        t = re.sub(r"\s+", " ", t)
        # Strip decorative and edge punctuation
        t = re.sub(r"^[\-=:;~|.,#\*\"\'_`/\\]+|[\-=:;~|.,#\*\"\'_`/\\]+$", "", t).strip()
        return t

    @classmethod
    def classify_all(cls, raw_text: str, ocr_confidence: float) -> List[ClassifiedEntity]:
        norm = cls.normalize_text(raw_text)

        # Reject class designations and coach serials BEFORE any service-number extraction.
        if re.search(r"\b\d{4,6}\s*/\s*[A-Z]\b", norm):
            return [ClassifiedEntity("COACH_SERIAL", raw_text, norm, ocr_confidence, notes="slash_coach_serial")]
        if norm in COACH_CLASS_DESIGNATIONS:
            return [ClassifiedEntity("COACH_DESIGNATION", raw_text, norm, ocr_confidence, notes="exact_class_designation")]

        # Multi-number boards: e.g. "12445 / 12446"
        pair_or_group = re.fullmatch(r"\d{5}(?:\s*[/|,\-]\s*\d{5}){1,5}", norm)
        if pair_or_group:
            numbers = re.findall(r"(?<!\d)\d{5}(?!\d)", norm)
            return [
                ClassifiedEntity("SERVICE_NUMBER", raw_text, n, ocr_confidence, notes="multi_number_board")
                for n in dict.fromkeys(numbers)
            ]
        return [cls.classify(raw_text, ocr_confidence)]

    @classmethod
    def _match_train_name_against_registry(
        cls,
        norm_text: str,
    ) -> Tuple[Optional[str], List[str], Optional[str]]:
        """
        Phase 7C Train Name Matching Algorithm:
        - Whole-token matching plus restricted edit-distance matching.
        - Requires at least two informative matching tokens, unless explicitly registered single-token alias.
        - Final-word prefix allowed only if >= 4 chars and uniquely identifies one canonical word.
        - Returns: (canonical_name, candidate_names, rejection_reason)
        """
        q_tokens = norm_text.split()
        if not q_tokens:
            return None, [], "EMPTY_STRING"

        # Algorithm §5: Single-token input handling
        if len(q_tokens) == 1:
            for reg_key, reg_entry in TRAIN_NAME_REGISTRY.items():
                if reg_entry.allow_single_token:
                    for alias in reg_entry.aliases:
                        if alias.split() == q_tokens:
                            return reg_entry.canonical_name, [reg_entry.canonical_name], None
            return None, [], "INSUFFICIENT_NAME_TOKENS"

        surviving_canonical: Set[str] = set()
        rejection_reason: Optional[str] = None

        for reg_key, reg_entry in TRAIN_NAME_REGISTRY.items():
            for alias in reg_entry.aliases:
                a_tokens = alias.split()

                # Query cannot have more tokens than alias
                if len(q_tokens) > len(a_tokens):
                    continue

                for offset in range(len(a_tokens) - len(q_tokens) + 1):
                    informative_matches = 0
                    matched_all_query_tokens = True

                    for i, q_tok in enumerate(q_tokens):
                        is_last_q_tok = (i == len(q_tokens) - 1)
                        target_a_tok = a_tokens[offset + i]

                        token_matched = False

                        # 1. Exact whole-token match
                        if q_tok == target_a_tok:
                            token_matched = True

                        # 2. Restricted edit-distance match for words with length >= 6
                        elif len(target_a_tok) >= 6 and len(q_tok) >= 5:
                            if levenshtein_distance(q_tok, target_a_tok) <= 1:
                                token_matched = True

                        # 3. Final-word prefix rule (Algorithm §6):
                        # Permit a final-word prefix only when >= 4 characters and uniquely identifies
                        # one canonical word among eligible registry entries
                        elif is_last_q_tok and len(q_tok) >= 4 and target_a_tok.startswith(q_tok):
                            preceding_matches = []
                            for other_entry in TRAIN_NAME_REGISTRY.values():
                                for other_alias in other_entry.aliases:
                                    o_toks = other_alias.split()
                                    if len(o_toks) > (offset + i) and o_toks[:offset + i] == a_tokens[:offset + i]:
                                        if o_toks[offset + i].startswith(q_tok):
                                            preceding_matches.append(o_toks[offset + i])
                            distinct_targets = set(preceding_matches)
                            if len(distinct_targets) == 1:
                                token_matched = True

                        if token_matched:
                            if q_tok not in GENERIC_NAME_TOKENS and target_a_tok not in GENERIC_NAME_TOKENS:
                                informative_matches += 1
                        else:
                            matched_all_query_tokens = False
                            break

                    if matched_all_query_tokens:
                        # Algorithm §5: Require at least two informative matching tokens,
                        # except for explicitly approved unique aliases where all tokens match.
                        is_exact_approved_alias = (len(q_tokens) == len(a_tokens))
                        if informative_matches >= 2 or is_exact_approved_alias:
                            surviving_canonical.add(reg_entry.canonical_name)
                            break
                        else:
                            rejection_reason = "LESS_THAN_TWO_INFORMATIVE_TOKENS"

        candidates = sorted(list(surviving_canonical))
        if len(candidates) == 1:
            return candidates[0], candidates, None
        elif len(candidates) > 1:
            return None, candidates, None
        else:
            return None, [], rejection_reason or "NO_APPROVED_TRAIN_NAME_MATCH"

    @classmethod
    def classify(cls, raw_text: str, ocr_confidence: float) -> ClassifiedEntity:
        norm = cls.normalize_text(raw_text)

        if not norm:
            return ClassifiedEntity(
                "OTHER_TEXT", raw_text, "", ocr_confidence,
                is_ambiguous=True, rejection_reason="EMPTY_TEXT", notes="empty_string"
            )

        # 1. Coach Serial Number: e.g. "217152/C", "08234/A", "2249/C", or 6 digits "243560"
        if re.search(r"\b\d{4,6}\s*/\s*[A-Z]\b", norm) or (len(norm) == 6 and norm.isdigit()):
            return ClassifiedEntity(
                "COACH_SERIAL", raw_text, norm, ocr_confidence, notes="coach_serial_pattern"
            )

        # 2. Coach Designation / Negative Stencil
        if norm in COACH_CLASS_DESIGNATIONS:
            return ClassifiedEntity(
                "COACH_DESIGNATION", raw_text, norm, ocr_confidence,
                rejection_reason="COACH_CLASS_DESIGNATION", notes="exact_class_designation"
            )

        for desig in COACH_CLASS_DESIGNATIONS:
            if re.fullmatch(rf"{desig}\b", norm):
                return ClassifiedEntity(
                    "COACH_DESIGNATION", raw_text, norm, ocr_confidence,
                    rejection_reason="COACH_CLASS_DESIGNATION", notes="class_designation_regex"
                )

        # 3. Reject Seating, Emergency Window, and Interior Signs
        for pat in SEATING_SAFETY_PATTERNS:
            if pat.search(norm):
                return ClassifiedEntity(
                    "OTHER_TEXT", raw_text, norm, ocr_confidence,
                    rejection_reason="SEATING_SAFETY_SIGN", notes="rejected_sign"
                )

        # 4. Exact 5-digit Service Number
        if re.fullmatch(r"\d{5}", norm):
            return ClassifiedEntity(
                "SERVICE_NUMBER", raw_text, norm, ocr_confidence, notes="exact_5_digits"
            )

        # Check for multiple 5-digit numbers separated by slash or dash: e.g. "12477/12478"
        multi_5d = re.findall(r"\b\d{5}\b", norm)
        if len(multi_5d) >= 1 and len(norm) <= 15:
            return ClassifiedEntity(
                "SERVICE_NUMBER", raw_text, multi_5d[0], ocr_confidence, notes="embedded_5_digits"
            )

        # 5. Evidence-Preserving Train Name Recognition (§Task 1)
        can_name, cand_names, rej_reason = cls._match_train_name_against_registry(norm)
        if can_name:
            return ClassifiedEntity(
                "TRAIN_NAME", raw_text, norm, ocr_confidence,
                canonical_name=can_name, candidate_names=[can_name],
                is_ambiguous=False, notes="curated_train_name_match"
            )
        elif cand_names:
            # Ambiguous surviving candidates (§8: preserve as ambiguous rather than choosing arbitrarily)
            return ClassifiedEntity(
                "TRAIN_NAME", raw_text, norm, ocr_confidence,
                canonical_name=None, candidate_names=cand_names,
                is_ambiguous=True, notes="ambiguous_multiple_candidates"
            )

        # 6. Known Endpoints / Station names alone
        if norm in KNOWN_ENDPOINTS:
            return ClassifiedEntity("ENDPOINT", raw_text, norm, ocr_confidence, notes="exact_endpoint")

        for ep in KNOWN_ENDPOINTS:
            if ep == norm and len(ep) >= 4:
                return ClassifiedEntity("ENDPOINT", raw_text, ep, ocr_confidence, notes="endpoint_full")

        if norm in {"SVDK", "NDLS", "HAPA", "JAT", "LKO", "BSB", "ASR", "CDG", "DDN", "ADI"}:
            return ClassifiedEntity("ENDPOINT", raw_text, norm, ocr_confidence, notes="station_code")

        # 7. Default to OTHER_TEXT with explicit rejection reason (§9)
        return ClassifiedEntity(
            "OTHER_TEXT", raw_text, norm, ocr_confidence,
            is_ambiguous=True, rejection_reason=rej_reason or "UNCLASSIFIED", notes="unclassified"
        )
