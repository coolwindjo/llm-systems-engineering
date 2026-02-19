from __future__ import annotations

import re
from typing import Any, Dict, List

JD_KEYWORD_CATEGORY_RULES: List[tuple[str, List[str]]] = [
    ("Safety & Compliance", ["aspice", "iso 26262", "sotif", "asil", "safety", "fmea"]),
    ("Testing & Quality", ["test", "testing", "validation", "verification", "traceability", "quality"]),
    (
        "ADAS Core",
        [
            "adas",
            "radar",
            "camera",
            "lidar",
            "fusion",
            "perception",
            "localization",
            "tracking",
            "mapping",
        ],
    ),
    (
        "Runtime & Implementation",
        [
            "c++",
            "cpp",
            "misra",
            "autosar",
            "dma",
            "determinism",
            "real-time",
            "concurrency",
            "memory",
            "thread",
        ],
    ),
    (
        "Tools & Framework",
        [
            "python",
            "docker",
            "can",
            "lin",
            "ethernet",
            "git",
            "jenkins",
            "jira",
            "matlab",
            "simulink",
            "ros",
            "linux",
        ],
    ),
]

_REDUNDANT_PREFIXES = (
    "experience in",
    "experience with",
    "knowledge of",
    "knowledge on",
    "proven experience in",
    "ability to",
    "proven ability to",
    "strong focus on",
    "good understanding of",
    "solid understanding of",
    "expertise in",
    "required to have",
)


def _normalize_term_item(raw: Any) -> str:
    term = str(raw).strip()
    if not term:
        return ""

    split_items = [item.strip() for item in re.split(r"[;\n|,]+", term) if item.strip()]
    if not split_items:
        split_items = [term]

    normalized_items: List[str] = []
    for item in split_items:
        item = item.replace("C/C++", "C++").replace("c/c++", "C++")
        item = re.sub(r"\s{2,}", " ", item).strip(" -")
        item = item.replace("  ", " ")
        for prefix in _REDUNDANT_PREFIXES:
            pattern = re.compile(rf"^{re.escape(prefix)}\s+", re.IGNORECASE)
            cleaned = pattern.sub("", item).strip()
            if cleaned != item:
                item = cleaned
                break
        if not item:
            continue
        normalized_items.append(item)
    return normalized_items[0] if normalized_items else ""


def normalize_terms(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    seen: set[str] = set()

    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        items = [part.strip() for part in re.split(r"[;\n|,]+", text) if part.strip()]
        if not items:
            continue

        for item in items:
            term = _normalize_term_item(item)
            if not term:
                continue
            term = term.strip()
            if len(term) < 2:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(term)

    return normalized


def _normalize_tech_stack_terms(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    seen: set[str] = set()

    for raw in values:
        text = str(raw).strip()
        if not text:
            continue

        items = [part.strip() for part in re.split(r"[;\n|,]+", text) if part.strip()]
        if not items:
            continue

        for item in items:
            term = _normalize_term_item(item)
            if not term:
                continue
            if term == "C++" and item.replace(" ", "").lower() == "c/c++":
                term = "C/C++"

            if len(term) < 2:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(term)

    return normalized


def _normalize_terms(values: Any) -> List[str]:
    """Backward-compatible helper kept for legacy internal imports."""
    return normalize_terms(values)


def classify_term(term: str) -> str:
    candidate = term.lower()
    for category, hints in JD_KEYWORD_CATEGORY_RULES:
        if any(hint in candidate for hint in hints):
            return category
    return "Role Requirements"


def normalize_catalog(values: Dict[str, List[str]]) -> Dict[str, List[str]]:
    output: Dict[str, List[str]] = {}
    for category, items in values.items():
        cleaned = normalize_terms(items)
        if cleaned:
            output[category] = cleaned
    return output


def build_jd_keyword_catalog(profile: Dict[str, Any]) -> Dict[str, List[str]]:
    requirements = normalize_terms(profile.get("key_requirements", []))
    tech_stack = _normalize_tech_stack_terms(profile.get("tech_stack", []))

    catalog: Dict[str, List[str]] = {
        "Role Requirements": [],
        "Safety & Compliance": [],
        "Testing & Quality": [],
        "ADAS Core": [],
        "Runtime & Implementation": [],
        "Tools & Framework": [],
    }

    for term in requirements:
        inferred = classify_term(term)
        if inferred in catalog:
            catalog[inferred].append(term)
            continue
        catalog["Role Requirements"].append(term)

    for term in tech_stack:
        catalog["Tools & Framework"].append(term)

    normalized_catalog: Dict[str, List[str]] = {}
    for category, terms in catalog.items():
        if category == "Tools & Framework":
            normalized_catalog[category] = _normalize_tech_stack_terms(terms)
        else:
            normalized_catalog[category] = normalize_terms(terms)

    return normalized_catalog
