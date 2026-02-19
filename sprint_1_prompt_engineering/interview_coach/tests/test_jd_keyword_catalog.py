from __future__ import annotations

from services import jd_keyword_catalog as catalog


def test_normalize_term_item_supports_prefix_cleanup() -> None:
    assert catalog._normalize_term_item("Experience in C/C++") == "C++"
    assert catalog._normalize_term_item("  knowledge on Python") == "Python"
    assert catalog._normalize_term_item("ADAS, Perception") == "ADAS"


def test_build_jd_keyword_catalog_consistent_with_existing_expectations() -> None:
    profile = {
        "key_requirements": ["Safety", "ISO 26262", "ADAS stack", "planning and test"],
        "tech_stack": ["C/C++", "Docker", "python"],
    }
    catalog_data = catalog.build_jd_keyword_catalog(profile)

    assert catalog_data["Safety & Compliance"] == ["Safety", "ISO 26262"]
    assert catalog_data["ADAS Core"] == ["ADAS stack"]
    assert catalog_data["Tools & Framework"] == ["C/C++", "Docker", "python"]
    assert catalog_data["Testing & Quality"] == ["planning and test"]
