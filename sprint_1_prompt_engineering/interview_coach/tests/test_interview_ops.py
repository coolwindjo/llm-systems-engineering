from __future__ import annotations

import json
from pathlib import Path

from services.interview_ops import DEFAULT_CRITIQUE_PERSONA, get_critique_persona_prompt
from services.interview_ops import _speaker_key


def test_get_critique_persona_prompt_from_template(tmp_path: Path) -> None:
    template_path = tmp_path / "system_prompts.json"
    template_path.write_text(
        json.dumps(
            {
                "critique_persona": "Critique for {interviewer_name}: evaluate answer quality strictly.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text = get_critique_persona_prompt(interviewer_name="Jordan", path=template_path)
    assert "Critique for Jordan" in text
    assert text == "Critique for Jordan: evaluate answer quality strictly."


def test_get_critique_persona_prompt_selects_interviewer_technique_preference(tmp_path: Path) -> None:
    template_path = tmp_path / "system_prompts.json"
    template_path.write_text(
        json.dumps(
            {
                "critique_persona": {
                    "default": "default {interviewer_name} {technique} {interviewer_key}",
                    "by_technique": {"chain_of_thought": "technique {interviewer_name}"},
                    "by_interviewer": {_speaker_key("Generic AI Interviewer"): "interviewer {interviewer_name}"},
                    "by_combo": {"generic_ai_interviewer.chain_of_thought": "combo {interviewer_name}"},
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    combo = get_critique_persona_prompt(
        interviewer_name="Jordan",
        interviewer_key="Generic AI Interviewer",
        technique="chain_of_thought",
        path=template_path,
    )
    assert combo == "combo Jordan"

    by_interviewer = get_critique_persona_prompt(
        interviewer_name="Jordan",
        interviewer_key="Generic AI Interviewer",
        technique="few_shot",
        path=template_path,
    )
    assert by_interviewer == "interviewer Jordan"

    by_technique = get_critique_persona_prompt(
        interviewer_name="Jordan",
        interviewer_key="Other",
        technique="chain_of_thought",
        path=template_path,
    )
    assert by_technique == "technique Jordan"

    default = get_critique_persona_prompt(
        interviewer_name="Jordan",
        interviewer_key="Other",
        technique="zero_shot",
        path=template_path,
    )
    assert default == "default Jordan zero_shot other"


def test_get_critique_persona_prompt_uses_jd_keyword_catalog_focus_terms(tmp_path: Path) -> None:
    template_path = tmp_path / "system_prompts.json"
    template_path.write_text(
        json.dumps(
            {
                "critique_persona": {
                    "default": "Focus terms: {focus_terms}\n{interviewer_name} at {company} on {position}",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text = get_critique_persona_prompt(
        interviewer_name="Jordan",
        interviewer_key="Other",
        technique="few_shot",
        jd_profile={
            "jd_keyword_catalog": {
                "Safety & Compliance": ["ASPICE CL3", "ISO 26262"],
                "Testing & Quality": ["Vehicle validation"],
                "Tools & Framework": ["Docker", "Git"],
            },
        },
        path=template_path,
    )

    assert text == "Focus terms: ASPICE CL3, ISO 26262, Vehicle validation, Docker, Git\nJordan at Capgemini on this role"


def test_get_critique_persona_prompt_formats_jd_and_interviewer_context(tmp_path: Path) -> None:
    template_path = tmp_path / "system_prompts.json"
    template_path.write_text(
        json.dumps(
            {
                "critique_persona": {
                    "default": "Critique for {interviewer_name} at {company} covering {position}.",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text = get_critique_persona_prompt(
        interviewer_name="Jordan",
        interviewer_key="Other",
        technique="few_shot",
        jd_profile={"company": "Acme", "position": "ADAS Lead"},
        jd_title="Fallback Title",
        path=template_path,
    )

    assert text == "Critique for Jordan at Acme covering ADAS Lead."


def test_get_critique_persona_prompt_selects_position_template(tmp_path: Path) -> None:
    template_path = tmp_path / "system_prompts.json"
    template_path.write_text(
        json.dumps(
            {
                "critique_persona": {
                    "default": "default",
                    "by_position": {
                        "adas": "ADAS position template for {interviewer_name}",
                        "safety": "Safety template for {position}",
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text = get_critique_persona_prompt(
        interviewer_name="Jordan",
        interviewer_key="Other",
        technique="zero_shot",
        jd_profile={"position": "Senior AD/ADAS Developer"},
        path=template_path,
    )

    assert text == "ADAS position template for Jordan"


def test_get_critique_persona_prompt_selects_company_template(tmp_path: Path) -> None:
    template_path = tmp_path / "system_prompts.json"
    template_path.write_text(
        json.dumps(
            {
                "critique_persona": {
                    "default": "default",
                    "by_company": {
                        "capgemini": "Capgemini template for {company}",
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text = get_critique_persona_prompt(
        interviewer_name="Jordan",
        interviewer_key="Other",
        technique="zero_shot",
        jd_profile={"company": "Capgemini Engineering"},
        path=template_path,
    )

    assert text == "Capgemini template for Capgemini Engineering"


def test_get_critique_persona_prompt_uses_interviewer_style_profile(tmp_path: Path) -> None:
    template_path = tmp_path / "system_prompts.json"
    template_path.write_text(
        json.dumps(
            {
                "critique_persona": {
                    "default": "Default style for {interviewer_name}",
                    "by_style": {
                        "ai_ml": "AI style for {interviewer_name}",
                        "technical_general": "General style for {interviewer_name}",
                    },
                    "by_style_and_technique": {
                        "ai_ml.chain_of_thought": "AI chain of thought for {interviewer_name}",
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text = get_critique_persona_prompt(
        interviewer_name="Jordan",
        interviewer_key="New Recruiter",
        technique="chain_of_thought",
        interviewer_profile={"role": "AI Researcher", "background": "Machine Learning and Deep Learning lead"},
        path=template_path,
    )
    assert text == "AI chain of thought for Jordan"


def test_get_critique_persona_prompt_respects_explicit_style_profile(tmp_path: Path) -> None:
    template_path = tmp_path / "system_prompts.json"
    template_path.write_text(
        json.dumps(
            {
                "critique_persona": {
                    "default": "Default style for {interviewer_name}",
                    "by_style": {
                        "safety_management": "Safety style for {interviewer_name}",
                    },
                    "by_style_and_technique": {
                        "safety_management.chain_of_thought": "Safety chain for {interviewer_name}",
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text = get_critique_persona_prompt(
        interviewer_name="Jordan",
        interviewer_key="New Recruiter",
        technique="chain_of_thought",
        interviewer_profile={"critique_profile": {"style": "safety_management"}},
        path=template_path,
    )
    assert text == "Safety chain for Jordan"


def test_get_critique_persona_prompt_injects_focus_terms_from_jd_text(tmp_path: Path) -> None:
    template_path = tmp_path / "system_prompts.json"
    template_path.write_text(
        json.dumps(
            {
                "critique_persona": {
                    "default": "Focus terms: {focus_terms}",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text = get_critique_persona_prompt(
        interviewer_name="Jordan",
        interviewer_key="Other",
        technique="zero_shot",
        jd_profile={
            "job_description": "You must deliver with ASPICE CL3 and ISO 26262 compliance.",
        },
        jd_title="ADAS Role",
        path=template_path,
    )

    assert text == "Focus terms: ASPICE CL3, ISO 26262"


def test_get_critique_persona_prompt_injects_focus_terms_from_interviewer_profile(tmp_path: Path) -> None:
    template_path = tmp_path / "system_prompts.json"
    template_path.write_text(
        json.dumps(
            {
                "critique_persona": {
                    "default": "Focus terms: {focus_terms}",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text = get_critique_persona_prompt(
        interviewer_name="Jordan",
        interviewer_key="Other",
        technique="zero_shot",
        jd_profile={"key_requirements": ["Safety case ownership"]},
        interviewer_profile={"critique_profile": {"focus_terms": ["Traceability", "Safety case"]}},
        path=template_path,
    )

    assert text == "Focus terms: Safety case ownership, Traceability, Safety case"


def test_get_critique_persona_prompt_fallback_when_template_missing(tmp_path: Path) -> None:
    text = get_critique_persona_prompt(
        interviewer_name="Jordan",
        path=tmp_path / "does_not_exist.json",
    )
    expected = DEFAULT_CRITIQUE_PERSONA.format(
        company="Capgemini",
        interviewer_name="Jordan",
        position="this role",
        technical_domain="General software",
        technical_scope="software engineering",
        quality_metric="Technical quality",
        focus_terms="Technical quality, General software implementation quality",
    )
    assert text == expected


def test_get_critique_persona_prompt_infers_domain_and_metric_from_jd(tmp_path: Path) -> None:
    template_path = tmp_path / "system_prompts.json"
    template_path.write_text(
        json.dumps(
            {
                "critique_persona": {
                    "default": "{company}::{technical_domain}::{technical_scope}::{quality_metric}::{focus_terms}",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text = get_critique_persona_prompt(
        interviewer_name="Jordan",
        jd_profile={
            "position": "Senior AD/ADAS Developer",
            "key_requirements": ["ASPICE CL3", "C++ architecture"],
            "tech_stack": ["Python", "Docker"],
        },
        path=template_path,
    )

    assert text == (
        "Capgemini::ADAS / automotive::embedded C++ and diagnostics::ASPICE CL3::ASPICE CL3, C++ architecture, "
        "Python, Docker"
    )
