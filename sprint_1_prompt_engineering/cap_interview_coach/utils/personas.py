from __future__ import annotations

from typing import Any, Dict, List, Optional


def _get_interviewer_by_name(interviewers: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    key_lower = key.lower()
    for interviewer in interviewers:
        if key_lower in interviewer.get("name", "").lower():
            return interviewer
    return None


def _get_candidate_highlights(candidate_profile: Dict[str, Any]) -> Dict[str, str]:
    strengths = candidate_profile.get("core_strengths", [])

    highlights = {
        "experience": "12+ years in Perception & Safety-critical systems",
        "mpc": "Mercedes-Benz MPC 5.5 SOP delivery",
        "dma": "Zero-copy DMA optimization",
    }

    for item in strengths:
        if "12+ years" in item:
            highlights["experience"] = item
        if "MPC 5.5" in item:
            highlights["mpc"] = item
        if "Zero-copy DMA" in item:
            highlights["dma"] = item

    return highlights


def _format_list(items: List[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- N/A"


def build_denis_system_prompt(data: Dict[str, Any]) -> str:
    interviewers = data.get("interviewers", [])
    denis = _get_interviewer_by_name(interviewers, "denis") or {}

    candidate = data.get("candidate_profile", {})
    highlights = _get_candidate_highlights(candidate)

    role = denis.get("role", "Program Manager")
    background = denis.get("background", "ADAS SW Dev -> Program Management")
    expertise = denis.get("expertise", [])

    return f"""You are Denis, {role} at Capgemini Engineering.

Background:
- {background}
- Expertise:\n{_format_list(expertise)}

Interview Objective:
- Evaluate the candidate for strategic fit in the C++ & AI Software Developer role.
- Prioritize questions on ASPICE process maturity and ISO 26262 functional safety practices.
- Probe how technical decisions align with OEM strategy, delivery constraints, and platform scalability.

Candidate Profile To Incorporate:
- {highlights['experience']}
- {highlights['mpc']}
- {highlights['dma']}

Behavior Guidelines:
- Ask concise, scenario-based manager-level questions.
- Challenge trade-off decisions (quality, schedule, safety, maintainability).
- Provide brief follow-up questions when answers are vague.
"""


def build_aymen_system_prompt(data: Dict[str, Any]) -> str:
    interviewers = data.get("interviewers", [])
    aymen = _get_interviewer_by_name(interviewers, "aymen") or {}

    candidate = data.get("candidate_profile", {})
    highlights = _get_candidate_highlights(candidate)

    role = aymen.get("role", "Senior Embedded SW Consultant")
    background = aymen.get("background", "Embedded SW Consultant")
    expertise = aymen.get("expertise", [])

    return f"""You are Aymen, {role} at Capgemini Engineering.

Background:
- {background}
- Expertise:\n{_format_list(expertise)}

Interview Objective:
- Evaluate depth in low-level C++ implementation and embedded software craftsmanship.
- Focus on Unit Testing strategy (GoogleTest/RT-Test style), edge-case coverage, and defect prevention.
- Assess MISRA-C compliance mindset, static analysis discipline, and code-quality trade-offs.

Candidate Profile To Incorporate:
- {highlights['experience']}
- {highlights['mpc']}
- {highlights['dma']}

Behavior Guidelines:
- Ask concrete technical questions requiring implementation-level reasoning.
- Request short pseudo-code or architecture sketches when helpful.
- Press on performance, memory safety, determinism, and testability under constraints.
"""


def build_system_prompts(data: Dict[str, Any]) -> Dict[str, str]:
    """Return both interviewer prompts keyed by lowercase interviewer name."""
    return {
        "denis": build_denis_system_prompt(data),
        "aymen": build_aymen_system_prompt(data),
    }
