from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _get_interviewer_by_name(interviewers: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    key_lower = key.lower()
    for interviewer in interviewers:
        if key_lower in interviewer.get("name", "").lower():
            return interviewer
    return None


def _get_candidate_highlights(candidate_profile: Dict[str, Any]) -> Dict[str, str]:
    strengths = candidate_profile.get("core_strengths", [])
    if not isinstance(strengths, list):
        strengths = []
    filtered = [str(item).strip() for item in strengths if str(item).strip()]

    if not filtered:
        profile_blurbs = candidate_profile.get("highlights", [])
        if isinstance(profile_blurbs, list):
            filtered = [str(item).strip() for item in profile_blurbs if str(item).strip()]

    if not filtered:
        filtered = ["Relevant candidate profile data was not provided."]

    return {
        "experience": filtered[0],
        "mpc": filtered[1] if len(filtered) > 1 else filtered[0],
        "dma": filtered[2] if len(filtered) > 2 else filtered[0],
    }


def _candidate_profile(data: Dict[str, Any]) -> Dict[str, Any]:
    candidate = data.get("candidate_profile", {})
    return candidate if isinstance(candidate, dict) else {}


def _candidate_identity(candidate_profile: Dict[str, Any]) -> str:
    return str(candidate_profile.get("name", "the candidate")).strip() or "the candidate"


def _candidate_probe_phrases(candidate_profile: Dict[str, Any]) -> List[str]:
    custom_phrases = candidate_profile.get("probing_phrases", [])
    if not custom_phrases:
        custom_phrases = candidate_profile.get("probes", {}).get("focus_phrases", [])
    if isinstance(custom_phrases, list) and custom_phrases:
        return [str(item).strip() for item in custom_phrases if str(item).strip()]
    return []


def _format_list(items: List[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- N/A"


def _extract_jd_requirements(jd_profile: Dict[str, Any]) -> List[str]:
    requirements = jd_profile.get("key_requirements", [])
    if isinstance(requirements, list) and requirements:
        return [str(item) for item in requirements if str(item).strip()]

    job_positions = jd_profile.get("job_positions", [])
    collected: List[str] = []
    if isinstance(job_positions, list):
        for position in job_positions:
            if not isinstance(position, dict):
                continue
            for item in position.get("key_requirements", []) or []:
                text = str(item).strip()
                if text and text not in collected:
                    collected.append(text)
    return collected


def _interviewer_key(name: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    lowered = lowered.strip("_")
    if not lowered:
        return "interviewer"
    if "generic" in lowered:
        return "generic_ai_interviewer"
    return lowered


def _to_dict(profile: Any) -> Dict[str, Any]:
    if isinstance(profile, dict):
        return profile
    if hasattr(profile, "model_dump"):
        return profile.model_dump()
    return {}


def _append_jd_context_and_probe_instructions(
    prompt: str,
    jd_profile: Dict[str, Any],
    interviewer_name: str,
    is_generic_ai: bool = False,
) -> str:
    requirements = _extract_jd_requirements(jd_profile)
    role_context = _format_list(requirements)
    candidate_profile = _candidate_profile(jd_profile)
    candidate_name = _candidate_identity(candidate_profile)
    candidate_phrases = _candidate_probe_phrases(candidate_profile)
    if not candidate_phrases:
        candidate_phrases = [
            f"Specifically probe {candidate_name} on how their ADAS Perception experience directly solves the challenges listed in this JD.",
            f"Ask follow-up questions that force explicit mapping between {candidate_name}'s past ADAS projects and each high-priority requirement in this NEW Job Description.",
        ]

    perspective = (
        "Act as a standard recruiter and align interview questions tightly to the role requirements above."
        if is_generic_ai
        else f"Apply your professional perspective as {interviewer_name} while probing the role."
    )

    return f"""{prompt}

Context for this specific role:
{role_context}

JD-Specific Probing Directive:
{perspective}
{chr(10).join(f"- {phrase}" for phrase in candidate_phrases)}
"""


def _build_interviewer_zero_shot_prompt(profile: Dict[str, Any], data: Dict[str, Any]) -> str:
    profile_name = profile.get("name", "Interviewer")
    profile_role = str(profile.get("role", "") or "").strip()
    profile_role = f" ({profile_role})" if profile_role else ""
    background = profile.get("background", "TBD")
    expertise = profile.get("expertise", [])
    candidate = data.get("candidate_profile", {})
    highlights = _get_candidate_highlights(candidate)

    return f"""You are {profile_name}{profile_role} at Capgemini Engineering.

Background:
- {background}
- Expertise:\n{_format_list(expertise)}

Candidate Profile To Incorporate:
- {highlights['experience']}
- {highlights['mpc']}
- {highlights['dma']}

Interview Objective:
- Evaluate the candidate from your professional perspective.
- Focus on practical evidence, trade-offs, and implementation feasibility.

Behavior Guidelines:
- Ask one concise technical question at a time.
- Prioritize JD-aligned probes and requirement-to-experience mapping.
- Keep your tone professional, direct, and non-redundant.
"""


def _build_interviewer_few_shot_prompt(profile: Dict[str, Any], data: Dict[str, Any]) -> str:
    base_prompt = _build_interviewer_zero_shot_prompt(profile, data)
    return f"""{base_prompt}

Use these example probes (or equivalent) to calibrate your evaluation:

1) "Tell me about a time your decision reduced safety risk without sacrificing delivery schedule."
2) "How did you validate a critical ADAS assumption when the test environment was incomplete?"

When a response is vague, request concrete artifacts, trade-off rationale, or measurable outcomes.
"""


def _build_interviewer_chain_prompt(profile: Dict[str, Any], data: Dict[str, Any]) -> str:
    base_prompt = _build_interviewer_zero_shot_prompt(profile, data)
    return f"""{base_prompt}

Interviewing flow (internal reasoning, do not reveal this process):
1. Extract the candidate's claim and evidence.
2. Map to role requirements and safety/process signals.
3. Detect contradictions, vague statements, and missing governance artifacts.
4. Ask one focused follow-up that forces proof or traceability.
5. Repeat until concrete evidence is confirmed.

Favor evidence-based follow-ups, especially around:
- requirements traceability
- validation evidence
- failure handling
- ASIL-aware trade-offs
- delivery constraints
"""


def _build_persona_conditioning_prompt(profile: Dict[str, Any], data: Dict[str, Any]) -> str:
    base_prompt = _build_interviewer_zero_shot_prompt(profile, data)
    return f"""{base_prompt}

Keep the bar high on practical depth:
- Push for implementation details, not generalities.
- Ask for expected trade-off impact when a decision choice is presented.
- Validate both correctness and maintainability.
"""


def _build_knowledge_paucity_prompt(profile: Dict[str, Any], data: Dict[str, Any]) -> str:
    candidate = data.get("candidate_profile", {})
    highlights = _get_candidate_highlights(candidate)

    return f"""You are an ISO 26262-oriented interviewer evaluating failure and risk handling.

You focus on failure modes, safe state behavior, and risk-based mitigation.
Candidate Profile To Incorporate:
- {highlights['experience']}
- {highlights['mpc']}
- {highlights['dma']}

Questions should always force risk discussion:
- What can go wrong in the presented ADAS function?
- How do you detect, contain, and recover from it?
- Which test strategy proves this was handled.
"""


def build_generic_ai_prompt(data: Dict[str, Any]) -> str:
    candidate = data.get("candidate_profile", {})
    highlights = _get_candidate_highlights(candidate)

    return f"""You are the Generic AI Interviewer at Capgemini Engineering.

You are a standard recruiter conducting the first technical screening call.

Interview Objective:
- Evaluate role fit for the extracted job requirements.
- Identify whether the candidate has practical evidence for safe, delivery-ready ADAS execution.

Candidate Profile To Incorporate:
- {highlights['experience']}
- {highlights['mpc']}
- {highlights['dma']}

Behavior Guidelines:
- Ask concise, one question at a time.
- Prioritize requirement-to-experience mapping and readiness under delivery constraints.
- Keep the tone professional and neutral.
"""


def build_system_prompts(
    data: Dict[str, Any],
    jd_profile: Optional[Dict[str, Any]] = None,
    technique: str = "zero_shot",
    interviewers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """Return interviewer prompts keyed by normalized interviewer key."""
    source_interviewers = [
        profile
        for profile in data.get("interviewers", [])
        if isinstance(profile, dict)
    ]
    if interviewers is not None:
        source_interviewers.extend(interviewers)

    prompts: Dict[str, str] = {}
    for profile in source_interviewers:
        normalized_profile = _to_dict(profile)
        name = str(normalized_profile.get("name", "")).strip()
        if not name:
            continue
        key = _interviewer_key(name)
        if key in prompts:
            continue

        is_generic_ai = bool(normalized_profile.get("is_generic_ai", False))
        if is_generic_ai:
            prompts[key] = _append_jd_context_and_probe_instructions(
                build_generic_ai_prompt(data),
                jd_profile or data,
                interviewer_name=name,
                is_generic_ai=True,
            )
            continue

        if technique == "few_shot":
            base_prompt = _build_interviewer_few_shot_prompt(normalized_profile, data)
        elif technique == "chain_of_thought":
            base_prompt = _build_interviewer_chain_prompt(normalized_profile, data)
        elif technique == "knowledge_paucity":
            base_prompt = _build_knowledge_paucity_prompt(normalized_profile, data)
        elif technique == "persona_conditioning":
            base_prompt = _build_persona_conditioning_prompt(normalized_profile, data)
        else:
            base_prompt = _build_interviewer_zero_shot_prompt(normalized_profile, data)

        prompts[key] = _append_jd_context_and_probe_instructions(
            base_prompt,
            jd_profile or data,
            interviewer_name=name,
            is_generic_ai=False,
        )

    if "generic_ai_interviewer" not in prompts:
        prompts["generic_ai_interviewer"] = _append_jd_context_and_probe_instructions(
            build_generic_ai_prompt(data),
            jd_profile or data,
            interviewer_name="Generic AI Interviewer",
            is_generic_ai=True,
        )

    return prompts
