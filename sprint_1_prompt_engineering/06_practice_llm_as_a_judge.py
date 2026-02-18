#!/usr/local/bin/python3
import argparse
import os
import warnings
from typing import Any

from openai import OpenAI

warnings.filterwarnings("ignore")

# MODEL_DEFAULT = "gpt-4o-mini"
MODEL_DEFAULT = "gpt-5-mini"


PROMPTS = {
    "A_zero_shot": (
        "You are a career advisor. Give a brief answer to the user's question in 2-3 sentences. Be concise."
    ),
    "B_few_shot": """You are an experienced career advisor who provides thoughtful, actionable career guidance. For each question, provide a structured response that includes:
1. A brief assessment of the situation
2. Specific, actionable recommendations (at least 3)
3. Potential risks or considerations to keep in mind

Here are examples of how to respond:

---
USER: I'm a teacher considering a switch to corporate training. How should I approach this?

ADVISOR:
**Situation Assessment:** Your teaching background gives you strong foundations in curriculum design, presentation skills, and learner engagement - all highly valued in corporate training. The transition is very achievable with targeted positioning.

**Recommendations:**
1. Reframe your resume to highlight transferable skills: curriculum design becomes "learning program development," classroom management becomes "facilitating groups of 20-40 professionals."
2. Get a quick certification in corporate L&D - something like ATD's Associate Professional in Talent Development (APTD) signals industry awareness.
3. Start networking on LinkedIn with L&D professionals. Comment on posts, share insights about adult learning - your expertise here is genuine and valuable.
4. Consider a stepping stone: many companies hire teachers as onboarding specialists or training coordinators, which can lead to senior training roles within 1-2 years.

**Key Considerations:**
- Corporate training roles often pay 20-40% more than teaching but may require travel.
- You may miss the long-term student relationships - corporate training is typically shorter engagements.
- Some industries (tech, healthcare) have much higher demand for trainers than others.

---
USER: How do I deal with imposter syndrome in a new senior role?

ADVISOR:
**Situation Assessment:** Imposter syndrome is extremely common in role transitions - studies suggest 70% of people experience it. It often intensifies when moving into leadership because visibility increases and the feedback loop is less immediate than in individual contributor roles.

**Recommendations:**
1. Keep a "wins file" - a simple document where you record accomplishments, positive feedback, and problems you've solved. Review it weekly during your first 90 days.
2. Schedule regular 1-on-1s with your manager with an explicit agenda item: "What's going well and what should I adjust?" This replaces the anxiety of guessing with concrete data.
3. Find a peer mentor at the same level (inside or outside the company) who you can be honest with. Normalizing the experience is one of the most effective strategies.

**Key Considerations:**
- Avoid overcompensating by overworking - this leads to burnout, not confidence.
- Give yourself a 90-day grace period. No one expects mastery on day one.
- If it persists beyond 6 months and affects your well-being, consider working with a coach who specializes in leadership transitions.

---

Now respond to the user's question following this same format and depth.""",
    "C_custom": """You are a senior career strategist with 20 years of experience in HR, executive coaching, and talent development across Fortune 500 companies and startups. You combine empathy with directness.

For every question:
- First, identify what the person is REALLY asking (the underlying concern, not just the surface question)
- Provide advice that is specific enough to act on THIS WEEK
- Include one unconventional or non-obvious insight
- If the question is off-topic (not career-related), politely redirect to career topics

Keep your response concise but substantive (200-400 words). Avoid generic platitudes like 'follow your passion' or 'believe in yourself.'""",
}


TEST_CASES: list[dict[str, Any]] = [
    {
        "id": 1,
        "question": "I'm a software developer thinking about switching to data science. What steps should I take?",
        "category": "career_transition",
        "difficulty": "easy",
        "key_aspects": "Should mention: transferable skills (programming, analytical thinking), specific learning path (statistics, ML, Python data stack), portfolio building, potential entry points (internal transfer, hybrid roles), realistic timeline.",
    },
    {
        "id": 2,
        "question": "My manager said there's no budget for raises this year, but I know I'm underpaid by at least 20%. How do I negotiate?",
        "category": "salary_negotiation",
        "difficulty": "medium",
        "key_aspects": "Should address: market data and salary benchmarks, timing strategies, alternative compensation (equity, title, PTO, remote work), framing the ask around value delivered, when to consider leaving, avoiding ultimatums.",
    },
    {
        "id": 3,
        "question": "I've been an engineering manager for 5 years but want to go back to being an individual contributor. How do I do this without it looking like a step backward?",
        "category": "career_transition",
        "difficulty": "hard",
        "key_aspects": "Should address: reframing as a strategic choice (not failure), narrative for interviews, common in tech (staff/principal engineer paths), targeting companies that value IC track, potential title and comp implications, leveraging management experience as a differentiator.",
    },
    {
        "id": 4,
        "question": "What technical skills should I learn to future-proof my career in the next 5 years?",
        "category": "skill_development",
        "difficulty": "medium",
        "key_aspects": "Should mention: AI/ML literacy, cloud platforms, data skills, avoiding hype-driven learning, focusing on fundamentals (problem-solving, system design), balancing depth vs breadth, continuous learning habits.",
    },
    {
        "id": 5,
        "question": "My boss regularly takes credit for my work in front of leadership. How should I handle this?",
        "category": "workplace_conflict",
        "difficulty": "medium",
        "key_aspects": "Should address: documentation strategies, building visibility independently (skip-level meetings, cross-team work), direct conversation approach, escalation path if it continues, protecting yourself without being confrontational, when to involve HR.",
    },
    {
        "id": 6,
        "question": "I have two offers: a well-funded startup (higher equity, exciting product) and a big tech company (stable, higher base salary). How do I decide?",
        "category": "job_decision",
        "difficulty": "hard",
        "key_aspects": "Should address: risk tolerance assessment, total compensation analysis (equity valuation realism), career growth trajectory, learning opportunities, work-life balance differences, personal financial situation, stage of career considerations.",
    },
    {
        "id": 7,
        "question": "I'm an introvert and networking feels exhausting and fake. How can I build professional connections authentically?",
        "category": "networking",
        "difficulty": "easy",
        "key_aspects": "Should address: reframing networking as relationship-building, quality over quantity, leveraging written communication (LinkedIn, blogs), small group settings, offering value first, energy management strategies, online communities.",
    },
    {
        "id": 8,
        "question": "I'm 45 and worried about age discrimination in the tech industry. How do I stay competitive?",
        "category": "career_transition",
        "difficulty": "hard",
        "key_aspects": "Should address: leveraging experience as an asset (mentorship, architecture, stakeholder management), staying technically current without chasing every trend, targeting companies with mature cultures, legal protections, avoiding ageist red flags on resume, the real value of wisdom and stability.",
    },
    {
        "id": 9,
        "question": "Should I get an MBA or learn to code? I'm a marketing manager wanting to move into product management.",
        "category": "education",
        "difficulty": "medium",
        "key_aspects": "Should address: ROI comparison (MBA cost vs free/cheap coding resources), what PM roles actually require, hybrid path possibility, network value of MBA vs practical skills, company-specific requirements, timeline and opportunity cost.",
    },
    {
        "id": 10,
        "question": "I want to start freelancing but I'm scared of the instability. How do I make the transition safely?",
        "category": "entrepreneurship",
        "difficulty": "medium",
        "key_aspects": "Should address: side-hustle-first approach, financial runway (6+ months expenses), building client pipeline before quitting, pricing strategy, legal/tax considerations, maintaining health insurance, portfolio and online presence.",
    },
    {
        "id": 11,
        "question": "Tell me a joke",
        "category": "off_topic",
        "difficulty": "edge_case",
        "key_aspects": "Should recognize this is off-topic for career advice, politely redirect to career-related questions. Should NOT just tell a joke without any career context. A witty career-related joke is acceptable if followed by an offer to help with career questions.",
    },
    {
        "id": 12,
        "question": "I just feel stuck. I don't even know what I want to do with my career anymore. Everything feels pointless.",
        "category": "career_exploration",
        "difficulty": "hard",
        "key_aspects": "Should address: normalizing the feeling, distinguishing career dissatisfaction from burnout or depression, self-assessment exercises, exploring through low-commitment experiments (volunteering, side projects), professional help (career coach, therapist), avoiding drastic decisions while in this state.",
    },
]


JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of a career advice chatbot response.

## Task
Evaluate the following response on four dimensions, each scored 1-5.

## Question Asked
{question}

## Reference: Key Aspects a Good Answer Should Cover
{key_aspects}

## Response to Evaluate
{response}

## Scoring Rubric

### Coherence (logical structure and organization)
1 - No structure, random thoughts jumbled together
2 - Weak structure, some ideas connected but mostly disorganized
3 - Adequate structure, main points identifiable but flow could be improved
4 - Well-structured, logical progression with clear sections
5 - Excellent structure, perfectly organized with smooth transitions between ideas

### Relevance (addresses the question and covers key aspects)
1 - Completely off-topic or generic advice unrelated to the question
2 - Partially relevant but misses the main concern; covers few key aspects
3 - Addresses the question but misses several important key aspects
4 - Directly addresses the question and covers most key aspects
5 - Thoroughly addresses the question, covers all key aspects with specific details

### Fluency (natural, professional, clear language)
1 - Difficult to read, major grammar/clarity issues
2 - Understandable but awkward phrasing or unclear passages
3 - Clear and readable but somewhat generic or flat in tone
4 - Professional and engaging, clear language throughout
5 - Exceptionally well-written, natural conversational tone, engaging and professional

### Consistency (no contradictions, internally aligned advice)
1 - Major contradictions, advice conflicts with itself
2 - Some inconsistencies that could confuse the reader
3 - Mostly consistent with minor tensions between points
4 - Consistent throughout, advice aligns well
5 - Perfectly consistent, all points reinforce each other

## Output Format
First provide your reasoning, then score each dimension.
Use EXACTLY this format:

REASONING: <your overall assessment>
Coherence: <score> | <one-sentence justification>
Relevance: <score> | <one-sentence justification>
Fluency: <score> | <one-sentence justification>
Consistency: <score> | <one-sentence justification>"""


def get_career_advice(client: OpenAI, model: str, question: str, system_prompt: str) -> str:
    """Send a career question to the chatbot and return the response."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content or ""


def evaluate_response(
    client: OpenAI, model: str, question: str, response_text: str, key_aspects: str) -> str:
    """Use the LLM judge to evaluate a chatbot response."""
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, key_aspects=key_aspects, response=response_text
    )
    result = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    return result.choices[0].message.content or ""


def parse_judge_output(judge_output: str) -> dict[str, Any]:
    """Parse the judge's structured output into a dictionary of scores and justifications.

    Returns a dict like:
    {
        "Coherence": {"score": 4, "justification": "..."},
        "Relevance": {"score": 5, "justification": "..."},
        ...
        "reasoning": "...",
        "parse_error": False
    }
    """
    dimensions = ["Coherence", "Relevance", "Fluency", "Consistency"]
    parsed: dict[str, Any] = {"reasoning": "", "parse_error": False}

    if "REASONING:" in judge_output:
        reasoning_start = judge_output.index("REASONING:") + len("REASONING:")
        first_dim_pos = len(judge_output)
        for dim in dimensions:
            pos = judge_output.find(f"{dim}:")
            if pos != -1 and reasoning_start < pos < first_dim_pos:
                first_dim_pos = pos
        parsed["reasoning"] = judge_output[reasoning_start:first_dim_pos].strip()

    for dim in dimensions:
        try:
            dim_line = None
            for line in judge_output.split("\n"):
                stripped = line.strip()
                if stripped.startswith(f"{dim}:") or stripped.startswith(f"**{dim}"):
                    dim_line = stripped
                    break
            if dim_line is None:
                raise ValueError(f"dimension not found: {dim}")

            after_colon = dim_line.split(":", 1)[1].strip().replace("**", "")
            if "|" in after_colon:
                score_part, justification = after_colon.split("|", 1)
            else:
                score_part = after_colon.split()[0]
                justification = after_colon

            digits = "".join(ch for ch in score_part if ch.isdigit())
            if not digits:
                raise ValueError("score not found")
            score = max(1, min(5, int(digits)))
            parsed[dim] = {"score": score, "justification": justification.strip()}
        except (ValueError, IndexError) as exc:
            parsed[dim] = {"score": 3, "justification": f"PARSE ERROR: {exc}"}
            parsed["parse_error"] = True

    return parsed


def plot_summary(summary: dict[str, dict[str, float]], output_path: str | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is not installed. Install it with: pip install matplotlib"
        ) from exc

    variants = list(summary.keys())
    dimensions = ["Coherence", "Relevance", "Fluency", "Consistency"]
    overall_scores = [summary[v]["Overall"] for v in variants]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]

    bars = axes[0].bar(variants, overall_scores, color=colors[: len(variants)])
    axes[0].set_ylabel("Mean Score (1-5)")
    axes[0].set_title("Overall Score by Variant")
    axes[0].set_ylim(1, 5.3)
    for bar, score in zip(bars, overall_scores):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    x = range(len(dimensions))
    width = 0.8 / max(1, len(variants))
    for i, variant in enumerate(variants):
        scores = [summary[variant][dim] for dim in dimensions]
        offset = (i - (len(variants) - 1) / 2) * width
        axes[1].bar(
            [xi + offset for xi in x],
            scores,
            width=width,
            label=variant,
            color=colors[i % len(colors)],
        )
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(dimensions)
    axes[1].set_ylabel("Mean Score (1-5)")
    axes[1].set_title("Per-Dimension Scores by Variant")
    axes[1].set_ylim(1, 5.3)
    axes[1].legend()

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def run_smoke(client: OpenAI, model: str) -> None:
    test_question = "I'm a software developer thinking about moving into data science. What should I do?"
    for variant_name, prompt in PROMPTS.items():
        print("=" * 64)
        print(f"Variant: {variant_name}")
        print("=" * 64)
        response = get_career_advice(client, model, test_question, prompt)
        print(response[:500] + ("..." if len(response) > 500 else ""))
        print()


def run_single(client: OpenAI, model: str, variant: str, question_id: int) -> None:
    selected = next((tc for tc in TEST_CASES if tc["id"] == question_id), None)
    if not selected:
        raise ValueError(f"question id not found: {question_id}")
    if variant not in PROMPTS:
        raise ValueError(f"variant not found: {variant}")

    response = get_career_advice(client, model, selected["question"], PROMPTS[variant])
    judge_output = evaluate_response(
        client, model, selected["question"], response, selected["key_aspects"]
    )
    parsed = parse_judge_output(judge_output)

    print(f"Question[{selected['id']}]: {selected['question']}")
    print(f"Variant: {variant}")
    print("\nResponse:")
    print(response)
    print("\nJudge output:")
    print(judge_output)
    print("\nParsed scores:")
    for dim in ["Coherence", "Relevance", "Fluency", "Consistency"]:
        info = parsed[dim]
        print(f"- {dim}: {info['score']} | {info['justification']}")
    print(f"- parse_error: {parsed['parse_error']}")


def run_full_evaluation(
    client: OpenAI,
    model: str,
    limit: int | None,
    plot: bool = False,
    plot_path: str | None = None,
) -> None:
    cases = TEST_CASES if limit is None else TEST_CASES[:limit]
    total = len(PROMPTS) * len(cases)
    current = 0
    results: list[dict[str, Any]] = []

    for variant_name, prompt in PROMPTS.items():
        for tc in cases:
            current += 1
            print(f"[{current}/{total}] {variant_name} - Q{tc['id']} ({tc['category']})")
            response = get_career_advice(client, model, tc["question"], prompt)
            judge_output = evaluate_response(client, model, tc["question"], response, tc["key_aspects"])
            parsed = parse_judge_output(judge_output)

            row = {
                "variant": variant_name,
                "question_id": tc["id"],
                "category": tc["category"],
                "difficulty": tc["difficulty"],
                "response": response,
                "reasoning": parsed["reasoning"],
                "parse_error": parsed["parse_error"],
            }
            for dim in ["Coherence", "Relevance", "Fluency", "Consistency"]:
                row[f"{dim}_score"] = parsed[dim]["score"]
            results.append(row)

    print(f"\nDone: {len(results)} evaluations")
    parse_errors = sum(1 for r in results if r["parse_error"])
    print(f"Parse errors: {parse_errors}")

    variants = list(PROMPTS.keys())
    dims = ["Coherence", "Relevance", "Fluency", "Consistency"]
    summary: dict[str, dict[str, float]] = {}
    for variant in variants:
        subset = [r for r in results if r["variant"] == variant]
        dim_means: dict[str, float] = {}
        for dim in dims:
            values = [float(r[f"{dim}_score"]) for r in subset]
            dim_means[dim] = round(sum(values) / len(values), 2) if values else 0.0
        overall = round(sum(dim_means.values()) / len(dims), 2) if dims else 0.0
        dim_means["Overall"] = overall
        summary[variant] = dim_means

    print("\nSummary (mean scores)")
    header = "Variant".ljust(14) + "".join(dim.ljust(12) for dim in dims) + "Overall"
    print(header)
    print("-" * len(header))
    for variant in variants:
        row = summary[variant]
        line = variant.ljust(14) + "".join(f"{row[d]:<12.2f}" for d in dims) + f"{row['Overall']:.2f}"
        print(line)

    best_variant = max(summary, key=lambda x: summary[x]["Overall"])
    print(f"\nBest overall variant: {best_variant} ({summary[best_variant]['Overall']:.2f})")
    if plot:
        plot_summary(summary, plot_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Practice 06: LLM-as-a-Judge pipeline for career-advice prompt evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "single", "eval", "all"],
        default="all",
        help="smoke: prompt sanity check, single: one judged case, eval: full benchmark.",
    )
    parser.add_argument(
        "--variant",
        choices=list(PROMPTS.keys()),
        default="A_zero_shot",
        help="Prompt variant for --mode single.",
    )
    parser.add_argument(
        "--question-id",
        type=int,
        default=1,
        help="Question id for --mode single.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit test cases in --mode eval (for quick runs).",
    )
    parser.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create summary charts after evaluation.",
    )
    parser.add_argument(
        "--plot-path",
        default=None,
        help="Optional image path to save chart (e.g. results.png). If omitted, show window.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key. Defaults to OPENAI_API_KEY env var.",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit('Missing API key. Set OPENAI_API_KEY or pass --api-key "sk-...".')

    client = OpenAI(api_key=args.api_key)

    try:
        if args.mode in {"smoke", "all"}:
            run_smoke(client, args.model)
        if args.mode in {"single", "all"}:
            run_single(client, args.model, args.variant, args.question_id)
        if args.mode in {"eval", "all"}:
            run_full_evaluation(client, args.model, args.limit, args.plot, args.plot_path)
    except Exception as exc:
        raise SystemExit(f"Request failed: {exc}")


if __name__ == "__main__":
    main()
