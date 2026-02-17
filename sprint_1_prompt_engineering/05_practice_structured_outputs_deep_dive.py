#!/usr/local/bin/python3
import argparse
import json
import os
import warnings
from enum import IntEnum
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")


class Webpage(BaseModel):
    title: str = Field(description="Page title text.")
    paragraphs: Optional[list[str]] = Field(
        None, description="Text found in <p></p> tags. Use None when no paragraphs exist."
    )
    links: Optional[list[str]] = Field(
        None, description="URL values from href in <a></a> tags. Use None when no links exist."
    )
    images: Optional[list[str]] = Field(
        None, description="URL values from src in <img> tags. Use None when no images exist."
    )


class Rank(IntEnum):
    RANK_1 = 1
    RANK_2 = 2
    RANK_3 = 3
    RANK_4 = 4
    RANK_5 = 5


class RerankingResult(BaseModel):
    ordered_ranking: list[Rank] = Field(description="Ordered ranking from most similar to least similar.")


class TwoPhaseReasoning(BaseModel):
    rationale: str = Field(description="Short explanation for why each rank was chosen.")
    ordered_ranking: list[Rank] = Field(description="Ordered ranking from most similar to least similar.")


HTML_INPUT = """
<html>
  <title>Structured Outputs Demo</title>
  <body>
    <img src="test.gif" />
    <p>Hello world!</p>
  </body>
</html>
""".strip()

PRODUCT_INPUT = """
## Target Product
Product ID: X56HHGHH
Product Description: 80\" Samsung LED TV

## Candidate Products
1) Product ID: 125GHJJJGH
   Product Description: NVIDIA RTX 4060 GPU

2) Product ID: 76876876GHJ
   Product Description: Sony Walkman

3) Product ID: 433FGHHGG
   Product Description: Sony LED TV 56\"

4) Product ID: 777888887888
   Product Description: Blueray Sony Player

5) Product ID: JGHHJGJ56
   Product Description: BenQ PC Monitor 37\" 4K UHD
""".strip()


def validate_ranking(ranks: list[int]) -> None:
    if len(ranks) != 5:
        raise ValueError("ordered_ranking must include exactly 5 items.")
    if sorted(ranks) != [1, 2, 3, 4, 5]:
        raise ValueError("ordered_ranking must be a permutation of [1,2,3,4,5].")


def optional_fields_demo(client: OpenAI, model: str) -> Webpage:
    response = client.responses.parse(
        model=model,
        instructions="Parse the HTML and return page components exactly in the schema.",
        input=HTML_INPUT,
        text_format=Webpage,
    )
    return response.output_parsed


def enum_reranking_demo(client: OpenAI, model: str) -> RerankingResult:
    response = client.responses.parse(
        model=model,
        instructions=(
            "Rank candidate products by similarity to the target product. "
            "Return only ranking indices in descending similarity order."
        ),
        input=PRODUCT_INPUT,
        text_format=RerankingResult,
    )
    parsed = response.output_parsed
    validate_ranking([int(x) for x in parsed.ordered_ranking])
    return parsed


def two_phase_demo(client: OpenAI, model: str) -> TwoPhaseReasoning:
    reasoning_response = client.responses.create(
        model=model,
        instructions=(
            "Reason step-by-step and produce a concise natural language answer with: "
            "(1) similarity reasoning and (2) final ranking list using candidate numbers 1..5."
        ),
        input=PRODUCT_INPUT,
    )
    free_text = reasoning_response.output_text

    structuring_response = client.responses.parse(
        model=model,
        instructions=(
            "Convert the provided analysis into the target schema. "
            "Do not invent new candidates. Preserve the ranking intent."
        ),
        input=free_text,
        text_format=TwoPhaseReasoning,
    )
    parsed = structuring_response.output_parsed
    validate_ranking([int(x) for x in parsed.ordered_ranking])
    return parsed


def run(mode: str, client: OpenAI, model: str) -> None:
    if mode in {"optional", "all"}:
        optional_result = optional_fields_demo(client, model)
        print("--- Optional fields demo (HTML parsing) ---")
        print(optional_result.model_dump_json(indent=2))

    if mode in {"enum", "all"}:
        enum_result = enum_reranking_demo(client, model)
        print("--- Enum reranking demo ---")
        print(enum_result.model_dump_json(indent=2))

    if mode in {"two-phase", "all"}:
        two_phase_result = two_phase_demo(client, model)
        print("--- Two-phase reasoning + structuring demo ---")
        print(two_phase_result.model_dump_json(indent=2))



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Practice 05: Deeper structured outputs (Optional fields, Enums, and two-phase pipeline)."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["optional", "enum", "two-phase", "all"],
        default="all",
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model (for example: gpt-4o-mini or gpt-4o).",
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
        run(args.mode, client, args.model)
    except Exception as exc:
        raise SystemExit(f"Request failed: {exc}")


if __name__ == "__main__":
    main()
