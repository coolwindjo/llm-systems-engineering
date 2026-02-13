#!/usr/local/bin/python3
import argparse
import json
import os
import warnings
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")


class BookRecommendation(BaseModel):
    book_title: str = Field(description="The title of the recommended book.")
    author: str = Field(description="The human author of the recommended book.")
    genre: str = Field(description="The literary genre that best matches the request.")
    reading_time_minutes: int = Field(description="Estimated total reading time in minutes.")


SYSTEM_PROMPT = (
    "You are a librarian with expertise in book recommendations. "
    "Suggest highly-rated books that match the user's interests. "
    "Return values with the correct semantics: `book_title` is the book name and `author` is the person."
)

RAW_JSON_SCHEMA_FORMAT = {
    "type": "json_schema",
    "name": "book_recommendation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "book_title": {"type": "string"},
            "author": {"type": "string"},
            "genre": {"type": "string"},
            "reading_time_minutes": {"type": "integer"},
        },
        "required": ["book_title", "author", "genre", "reading_time_minutes"],
        "additionalProperties": False,
    },
}


def is_semantically_valid(rec: BookRecommendation) -> bool:
    if rec.book_title.strip().lower() in {"bookrecommendation", "recommendation"}:
        return False
    if rec.author.strip().lower() in {"bookrecommendation", "unknown", "n/a"}:
        return False
    return True


def recommendation_with_pydantic(client: OpenAI, model: str, user_query: str) -> BookRecommendation:
    response = client.responses.parse(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=user_query,
        text_format=BookRecommendation,
    )
    return response.output_parsed


def recommendation_with_raw_schema(client: OpenAI, model: str, user_query: str) -> BookRecommendation:
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=user_query,
        text={"format": RAW_JSON_SCHEMA_FORMAT},
    )

    parsed = json.loads(response.output_text)
    return BookRecommendation.model_validate(parsed)


def get_with_retry(
    mode: Literal["pydantic", "json-schema"], client: OpenAI, model: str, user_query: str, retries: int = 1
) -> BookRecommendation:
    fn = recommendation_with_pydantic if mode == "pydantic" else recommendation_with_raw_schema
    rec = fn(client, model, user_query)
    if is_semantically_valid(rec):
        return rec

    for _ in range(retries):
        rec = fn(
            client,
            model,
            user_query
            + " Ensure `book_title` is a real book name and `author` is the actual author's name.",
        )
        if is_semantically_valid(rec):
            return rec
    return rec


def run(mode: Literal["pydantic", "json-schema", "both"], client: OpenAI, model: str, query: str) -> None:
    if mode in {"pydantic", "both"}:
        pydantic_result = get_with_retry("pydantic", client, model, query)
        print("--- Pydantic schema result ---")
        print(pydantic_result)
        print(pydantic_result.model_dump_json(indent=2))

    if mode in {"json-schema", "both"}:
        raw_schema_result = get_with_retry("json-schema", client, model, query)
        print("--- Raw JSON schema result ---")
        print(raw_schema_result)
        print(raw_schema_result.model_dump_json(indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Practice 04: Structured outputs with Pydantic and raw JSON Schema."
    )
    parser.add_argument(
        "--mode",
        choices=["pydantic", "json-schema", "both"],
        default="both",
        help="Which structured-output approach to run.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model (for example: gpt-4o-mini or gpt-4o).",
    )
    parser.add_argument(
        "--query",
        default="Recommend me a science fiction book",
        help="Prompt for book recommendation.",
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
        run(args.mode, client, args.model, args.query)
    except Exception as exc:
        raise SystemExit(f"Request failed: {exc}")


if __name__ == "__main__":
    main()
