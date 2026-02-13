#!/usr/local/bin/python3
import argparse
import os
import warnings

from openai import OpenAI
from pydantic import BaseModel

warnings.filterwarnings("ignore")


class BookRecommendation(BaseModel):
    title: str
    author: str
    genre: str
    reading_time_minutes: int


SYSTEM_PROMPT = (
    "You are a librarian with expertise in book recommendations. "
    "Suggest highly-rated books that match the user's interests."
)


def get_book_recommendation(client: OpenAI, user_query: str, model: str) -> BookRecommendation:
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        response_format=BookRecommendation,
    )
    return response.choices[0].message.parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Test structured JSON-like outputs with Pydantic.")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (example: gpt-4o-mini or gpt-4o).",
    )
    parser.add_argument(
        "--query",
        default="Recommend me a science fiction book",
        help="User query for the recommendation prompt.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key. If omitted, reads OPENAI_API_KEY from env.",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit('Missing API key. Set OPENAI_API_KEY or pass --api-key "sk-...".')

    client = OpenAI(api_key=args.api_key)

    try:
        parsed = get_book_recommendation(client, args.query, args.model)
    except Exception as exc:
        raise SystemExit(f"Request failed: {exc}")

    print(parsed)
    print(parsed.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
