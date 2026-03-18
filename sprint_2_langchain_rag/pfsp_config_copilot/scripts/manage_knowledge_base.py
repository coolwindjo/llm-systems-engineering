from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.config import DEFAULT_MODEL_NAME, get_openai_api_key, load_local_env  # noqa: E402
from services.knowledge_base import (  # noqa: E402
    StandardsKnowledgeBase,
    list_reference_documents,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List, reset, or delete documents from the Chroma-backed standards knowledge base."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List stored reference files.")
    subparsers.add_parser("reset", help="Reset and rebuild the vectorstore from current Markdown files.")

    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete a stored reference file and rebuild the vectorstore.",
    )
    delete_parser.add_argument(
        "document_name",
        help="Stored file name under data/standards/, for example ISO_DIS_26262-6(en).PDF",
    )
    return parser.parse_args()


def _build_knowledge_base() -> StandardsKnowledgeBase:
    load_local_env()
    api_key = get_openai_api_key()
    return StandardsKnowledgeBase(api_key=api_key)


def main() -> int:
    args = parse_args()

    if args.command == "list":
        result = {
            "documents": list_reference_documents(),
        }
    elif args.command == "reset":
        kb = _build_knowledge_base()
        result = kb.reset_knowledge_base()
        result["model_name"] = DEFAULT_MODEL_NAME
    else:
        kb = _build_knowledge_base()
        result = kb.delete_document_and_rebuild(args.document_name)
        result["model_name"] = DEFAULT_MODEL_NAME

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
