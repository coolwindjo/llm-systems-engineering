from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.config import DEFAULT_MODEL_NAME, get_openai_api_key, load_local_env  # noqa: E402
from services.knowledge_base import StandardsKnowledgeBase  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a PDF or Markdown file into the standards corpus and rebuild the Chroma vectorstore."
    )
    parser.add_argument("document_path", type=Path, help="Path to the source PDF or Markdown file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_local_env()
    api_key = get_openai_api_key()
    kb = StandardsKnowledgeBase(api_key=api_key)
    result = kb.import_document_and_rebuild(args.document_path)
    result["model_name"] = DEFAULT_MODEL_NAME
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
