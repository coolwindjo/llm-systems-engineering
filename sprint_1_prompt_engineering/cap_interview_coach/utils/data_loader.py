from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "interview_data.json"


def load_interview_data(path: str | Path = DEFAULT_DATA_PATH) -> Dict[str, Any]:
    """Load and return interview data from a JSON file.

    Args:
        path: JSON file path. Defaults to data/interview_data.json.

    Returns:
        Parsed JSON content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON content is not an object.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Interview data file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Interview data root must be a JSON object.")

    return data
