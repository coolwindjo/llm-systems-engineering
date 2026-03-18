from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = [
    PROJECT_ROOT / "app.py",
    *sorted((PROJECT_ROOT / "services").glob("*.py")),
    *sorted((PROJECT_ROOT / "components").glob("*.py")),
]


def test_project_functions_stay_under_200_lines() -> None:
    oversized_functions: list[str] = []

    for path in SOURCE_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                line_count = (node.end_lineno or node.lineno) - node.lineno + 1
                if line_count > 200:
                    oversized_functions.append(f"{path.name}:{node.name}:{line_count}")

    assert oversized_functions == []


def test_readme_stays_concise_for_reviewer_scanability() -> None:
    readme_path = PROJECT_ROOT / "README.md"
    line_count = len(readme_path.read_text(encoding="utf-8").splitlines())

    assert line_count <= 120
