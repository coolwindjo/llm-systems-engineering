from __future__ import annotations

import pathlib
import sys


_CANDIDATE_PATHS = (
    pathlib.Path(__file__).resolve().parents[3],  # workspace root for package import
    pathlib.Path(__file__).resolve().parents[1],  # interview_coach root for `scripts` / local imports
)

for path in _CANDIDATE_PATHS:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
