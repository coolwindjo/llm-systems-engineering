from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DummyContextManager:
    def __init__(self, value=None) -> None:
        self.value = value

    def __enter__(self):
        return self.value if self.value is not None else self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class SessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value) -> None:
        self[key] = value
