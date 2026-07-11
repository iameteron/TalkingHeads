from __future__ import annotations

from typing import Any


def render(goal: Any) -> str:
    if goal is None:
        return ""
    return str(goal).strip()
