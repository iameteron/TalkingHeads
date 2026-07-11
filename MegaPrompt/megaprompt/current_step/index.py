from __future__ import annotations

from typing import Any


def render(value: Any) -> str:
    try:
        step = int(value)
    except (TypeError, ValueError):
        step = 1
    return str(max(1, step))
