from __future__ import annotations

from typing import Any


def render(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    return text or "No repeated action pattern detected."
