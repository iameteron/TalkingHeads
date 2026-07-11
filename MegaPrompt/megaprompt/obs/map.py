from __future__ import annotations

from typing import Any

from ._shared import render_map_text


def render(state: Any) -> str:
    return render_map_text(state)
