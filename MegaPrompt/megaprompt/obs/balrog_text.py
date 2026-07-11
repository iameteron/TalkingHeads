from __future__ import annotations

from typing import Any

from ._shared import render_balrog_text


def render(state: Any) -> str:
    return render_balrog_text(state)
