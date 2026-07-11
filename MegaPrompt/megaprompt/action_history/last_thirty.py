from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _action_text(item: Any) -> str:
    if isinstance(item, dict):
        raw = item.get("action") or item.get("action_raw") or item.get("name") or ""
    else:
        raw = item
    return " ".join(str(raw or "").strip().split())


def render(actions: Any) -> str:
    if actions is None:
        return "No actions have been executed yet."
    if isinstance(actions, str):
        history = [_action_text(actions)]
    elif isinstance(actions, Iterable):
        history = [_action_text(item) for item in actions]
    else:
        history = [_action_text(actions)]
    history = [item for item in history if item]
    if not history:
        return "No actions have been executed yet."

    window = history[-30:]
    lines = [
        f"{idx}. {action}"
        for idx, action in enumerate(window, 1)
    ]
    return "\n".join(lines)
