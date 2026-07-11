from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

QUESTION_MARKER = "__ASK_OPERATOR__"
_ACTION_TAG_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)


def _extract_actions_from_text(text: str) -> list[str]:
    return [match.strip() for match in _ACTION_TAG_RE.findall(text) if match.strip()]


def normalize_actions(actions: Any) -> list[str]:
    if actions is None:
        return []
    if isinstance(actions, str):
        return _extract_actions_from_text(actions)
    if not isinstance(actions, Iterable):
        return _extract_actions_from_text(str(actions))

    normalized: list[str] = []
    for item in actions:
        if isinstance(item, str):
            normalized.extend(_extract_actions_from_text(item))
            continue
        elif isinstance(item, dict):
            question = str(item.get("question") or "").strip()
            action_values = _extract_actions_from_text(str(item.get("action") or ""))
            action = action_values[-1] if action_values else ""
            if question:
                normalized.append(QUESTION_MARKER)
        else:
            action_values = _extract_actions_from_text(str(item))
            action = action_values[-1] if action_values else ""
        if action:
            normalized.append(action)
    return normalized

