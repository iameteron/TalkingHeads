from __future__ import annotations

from collections.abc import Iterable
from typing import Any


ACTION_TEMPLATE = """
The possible list of actions you can take:
{actions}
"""


def _return_actions_text(actions: Iterable[str]) -> str:
    items = [a.strip() for a in actions if a and a.strip()]
    return "\n".join(f"- {a}" for a in items)


def render(actions: Any) -> str:
    if actions is None:
        return ""
    if isinstance(actions, str):
        return ACTION_TEMPLATE.format(actions=_return_actions_text([actions]))
    if isinstance(actions, Iterable):
        return ACTION_TEMPLATE.format(actions=_return_actions_text([str(a) for a in actions]))
    return ACTION_TEMPLATE.format(actions=_return_actions_text([str(actions)]))
