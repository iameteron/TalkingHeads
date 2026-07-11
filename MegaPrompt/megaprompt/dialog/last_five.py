from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _step_suffix(turn: dict[str, Any], key: str, label: str) -> str:
    try:
        step = int(turn.get(key) or 0)
    except (TypeError, ValueError):
        step = 0
    return f" ({label} at step {step})" if step > 0 else ""


def render(dialog: Any) -> str:
    if dialog is None:
        return ""
    if isinstance(dialog, str):
        return dialog.strip()
    if isinstance(dialog, dict):
        history = [dialog]
    elif isinstance(dialog, Iterable):
        history = [turn for turn in dialog if isinstance(turn, dict)]
    else:
        return str(dialog).strip()

    lines: list[str] = []
    for turn in history[-2:]:
        agent_msg = str(
            turn.get("agent")
            or turn.get("question")
            or (turn.get("content") if str(turn.get("role", "")).lower() in {"user", "agent"} else "")
            or ""
        ).strip()
        oracle_msg = str(
            turn.get("oracle")
            or turn.get("answer")
            or (
                turn.get("content")
                if str(turn.get("role", "")).lower() in {"assistant", "oracle", "operator"}
                else ""
            )
            or ""
        ).strip()
        if agent_msg:
            lines.append(f"Agent{_step_suffix(turn, 'question_step', 'asked')}: {agent_msg}")
        if oracle_msg:
            lines.append(f"Operator{_step_suffix(turn, 'answer_step', 'answered')}: {oracle_msg}")
    return "\n".join(lines).strip()
