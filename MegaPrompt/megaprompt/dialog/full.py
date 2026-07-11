from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any


MAX_MESSAGE_CHARS = int(os.environ.get("ARC_DIALOG_MAX_MESSAGE_CHARS", "1200"))
MAX_TOTAL_CHARS = int(os.environ.get("ARC_DIALOG_MAX_TOTAL_CHARS", "16000"))


def _compact(text: str, *, max_chars: int = MAX_MESSAGE_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(120, max_chars - 80)
    return f"{text[:keep].rstrip()} ... [truncated {len(text) - keep} chars]"


def _step_suffix(turn: dict[str, Any], key: str, label: str) -> str:
    try:
        step = int(turn.get(key) or 0)
    except (TypeError, ValueError):
        step = 0
    return f" ({label} at step {step})" if step > 0 else ""


def render(dialog: Any) -> str:
    if dialog is None:
        return "No messages from the human operator yet."
    if isinstance(dialog, str):
        text = dialog.strip()
        return text or "No messages from the human operator yet."
    if isinstance(dialog, dict):
        history = [dialog]
    elif isinstance(dialog, Iterable):
        history = [turn for turn in dialog if isinstance(turn, dict)]
    else:
        text = str(dialog).strip()
        return text or "No messages from the human operator yet."

    rendered_turns: list[list[str]] = []
    for idx, turn in enumerate(history, 1):
        agent_msg = str(
            turn.get("agent")
            or turn.get("question")
            or (turn.get("content") if str(turn.get("role", "")).lower() in {"user", "agent"} else "")
            or ""
        ).strip()
        operator_msg = str(
            turn.get("oracle")
            or turn.get("answer")
            or (
                turn.get("content")
                if str(turn.get("role", "")).lower() in {"assistant", "oracle", "operator"}
                else ""
            )
            or ""
        ).strip()
        turn_lines: list[str] = []
        if agent_msg:
            turn_lines.append(
                f"Turn {idx} Agent{_step_suffix(turn, 'question_step', 'asked')}: {_compact(agent_msg)}"
            )
        if operator_msg:
            turn_lines.append(
                f"Turn {idx} Human operator{_step_suffix(turn, 'answer_step', 'answered')}: {_compact(operator_msg)}"
            )
        if turn_lines:
            rendered_turns.append(turn_lines)

    lines: list[str] = []
    total = 0
    omitted = 0
    for turn_lines in reversed(rendered_turns):
        turn_text = "\n".join(turn_lines)
        next_total = total + len(turn_text) + (2 if lines else 0)
        if lines and next_total > MAX_TOTAL_CHARS:
            omitted += 1
            continue
        lines[:0] = turn_lines + ([""] if lines else [])
        total = next_total
    if omitted:
        lines[:0] = [f"[{omitted} older dialog turn(s) omitted to fit the ARC prompt budget.]", ""]
    return "\n".join(lines).strip() or "No messages from the human operator yet."
