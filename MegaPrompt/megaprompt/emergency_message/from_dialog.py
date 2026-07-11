from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..action_history._shared import QUESTION_MARKER, normalize_actions

_NO_EMERGENCY = "No emergency messages."
_EMERGENCY_REPLY = (
    "The Operator has sent a new reply. Persist any important knowledge from this answer "
    "to the Knowledge Database using UPDATE_DATABASE — the operator link may drop soon."
)


def _dialog_turns(dialog: Any) -> list[dict[str, Any]]:
    if dialog is None:
        return []
    if isinstance(dialog, dict):
        return [dialog]
    if isinstance(dialog, Iterable) and not isinstance(dialog, (str, bytes)):
        return [turn for turn in dialog if isinstance(turn, dict)]
    return []


def _operator_answer(turn: dict[str, Any]) -> str:
    return str(
        turn.get("oracle")
        or turn.get("answer")
        or (
            turn.get("content")
            if str(turn.get("role", "")).lower() in {"assistant", "oracle", "operator"}
            else ""
        )
        or ""
    ).strip()


def _last_operator_answer(dialog: Any) -> str:
    for turn in reversed(_dialog_turns(dialog)):
        answer = _operator_answer(turn)
        if answer:
            return answer
    return ""


def _is_question_marker(value: str) -> bool:
    token = value.strip().upper()
    return token == QUESTION_MARKER or token.startswith("ASK_OPERATOR")


def _normalize_action_tokens(action_history: Any) -> list[str]:
    normalized = normalize_actions(action_history)
    if normalized:
        return normalized
    if action_history is None:
        return []
    if isinstance(action_history, str):
        token = action_history.strip()
        return [token] if token else []
    if isinstance(action_history, Iterable) and not isinstance(action_history, (str, bytes)):
        tokens: list[str] = []
        for item in action_history:
            token = str(item or "").strip()
            if token:
                tokens.append(token)
        return tokens
    token = str(action_history or "").strip()
    return [token] if token else []


def _has_update_database_since_last_question(action_history: Any) -> bool:
    normalized = _normalize_action_tokens(action_history)
    if not normalized:
        return False

    last_question_idx = -1
    for idx, value in enumerate(normalized):
        if _is_question_marker(value):
            last_question_idx = idx

    window = normalized[last_question_idx + 1 :] if last_question_idx >= 0 else normalized
    return any(action.strip().upper() == "UPDATE_DATABASE" for action in window)


def render(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip() or _NO_EMERGENCY

    dialog = payload
    action_history: Any = []
    if isinstance(payload, dict):
        dialog = payload.get("dialog")
        action_history = payload.get("action_history") or []

    if not _last_operator_answer(dialog):
        return _NO_EMERGENCY
    if _has_update_database_since_last_question(action_history):
        return _NO_EMERGENCY
    return _EMERGENCY_REPLY
