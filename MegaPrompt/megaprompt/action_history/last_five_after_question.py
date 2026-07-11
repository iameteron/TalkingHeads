from __future__ import annotations

from typing import Any

from ._shared import QUESTION_MARKER, normalize_actions


def _is_question_marker(value: str) -> bool:
    token = value.strip().upper()
    return token == QUESTION_MARKER or token.startswith("ASK_OPERATOR")


def render(actions: Any) -> str:
    normalized = normalize_actions(actions)
    if not normalized:
        return "No actions were taken since the last operator question."

    # Count only the latest consecutive ASK_OPERATOR streak (from the end).
    operator_call_count = 0
    for value in reversed(normalized):
        if _is_question_marker(value):
            operator_call_count += 1
        else:
            break
    # Show the hard warning only when the latest turn is another operator call.
    # Otherwise old ASK_OPERATOR events can leak warning text into regular action history.
    is_asking_now = bool(normalized and _is_question_marker(normalized[-1]))

    last_question_idx = -1
    for idx, value in enumerate(normalized):
        if _is_question_marker(value):
            last_question_idx = idx

    # If ASK_OPERATOR is present, keep only actions after it.
    # If not present, treat all provided actions as post-question history.
    window = normalized[last_question_idx + 1 :] if last_question_idx >= 0 else normalized
    window = [
        action
        for action in window
        if (not _is_question_marker(action)) and action.strip().upper() != "NO_TAG"
    ]
    if not window:
        return "No actions were taken since the last operator question."

    lines = [f"- {action}" for action in window[-5:]]
    history_text = "\n".join(lines)

    if operator_call_count > 3 and is_asking_now:
        warning = (
            "You call operator more than 3 times! This is too much! "
            "Now it not availavle! So predict the action from action list!"
        )
        return f"{warning}\n{history_text}"
    
    if len(window) > 6:
        nudge = (
            "\n\n## Action History Ananlysis: You are moving forward without any progress! I need to contact the operator immediately! Action: ASK_OPERATOR."
       
        )
        history_text = f"{history_text}{nudge}"

    elif len(window) > 3:
        nudge = (
            "\n\n## Action History Ananlysis: You are moving forward confidently—but are you sure this is the right direction? "
            "If you are uncertain, consider asking the operator with ASK_OPERATOR."
        )
        history_text = f"{history_text}{nudge}"
    

    return history_text

