from __future__ import annotations

from typing import Any, Mapping


def _get(obs: Any, key: str, default: Any = "") -> Any:
    if isinstance(obs, Mapping):
        return obs.get(key, default)
    return getattr(obs, key, default)


def render(obs: Any) -> str:
    game_id = str(_get(obs, "game_id", "")).strip()
    title = str(_get(obs, "title", game_id)).strip() or game_id
    state = str(_get(obs, "state", "UNKNOWN")).strip()
    levels_completed = _get(obs, "levels_completed", 0)
    actions = _get(obs, "available_actions", []) or []
    width = _get(obs, "w", 0)
    height = _get(obs, "h", 0)
    current_png_b64 = str(_get(obs, "png_b64", "")).strip()
    previous_width = _get(obs, "previous_w", width)
    previous_height = _get(obs, "previous_h", height)
    previous_png_b64 = str(_get(obs, "previous_png_b64", "")).strip()
    previous_action = str(_get(obs, "previous_action", "")).strip()

    lines = [
        f"Game: {game_id} ({title})".strip(),
        f"State: {state}",
        f"Levels completed: {levels_completed}",
        f"Available actions: {', '.join(str(a) for a in actions) or 'none'}",
        "Coordinate convention: x=column 0..63 left-to-right, y=row 0..63 top-to-bottom.",
    ]
    if previous_action:
        lines.append(f"Latest environment action before the current frame: {previous_action}.")
    if previous_png_b64:
        lines.extend(
            [
                "",
                f"Previous observation image: {previous_width}x{previous_height} PNG, attached below.",
                f"[[image:data:image/png;base64,{previous_png_b64}]]",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Previous observation image: unavailable on the first tick or after reset.",
            ]
        )
    if current_png_b64:
        lines.extend(
            [
                "",
                f"Current observation image: {width}x{height} PNG, attached below.",
                f"[[image:data:image/png;base64,{current_png_b64}]]",
            ]
        )
    else:
        lines.append("Current observation image unavailable.")
    return "\n".join(lines).strip()
