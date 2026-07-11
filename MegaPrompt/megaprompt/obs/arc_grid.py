from __future__ import annotations

from typing import Any, Mapping


_PALETTE_TEXT = (
    "0 white, 1 off-white, 2 light-gray, 3 gray, 4 dark-gray, 5 black, "
    "6 magenta, 7 light-magenta, 8 red, 9 blue, a light-blue, b yellow, "
    "c orange, d maroon, e green, f purple"
)


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
    frame_grid = str(_get(obs, "frame_grid", "")).strip()

    lines = [
        f"Game: {game_id} ({title})".strip(),
        f"State: {state}",
        f"Levels completed: {levels_completed}",
        f"Available actions: {', '.join(str(a) for a in actions) or 'none'}",
    ]
    if frame_grid:
        lines.extend(
            [
                "",
                "Current frame grid:",
                "- 64 rows x 64 columns.",
                "- Each character is a palette id in hexadecimal 0..f.",
                "- Coordinates use x=column 0..63 left-to-right and y=row 0..63 top-to-bottom.",
                f"- Palette: {_PALETTE_TEXT}.",
                frame_grid,
            ]
        )
    return "\n".join(lines).strip()
