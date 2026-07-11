from __future__ import annotations

from typing import Any, Mapping

from .arc_grid import render as render_grid


def _get(obs: Any, key: str, default: Any = "") -> Any:
    if isinstance(obs, Mapping):
        return obs.get(key, default)
    return getattr(obs, key, default)


def render(obs: Any) -> str:
    grid_text = render_grid(obs)
    width = _get(obs, "w", 0)
    height = _get(obs, "h", 0)
    png_b64 = str(_get(obs, "png_b64", "")).strip()

    lines = [
        grid_text,
        "",
        f"Frame image: {width}x{height} PNG, attached below.",
        "Use the image for spatial visual layout and the grid for exact coordinates/colors.",
    ]
    if png_b64:
        lines.extend(["", f"[[image:data:image/png;base64,{png_b64}]]"])
    else:
        lines.append("Frame image unavailable.")
    return "\n".join(lines).strip()
