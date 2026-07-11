from __future__ import annotations

from pathlib import Path
from typing import Any

_WORLD_PATH = (
    Path(__file__).resolve().parents[2]
    / "exo-planet_prompt"
    / "world"
    / "exo_planet_world_prompt.md"
)


def render(_value: Any) -> str:
    if not _WORLD_PATH.is_file():
        raise FileNotFoundError(f"exo-planet world description not found: {_WORLD_PATH}")
    return _WORLD_PATH.read_text(encoding="utf-8").strip()
