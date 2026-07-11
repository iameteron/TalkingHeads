from __future__ import annotations

from pathlib import Path

from . import load_knowledge, use_knowledge_paths
from .store import _RUNTIME_KNOWLEDGE_JSON_PATH

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CRAFTEXT_ROOT = _REPO_ROOT / "MegaPrompt" / "craftext_prompt"
_EXO_ROOT = _REPO_ROOT / "MegaPrompt" / "exo-planet_prompt"


def _is_exo_world_mode(world_mode: str | None) -> bool:
    token = str(world_mode or "").strip().lower()
    return token in {"exo", "exo-planet", "exo_planet"}


def base_knowledge_paths_for_world(world_mode: str | None) -> tuple[Path, Path]:
    if _is_exo_world_mode(world_mode):
        return (
            _EXO_ROOT / "knowledge_data.json",
            _EXO_ROOT / "knowledge_data.txt",
        )
    return (
        _CRAFTEXT_ROOT / "knowledge_data.json",
        _CRAFTEXT_ROOT / "knowledge_data.txt",
    )


def load_knowledge_with_world_fallback(world_mode: str | None) -> str:
    """Use active runtime knowledge paths when set, otherwise world base files."""
    if _RUNTIME_KNOWLEDGE_JSON_PATH.get() is not None:
        return load_knowledge()
    base_json, base_txt = base_knowledge_paths_for_world(world_mode)
    with use_knowledge_paths(json_path=base_json, txt_path=base_txt):
        return load_knowledge()
