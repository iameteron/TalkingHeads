from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_CRAFTEXT_BENCH = _ROOT.parent / "craftext_prompt" / "prompt_bench"
if str(_CRAFTEXT_BENCH) not in sys.path:
    sys.path.insert(0, str(_CRAFTEXT_BENCH))

from trajectory_fixture import (  # noqa: E402
    DEFAULT_TRAJECTORY_PATH,
    extract_state_from_entry,
    load_trajectory,
    load_trajectory_entry,
)

DEFAULT_STEP_IDX = -1

EXO_SR_SUCCESS_ACTIONS = frozenset(
    {"PLACE_REPLICATOR", "EXTRACT", "PLACE_BASALT_BEACON", "MAKE_BONE_DRILL"}
)

_EXO_GOALS: dict[str, str] = {
    "Place a table": "Deploy Replicator",
    "Collect 2 wood": "Collect 2 Biomass",
}

# Bench dialog uses exo vocabulary; craftext pickle keeps legacy operator text.
_EXO_ORACLE_DIALOG: dict[int, list[dict[str, str]]] = {
    -1: [
        {
            "question": "[Tick 1/1] What do I need for PLACE_REPLICATOR?",
            "answer": (
                "Collect two Biomass first, face empty Regolith Turf or Survey Trail, "
                "then use PLACE_REPLICATOR."
            ),
        }
    ],
}


def _localize_trajectory_entry(entry: dict[str, Any], *, step_idx: int) -> dict[str, Any]:
    localized = copy.deepcopy(entry)
    meta = localized.get("meta")
    if isinstance(meta, dict):
        goal = str(meta.get("goal", "")).strip()
        if goal in _EXO_GOALS:
            meta["goal"] = _EXO_GOALS[goal]
    if step_idx in _EXO_ORACLE_DIALOG:
        localized["oracle_dialog"] = copy.deepcopy(_EXO_ORACLE_DIALOG[step_idx])
    return localized


def load_exo_trajectory_entry(step_idx: int = DEFAULT_STEP_IDX) -> dict[str, Any]:
    entry = load_trajectory_entry(step_idx=step_idx)
    return _localize_trajectory_entry(entry, step_idx=step_idx)


__all__ = [
    "DEFAULT_STEP_IDX",
    "DEFAULT_TRAJECTORY_PATH",
    "EXO_SR_SUCCESS_ACTIONS",
    "extract_state_from_entry",
    "load_exo_trajectory_entry",
    "load_trajectory",
    "load_trajectory_entry",
]
