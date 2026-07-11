from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

DEFAULT_TRAJECTORY_PATH = (
    Path(__file__).resolve().parent / "extra_files" / "place_a_table_trajectory.pkl"
)


def load_trajectory(*, path: Path | str | None = None) -> list[dict[str, Any]]:
    trajectory_path = Path(path) if path is not None else DEFAULT_TRAJECTORY_PATH
    with trajectory_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected trajectory pickle to contain a list, got {type(data)!r}.")
    if not data:
        raise ValueError(f"Trajectory pickle is empty: {trajectory_path}")
    return data


def load_trajectory_entry(*, step_idx: int = -1, path: Path | str | None = None) -> dict[str, Any]:
    data = load_trajectory(path=path)
    try:
        return data[step_idx]
    except IndexError as exc:
        raise IndexError(
            f"Trajectory step {step_idx!r} is out of range for {len(data)} steps."
        ) from exc


def extract_state_from_entry(entry: dict[str, Any]) -> Any:
    env_state = entry.get("env_state")
    if not isinstance(env_state, dict):
        raise ValueError("Trajectory entry['env_state'] must be a dict.")

    craftax_state = env_state.get("craftax_state")
    if not isinstance(craftax_state, dict):
        raise ValueError("Trajectory entry['env_state']['craftax_state'] must be a dict.")

    required = ("map", "player_position", "player_direction", "inventory")
    missing = [name for name in required if name not in craftax_state]
    if missing:
        raise ValueError(f"craftax_state is missing required fields: {missing}")

    return SimpleNamespace(**craftax_state)
