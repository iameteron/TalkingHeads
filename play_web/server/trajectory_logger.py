import pickle
import random
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from faker import Faker

from .active_agent_helpers import (
    format_agent_observation_text_from_state,
    format_inventory_from_state,
)


TRAJECTORY_ROOT = Path(__file__).resolve().parent.parent.parent / "trajectories_logs"
TRAJECTORY_ROOT.mkdir(parents=True, exist_ok=True)


_FAKER = Faker("en_US")


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_trajectory_date(date_yyyymmdd: str) -> str:
    if len(date_yyyymmdd) == 8 and date_yyyymmdd.isdigit():
        return f"{date_yyyymmdd[0:4]}-{date_yyyymmdd[4:6]}-{date_yyyymmdd[6:8]}"
    return date_yyyymmdd


def generate_random_person_name() -> tuple[str, str]:
    return _FAKER.last_name(), _FAKER.first_name()


def trajectory_stem_parts(stem: str) -> tuple[str, List[str]]:
    prefix = "trajectory_"
    if not stem.startswith(prefix):
        return "", [stem] if stem else []
    parts = stem[len(prefix) :].split("_")
    if not parts:
        return "", []
    date_str = ""
    idx = 0
    if len(parts[0]) == 8 and parts[0].isdigit():
        date_str = parts[0]
        idx = 1
        if idx < len(parts) and len(parts[idx]) == 6 and parts[idx].isdigit():
            idx += 1
    return date_str, parts[idx:]


def display_name_from_filename(filename: str) -> str:
    stem = filename[:-4] if filename.lower().endswith(".pkl") else filename
    date_str, label_parts = trajectory_stem_parts(stem)
    label = " ".join(part for part in label_parts if part)
    if date_str and label_parts and len(label_parts) == 1 and label_parts[0].isdigit():
        return f"{format_trajectory_date(date_str)} ({label_parts[0]})"
    if date_str and label:
        return f"{label} {format_trajectory_date(date_str)}"
    if date_str:
        return format_trajectory_date(date_str)
    return label or stem or "unnamed"


def label_from_display_name(display_name: str, date_yyyymmdd: str) -> str:
    text = str(display_name or "").strip()
    if not text:
        return "unnamed"
    date_formatted = format_trajectory_date(date_yyyymmdd)
    if date_formatted and text.endswith(date_formatted):
        text = text[: -len(date_formatted)].strip()
    return text or "unnamed"


def sanitize_filename_label(label: str) -> str:
    parts = re.split(r"[\s_]+", str(label or "").strip())
    clean: List[str] = []
    for part in parts:
        token = "".join(ch for ch in part if ch.isalnum() or ch in ".-")
        if token:
            clean.append(token)
    return "_".join(clean) or "unnamed"


def build_trajectory_filename(*, when: Optional[datetime] = None) -> str:
    when = when or datetime.now()
    date_str = when.strftime("%Y%m%d")
    surname, first_name = generate_random_person_name()
    label = sanitize_filename_label(f"{surname}_{first_name}")
    return f"trajectory_{date_str}_{label}.pkl"


def unique_trajectory_path(filename: str) -> Path:
    path = TRAJECTORY_ROOT / filename
    if not path.exists():
        return path
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    for idx in range(2, 1000):
        candidate = TRAJECTORY_ROOT / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not allocate unique trajectory filename for: {filename}")


def _serialize_env_state(env_state: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    try:
        pos = getattr(env_state, "player_position", None)
        if pos is not None:
            result["player_position"] = list(map(int, np.array(pos).tolist()))
    except Exception:
        pass
    try:
        direction = getattr(env_state, "player_direction", None)
        if direction is not None:
            result["player_direction"] = int(direction)
    except Exception:
        pass
    try:
        inv = getattr(env_state, "inventory", None)
        if inv is not None:
            inv_dict: Dict[str, int] = {}
            for name in dir(inv):
                if name.startswith("_"):
                    continue
                val = getattr(inv, name)
                if isinstance(val, (int, float)) and val:
                    inv_dict[name] = int(val)
            result["inventory"] = inv_dict
            result["inventory_text"] = format_inventory_from_state(env_state)
    except Exception:
        pass
    try:
        m = getattr(env_state, "map", None)
        if m is not None:
            result["map"] = np.array(m, dtype=int).tolist()
    except Exception:
        pass
    try:
        result["observation_text"] = format_agent_observation_text_from_state(env_state)
    except Exception:
        pass
    return result


@dataclass
class TrajectoryStep:
    timestamp: str
    agent_prompt: str
    env_state: Dict[str, Any]
    oracle_dialog: List[Dict[str, str]]
    raw_answer: str
    parsed: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)


class TrajectoryLogger:
    def __init__(self, *, persist_tmp: bool = True) -> None:
        self._steps: List[TrajectoryStep] = []
        self._episode_id = f"episode_{_now_ts()}_{random.randint(0, 9999):04d}"
        self._tmp_path = TRAJECTORY_ROOT / f"{self._episode_id}.tmp.pkl"
        self._final_path: Optional[Path] = None
        self.persist_tmp = bool(persist_tmp)

    @property
    def steps(self) -> List[TrajectoryStep]:
        return self._steps

    @property
    def tmp_path(self) -> Path:
        return self._tmp_path

    @property
    def final_path(self) -> Optional[Path]:
        return self._final_path

    def _write_tmp(self) -> None:
        with self._tmp_path.open("wb") as f:
            pickle.dump([asdict(s) for s in self._steps], f)

    def add_step(
        self,
        *,
        agent_prompt: str,
        env_state: Any,
        oracle_dialog: List[Dict[str, str]],
        raw_answer: str,
        parsed: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        step = TrajectoryStep(
            timestamp=_now_ts(),
            agent_prompt=str(agent_prompt),
            env_state=_serialize_env_state(env_state),
            oracle_dialog=[dict(x) for x in oracle_dialog],
            raw_answer=str(raw_answer),
            parsed=dict(parsed),
            meta=dict(meta or {}),
        )
        self._steps.append(step)
        if self.persist_tmp:
            self._write_tmp()

    def save(self) -> Path:
        if not self._tmp_path.exists():
            self._write_tmp()
        if not self._tmp_path.exists():
            raise FileNotFoundError(f"Temporary trajectory file not found: {self._tmp_path}")
        final_path = unique_trajectory_path(build_trajectory_filename())
        self._tmp_path.rename(final_path)
        self._final_path = final_path
        return final_path

    def delete(self) -> None:
        if self._tmp_path.exists():
            self._tmp_path.unlink()


def load_trajectory(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    return data if isinstance(data, list) else [data]
