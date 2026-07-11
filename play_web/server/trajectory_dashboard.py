from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from craftax.craftax_classic.constants import Action
from oracle.knowledge.store import _iter_field_records, extract_to_database_blocks

from .active_agent_helpers import _sanitize_action_content, format_action_for_ui
from .campaign_mode import CRAFTAX_CAMPAIGN_TASKS, EXO_CAMPAIGN_TASKS
from .trajectory_logger import (
    TRAJECTORY_ROOT,
    display_name_from_filename,
    label_from_display_name,
    load_trajectory,
    sanitize_filename_label,
    trajectory_stem_parts,
    unique_trajectory_path,
)

_EXO_PROMPT_MARKER = "MC-3"

_CAMPAIGN_GOAL_ORDER: dict[str, int] = {}
for index, (craftax_task, exo_task) in enumerate(zip(CRAFTAX_CAMPAIGN_TASKS, EXO_CAMPAIGN_TASKS)):
    _CAMPAIGN_GOAL_ORDER[craftax_task.goal] = index
    _CAMPAIGN_GOAL_ORDER[exo_task.goal] = index
_CAMPAIGN_GOAL_LABELS: list[str] = []
_seen_campaign_goal_labels: set[str] = set()
for craftax_task, exo_task in zip(CRAFTAX_CAMPAIGN_TASKS, EXO_CAMPAIGN_TASKS):
    for goal in (craftax_task.goal, exo_task.goal):
        if goal not in _seen_campaign_goal_labels:
            _CAMPAIGN_GOAL_LABELS.append(goal)
            _seen_campaign_goal_labels.add(goal)
_EMPTY_GOAL_BUCKET: Dict[str, int] = {
    "questions": 0,
    "actions": 0,
    "steps": 0,
    "database_records": 0,
    "database_updates": 0,
}


@dataclass
class TrajectoryMeta:
    id: str
    filename: str
    display_name: str
    path: Path
    created_at: float
    size_bytes: int
    steps: int
    actions: int
    questions: int
    avg_answer_len_chars: float
    server: str
    interaction_mode: str = "oracle"
    model: str = ""
    model_mode: str = ""


_VIS_ROOT = Path(__file__).resolve().parent.parent / "external_visualization"
_ASSETS_DIR = _VIS_ROOT / "assets"
_EXO_MOD_DIR = _VIS_ROOT / "exo-planet_mod"
_EXO_MAPPING_FILE = _EXO_MOD_DIR / "texture_mapping.txt"
_OBS_SHAPE = (7, 9)
_TILE_SIZE = 32
_BLOCK_ASSET_NAMES = {
    0: "debug_tile.png",
    1: "debug_tile.png",
    2: "grass.png",
    3: "water.png",
    4: "stone.png",
    5: "tree.png",
    6: "wood.png",
    7: "path.png",
    8: "coal.png",
    9: "iron.png",
    10: "diamond.png",
    11: "table.png",
    12: "furnace.png",
    13: "sand.png",
    14: "lava.png",
    15: "plant_on_grass.png",
    16: "ripe_plant_on_grass.png",
}
_PLAYER_ASSET_BY_DIR = {
    1: "player-left.png",
    2: "player-right.png",
    3: "player-up.png",
    4: "player-down.png",
}
_BLOCK_TEXTURE_CACHE: Dict[tuple[str, int], Image.Image] = {}
_PLAYER_TEXTURE_CACHE: Dict[tuple[str, int], Image.Image] = {}


def _resolve_wildcard_path(raw_path: str) -> str | None:
    if "*" not in raw_path:
        return raw_path if Path(raw_path).exists() else None
    matches = sorted(glob(raw_path))
    return matches[0] if matches else None


def _parse_exo_texture_mapping() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not _EXO_MAPPING_FILE.exists():
        return mapping
    with _EXO_MAPPING_FILE.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or ">" not in line:
                continue
            key_raw, value_raw = line.split(">", 1)
            key = key_raw.strip()
            value = value_raw.strip()
            if key and value:
                mapping[key] = value
    return mapping


def _resolve_texture_path_for_theme(filename: str, texture_theme: str) -> Path:
    theme = (
        "exo-planet"
        if str(texture_theme).strip().lower() in {"exo", "exo-planet"}
        else "craftax"
    )
    if theme != "exo-planet":
        return _ASSETS_DIR / filename

    mapped = _parse_exo_texture_mapping().get(filename, "").strip()
    if mapped:
        if "+" in mapped:
            first = mapped.split("+", 1)[0].strip()
            first_resolved = _resolve_wildcard_path(str(_VIS_ROOT / first))
            if first_resolved:
                return Path(first_resolved)
        else:
            resolved = _resolve_wildcard_path(str(_VIS_ROOT / mapped))
            if resolved:
                return Path(resolved)

    exo_direct = _EXO_MOD_DIR / filename
    if exo_direct.exists():
        return exo_direct
    return _ASSETS_DIR / filename


def _is_exo_trajectory(data: List[Dict[str, Any]]) -> bool:
    """Detect exo-planet runs from agent prompts (Survey Unit MC-3)."""
    for step in data:
        prompt = str(step.get("agent_prompt") or "")
        if _EXO_PROMPT_MARKER in prompt:
            return True
    return False


def _texture_theme_for_trajectory(data: List[Dict[str, Any]]) -> str:
    return "exo-planet" if _is_exo_trajectory(data) else "craftax"


def _craftax_valid_actions() -> set[str]:
    return {name.upper() for name in Action.__members__.keys()}


def _exo_valid_actions() -> set[str]:
    try:
        import sys
        from pathlib import Path as _Path

        exo_root = _Path(__file__).resolve().parents[2] / "MegaPrompt" / "exo-planet_prompt"
        if str(exo_root) not in sys.path:
            sys.path.insert(0, str(exo_root))
        from actions import EXO_ENV_ACTIONS  # type: ignore

        return {str(name).upper() for name in EXO_ENV_ACTIONS}
    except Exception:
        return _craftax_valid_actions()


def _load_block_texture(block_id: int, *, texture_theme: str = "craftax") -> Image.Image:
    cache_key = (texture_theme, block_id)
    cached = _BLOCK_TEXTURE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    asset_name = _BLOCK_ASSET_NAMES.get(block_id, "debug_tile.png")
    path = _resolve_texture_path_for_theme(asset_name, texture_theme)
    texture = Image.open(path).convert("RGBA").resize(
        (_TILE_SIZE, _TILE_SIZE), Image.Resampling.NEAREST
    )
    _BLOCK_TEXTURE_CACHE[cache_key] = texture
    return texture


def _load_player_texture(player_direction: int, *, texture_theme: str = "craftax") -> Image.Image:
    cache_key = (texture_theme, player_direction)
    cached = _PLAYER_TEXTURE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    asset_name = _PLAYER_ASSET_BY_DIR.get(player_direction, "player-down.png")
    path = _resolve_texture_path_for_theme(asset_name, texture_theme)
    texture = Image.open(path).convert("RGBA").resize(
        (_TILE_SIZE, _TILE_SIZE), Image.Resampling.NEAREST
    )
    _PLAYER_TEXTURE_CACHE[cache_key] = texture
    return texture


def _render_observation_image(
    env_state: Dict[str, Any], *, texture_theme: str = "craftax"
) -> str:
    map_grid = env_state.get("map")
    player_pos = env_state.get("player_position")
    player_direction = int(env_state.get("player_direction") or 4)
    if not isinstance(map_grid, list) or not map_grid or not isinstance(player_pos, list):
        return ""
    try:
        grid = np.array(map_grid, dtype=np.int32)
        px, py = int(player_pos[0]), int(player_pos[1])
    except Exception:
        return ""
    if grid.ndim != 2 or px < 0 or py < 0:
        return ""

    obs_h, obs_w = _OBS_SHAPE
    half_h = obs_h // 2
    half_w = obs_w // 2
    canvas = Image.new("RGBA", (obs_w * _TILE_SIZE, obs_h * _TILE_SIZE), (20, 26, 40, 255))

    for local_x in range(obs_h):
        for local_y in range(obs_w):
            gx = px - half_h + local_x
            gy = py - half_w + local_y
            block_id = 0
            if 0 <= gx < grid.shape[0] and 0 <= gy < grid.shape[1]:
                block_id = int(grid[gx, gy])
            block_tex = _load_block_texture(block_id, texture_theme=texture_theme)
            canvas.alpha_composite(block_tex, dest=(local_y * _TILE_SIZE, local_x * _TILE_SIZE))

    player_tex = _load_player_texture(player_direction, texture_theme=texture_theme)
    canvas.alpha_composite(player_tex, dest=(half_w * _TILE_SIZE, half_h * _TILE_SIZE))

    out = io.BytesIO()
    canvas.convert("RGB").save(out, format="PNG")
    encoded = base64.b64encode(out.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _inspect_trajectory(path: Path) -> TrajectoryMeta:
    data = load_trajectory(path)
    actions = 0
    questions = 0
    answer_lens: List[int] = []
    server = "oracle"
    interaction_mode = "oracle"
    model = ""
    model_mode = ""
    for step in data:
        parsed: Dict[str, Any] = step.get("parsed", {}) or {}
        actions += int("action" in parsed)
        raw_answer = step.get("raw_answer", "")
        if isinstance(raw_answer, str) and raw_answer:
            answer_lens.append(len(raw_answer))
        meta = step.get("meta") or {}
        if isinstance(meta, dict):
            step_mode = str(meta.get("interaction_mode") or "").strip().lower()
            if step_mode == "human":
                interaction_mode = "human"
                server = "human"
            elif meta.get("server") == "human":
                interaction_mode = "human"
                server = "human"
            if not model and meta.get("active_agent_model"):
                model = str(meta.get("active_agent_model"))
            if not model_mode and meta.get("active_agent_mode"):
                model_mode = str(meta.get("active_agent_mode"))
    questions = _count_operator_questions(data)
    stat = path.stat()
    return TrajectoryMeta(
        id=path.name,
        filename=path.name,
        display_name=display_name_from_filename(path.name),
        path=path,
        created_at=stat.st_mtime,
        size_bytes=stat.st_size,
        steps=len(data),
        actions=actions,
        questions=questions,
        avg_answer_len_chars=float(sum(answer_lens) / len(answer_lens)) if answer_lens else 0.0,
        server=server,
        interaction_mode=interaction_mode,
        model=model,
        model_mode=model_mode,
    )


def list_trajectories() -> Dict[str, Any]:
    items: List[TrajectoryMeta] = []
    if not TRAJECTORY_ROOT.exists():
        return {"items": []}
    for path in sorted(TRAJECTORY_ROOT.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            items.append(_inspect_trajectory(path))
        except Exception:
            continue
    return {"items": [t.__dict__ for t in items]}


def delete_trajectories(ids: List[str]) -> Dict[str, Any]:
    deleted: List[str] = []
    if not ids or not TRAJECTORY_ROOT.exists():
        return {"deleted": deleted}
    for tid in ids:
        path = TRAJECTORY_ROOT / str(tid)
        try:
            if path.exists() and path.is_file():
                path.unlink()
                deleted.append(str(tid))
        except Exception:
            continue
    return {"deleted": deleted}


def rename_trajectory(old_id: str, new_display_name: str) -> Dict[str, Any]:
    if not TRAJECTORY_ROOT.exists():
        raise FileNotFoundError("No trajectories directory")
    old_path = TRAJECTORY_ROOT / old_id
    if not old_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {old_id}")
    date_str, _ = trajectory_stem_parts(old_path.stem)
    if not date_str:
        date_str = datetime.now().strftime("%Y%m%d")
    label = label_from_display_name(new_display_name, date_str)
    new_path = unique_trajectory_path(
        f"trajectory_{date_str}_{sanitize_filename_label(label)}.pkl"
    )
    if new_path.resolve() != old_path.resolve() and new_path.exists():
        raise FileExistsError(f"Target trajectory name already exists: {new_path.name}")
    old_path.rename(new_path)
    return _inspect_trajectory(new_path).__dict__


def _goal_label_from_step(step: Dict[str, Any]) -> str:
    meta = step.get("meta") or {}
    if isinstance(meta, dict):
        goal = str(meta.get("goal") or "").strip()
        if goal:
            return goal
    return "Unknown goal"


def _step_meta(step: Dict[str, Any]) -> Dict[str, Any]:
    meta = step.get("meta") or {}
    return meta if isinstance(meta, dict) else {}


def _step_superseded_by_operator_retry(data: List[Dict[str, Any]], idx: int) -> bool:
    """True when this step was replaced by an operator-feedback retry on the same tick."""
    if idx + 1 >= len(data):
        return False
    cur_meta = _step_meta(data[idx])
    next_meta = _step_meta(data[idx + 1])
    if not next_meta.get("operator_retry_kind"):
        return False
    return next_meta.get("step_index") == cur_meta.get("step_index")


def _non_system_dialog_turns(dialog: Any) -> List[tuple[str, str]]:
    turns: List[tuple[str, str]] = []
    if not isinstance(dialog, list):
        return turns
    for turn in dialog:
        if not isinstance(turn, dict):
            continue
        question = str(turn.get("question") or "").strip()
        if not question or question == "System":
            continue
        answer = str(turn.get("answer") or "").strip()
        turns.append((question, answer))
    return turns


def _goal_for_operator_question_at_or_before(
    data: List[Dict[str, Any]], max_idx: int, question: str
) -> str:
    for idx in range(max_idx, -1, -1):
        if _step_superseded_by_operator_retry(data, idx):
            continue
        parsed = data[idx].get("parsed", {}) or {}
        if str(parsed.get("question") or "").strip() == question:
            return _goal_label_from_step(data[idx])
    return _goal_label_from_step(data[max_idx])


def _operator_question_counts_by_goal(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count questions actually delivered to the operator, keyed by campaign goal."""
    counts: Dict[str, int] = {}
    prev_turns: List[tuple[str, str]] = []
    for idx, step in enumerate(data):
        turns = _non_system_dialog_turns(step.get("oracle_dialog"))
        new_turns = turns[len(prev_turns) :]
        prev_turns = turns
        for question, _answer in new_turns:
            goal = _goal_for_operator_question_at_or_before(data, idx, question)
            counts[goal] = counts.get(goal, 0) + 1

    if data:
        last_idx = len(data) - 1
        last_step = data[last_idx]
        if not _step_superseded_by_operator_retry(data, last_idx):
            parsed = last_step.get("parsed", {}) or {}
            pending_question = str(parsed.get("question") or "").strip()
            if pending_question and not any(
                q == pending_question for q, _ in prev_turns
            ):
                goal = _goal_label_from_step(last_step)
                counts[goal] = counts.get(goal, 0) + 1
    return counts


def _count_operator_questions(data: List[Dict[str, Any]]) -> int:
    return sum(_operator_question_counts_by_goal(data).values())


def _count_upsert_records_in_block(block: str) -> int:
    count = 0
    for fields in _iter_field_records(block):
        if str(fields.get("OP", "UPSERT")).upper() == "DELETE":
            continue
        if str(fields.get("TYPE", "")).strip():
            count += 1
    return count


def _database_records_in_step(step: Dict[str, Any], parsed: Dict[str, Any]) -> int:
    blocks: List[str] = []
    to_db = parsed.get("to_database")
    if isinstance(to_db, str) and to_db.strip():
        blocks.append(to_db.strip())
    else:
        raw_answer = step.get("raw_answer", "")
        if isinstance(raw_answer, str) and raw_answer.strip():
            blocks.extend(extract_to_database_blocks(raw_answer))
    return sum(_count_upsert_records_in_block(block) for block in blocks)


def _ordered_campaign_goals(seen: set[str]) -> List[str]:
    ordered = [g for g in _CAMPAIGN_GOAL_LABELS if g in seen]
    extras = sorted(g for g in seen if g not in _CAMPAIGN_GOAL_ORDER and g != "Unknown goal")
    ordered.extend(extras)
    if "Unknown goal" in seen:
        ordered.append("Unknown goal")
    return ordered


def _merge_goal_buckets(
    target: Dict[str, Dict[str, int]], source: Dict[str, Dict[str, int]]
) -> None:
    for goal, counts in source.items():
        bucket = target.setdefault(goal, dict(_EMPTY_GOAL_BUCKET))
        for key, value in counts.items():
            bucket[key] = int(bucket.get(key, 0)) + int(value)


def _record_action_tokens(
    parsed: Dict[str, Any],
    *,
    exo_mode: bool,
    valid_actions: set[str],
    action_counts: Dict[str, int],
) -> None:
    if exo_mode:
        label = format_action_for_ui(parsed, world_mode="exo-planet").strip().upper()
        tokens = [t for t in label.split() if t] if label else []
    else:
        raw_action = str(parsed.get("action") or "").strip()
        if not raw_action:
            return
        cleaned = _sanitize_action_content(raw_action)
        tokens = [t for t in cleaned.upper().split() if t]
    for token in tokens:
        if token in valid_actions:
            action_counts[token] = action_counts.get(token, 0) + 1
        else:
            action_counts["WRONG"] = action_counts.get("WRONG", 0) + 1


def _format_step_action(parsed: Dict[str, Any], *, exo_mode: bool) -> str:
    if exo_mode:
        return format_action_for_ui(parsed, world_mode="exo-planet")
    return str(parsed.get("action") or "")


def _process_trajectory_steps(
    data: List[Dict[str, Any]],
    *,
    per_goal: Dict[str, Dict[str, int]],
    action_counts: Dict[str, int],
    valid_actions: set[str],
    exo_mode: bool = False,
) -> tuple[int, int, int, int, int, int, float]:
    """Returns actions, questions, db_records, db_updates, answer_chars, answer_count, mean_answer_len."""
    episode_actions = 0
    episode_questions = 0
    episode_db_records = 0
    episode_db_updates = 0
    episode_answer_chars = 0
    episode_answer_count = 0
    question_counts_by_goal = _operator_question_counts_by_goal(data)
    for goal_label, question_count in question_counts_by_goal.items():
        bucket = per_goal.setdefault(goal_label, dict(_EMPTY_GOAL_BUCKET))
        bucket["questions"] += question_count
    episode_questions = sum(question_counts_by_goal.values())

    for idx, step in enumerate(data):
        parsed: Dict[str, Any] = step.get("parsed", {}) or {}
        goal_label = _goal_label_from_step(step)
        bucket = per_goal.setdefault(goal_label, dict(_EMPTY_GOAL_BUCKET))
        bucket["steps"] += 1
        if parsed.get("knowledge_updated") and not _step_superseded_by_operator_retry(
            data, idx
        ):
            record_count = _database_records_in_step(step, parsed)
            if record_count < 1:
                record_count = 1
            episode_db_updates += 1
            episode_db_records += record_count
            bucket["database_updates"] += 1
            bucket["database_records"] += record_count
        if "action" in parsed:
            episode_actions += 1
            bucket["actions"] += 1
            _record_action_tokens(
                parsed,
                exo_mode=exo_mode,
                valid_actions=valid_actions,
                action_counts=action_counts,
            )
        raw_answer = step.get("raw_answer", "")
        if isinstance(raw_answer, str) and raw_answer:
            episode_answer_chars += len(raw_answer)
            episode_answer_count += 1
    mean_answer_len = (
        float(episode_answer_chars) / episode_answer_count if episode_answer_count else 0.0
    )
    return (
        episode_actions,
        episode_questions,
        episode_db_records,
        episode_db_updates,
        episode_answer_chars,
        episode_answer_count,
        mean_answer_len,
    )


def _per_goal_stats_rows(per_goal: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for goal, counts in per_goal.items():
        rows.append(
            {
                "goal": goal,
                "questions": int(counts.get("questions", 0)),
                "actions": int(counts.get("actions", 0)),
                "steps": int(counts.get("steps", 0)),
                "database_records": int(counts.get("database_records", 0)),
                "database_updates": int(counts.get("database_updates", 0)),
            }
        )
    rows.sort(
        key=lambda r: (
            _CAMPAIGN_GOAL_ORDER.get(str(r["goal"]), 10_000),
            str(r["goal"]).lower(),
        )
    )
    return rows


def stats_for_trajectories(ids: List[str]) -> Dict[str, Any]:
    if not ids:
        return {
            "total_episodes": 0,
            "total_steps": 0,
            "total_actions": 0,
            "total_questions": 0,
            "question_pct": 0.0,
            "action_pct": 0.0,
            "mean_actions_per_episode": 0.0,
            "mean_answer_len_chars": 0.0,
            "questions_by_goal": [],
            "total_database_records": 0,
            "total_database_updates": 0,
            "runs": [],
            "campaign_goals": [],
        }
    total_steps = total_actions = total_questions = 0
    total_database_records = total_database_updates = 0
    total_answer_chars = total_answer_count = 0
    episodes = 0
    episodes_metrics: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    action_counts: Dict[str, int] = {}
    per_goal: Dict[str, Dict[str, int]] = {}
    goals_seen: set[str] = set()
    exo_trajectory_count = 0
    for tid in ids:
        path = TRAJECTORY_ROOT / tid
        if not path.exists():
            continue
        try:
            data = load_trajectory(path)
        except Exception:
            continue
        exo_mode = _is_exo_trajectory(data)
        if exo_mode:
            exo_trajectory_count += 1
        valid_actions = _exo_valid_actions() if exo_mode else _craftax_valid_actions()
        episodes += 1
        episode_steps = len(data)
        total_steps += episode_steps
        per_goal_run: Dict[str, Dict[str, int]] = {}
        (
            episode_actions,
            episode_questions,
            episode_db_records,
            episode_db_updates,
            episode_answer_chars,
            episode_answer_count,
            mean_answer_len,
        ) = _process_trajectory_steps(
            data,
            per_goal=per_goal_run,
            action_counts=action_counts,
            valid_actions=valid_actions,
            exo_mode=exo_mode,
        )
        _merge_goal_buckets(per_goal, per_goal_run)
        goals_seen.update(per_goal_run.keys())
        total_actions += episode_actions
        total_questions += episode_questions
        total_database_records += episode_db_records
        total_database_updates += episode_db_updates
        total_answer_chars += episode_answer_chars
        total_answer_count += episode_answer_count
        display_name = display_name_from_filename(path.name)
        episodes_metrics.append(
            {
                "id": tid,
                "display_name": display_name,
                "steps": episode_steps,
                "actions": episode_actions,
                "questions": episode_questions,
                "database_records": episode_db_records,
                "mean_answer_len_chars": mean_answer_len,
            }
        )
        runs.append(
            {
                "id": tid,
                "display_name": display_name,
                "by_goal": _per_goal_stats_rows(per_goal_run),
                "total_questions": episode_questions,
                "total_database_records": episode_db_records,
            }
        )
    if episodes == 0:
        return {
            "total_episodes": 0,
            "total_steps": 0,
            "total_actions": 0,
            "total_questions": 0,
            "question_pct": 0.0,
            "action_pct": 0.0,
            "mean_actions_per_episode": 0.0,
            "mean_answer_len_chars": 0.0,
            "episodes": [],
            "action_counts": {},
            "questions_by_goal": [],
            "total_database_records": 0,
            "total_database_updates": 0,
            "runs": [],
            "campaign_goals": [],
        }
    denom_steps = total_actions + total_questions or 1
    campaign_goals = _ordered_campaign_goals(goals_seen)
    return {
        "total_episodes": episodes,
        "total_steps": total_steps,
        "total_actions": total_actions,
        "total_questions": total_questions,
        "question_pct": 100.0 * total_questions / denom_steps,
        "action_pct": 100.0 * total_actions / denom_steps,
        "mean_actions_per_episode": float(total_actions) / episodes if episodes else 0.0,
        "mean_answer_len_chars": float(total_answer_chars) / total_answer_count if total_answer_count else 0.0,
        "episodes": episodes_metrics,
        "action_counts": action_counts,
        "questions_by_goal": _per_goal_stats_rows(per_goal),
        "total_database_records": total_database_records,
        "total_database_updates": total_database_updates,
        "runs": runs,
        "campaign_goals": campaign_goals,
        "exo_trajectory_count": exo_trajectory_count,
        "has_exo_trajectories": exo_trajectory_count > 0,
    }


def _operator_answer_for_question_step(
    data: List[Dict[str, Any]], idx: int, question: str
) -> str:
    """Return the operator answer for a question asked at step idx (non-system turns only)."""
    question = str(question or "").strip()
    if not question:
        return ""

    def _matching_answer(oracle_dialog: Any) -> str:
        if not isinstance(oracle_dialog, list):
            return ""
        for turn in reversed(oracle_dialog):
            if not isinstance(turn, dict):
                continue
            q = str(turn.get("question") or "").strip()
            if q == "System":
                continue
            a = str(turn.get("answer") or "").strip()
            if q == question and a:
                return a
        return ""

    step = data[idx]
    answer = _matching_answer(step.get("oracle_dialog"))
    if answer:
        return answer
    if idx + 1 < len(data):
        return _matching_answer(data[idx + 1].get("oracle_dialog"))
    return ""


def short_history_for_trajectory(trajectory_id: str, limit: int = 20) -> Dict[str, Any]:
    path = TRAJECTORY_ROOT / trajectory_id
    if not path.exists():
        raise FileNotFoundError(f"Trajectory not found: {trajectory_id}")
    data = load_trajectory(path)
    if not data:
        return {"id": trajectory_id, "steps": [], "world_mode": "craftax"}
    exo_mode = _is_exo_trajectory(data)
    world_mode = "exo-planet" if exo_mode else "craftax"
    steps: List[Dict[str, Any]] = []
    for idx, step in enumerate(data):
        env_state = step.get("env_state") or {}
        parsed = step.get("parsed") or {}
        observation_text = env_state.get("observation_text") or ""
        reasoning = parsed.get("reasoning") or ""
        action = _format_step_action(parsed, exo_mode=exo_mode)
        question = parsed.get("question") or ""
        operator_answer = _operator_answer_for_question_step(data, idx, question)
        if not (observation_text or reasoning or action or question or operator_answer):
            fallback = step.get("raw_answer") or step.get("agent_prompt") or ""
            if fallback:
                observation_text = str(fallback)
        steps.append(
            {
                "observation": str(observation_text),
                "reasoning": str(reasoning),
                "action": str(action),
                "question": str(question),
                "operator_answer": operator_answer,
            }
        )
    if limit and limit > 0:
        steps = steps[-limit:]
    return {"id": trajectory_id, "steps": steps, "world_mode": world_mode}


def play_history_for_trajectory(trajectory_id: str) -> Dict[str, Any]:
    path = TRAJECTORY_ROOT / trajectory_id
    if not path.exists():
        raise FileNotFoundError(f"Trajectory not found: {trajectory_id}")
    data = load_trajectory(path)
    if not data:
        return {"id": trajectory_id, "steps": [], "world_mode": "craftax"}

    exo_mode = _is_exo_trajectory(data)
    world_mode = "exo-planet" if exo_mode else "craftax"
    texture_theme = _texture_theme_for_trajectory(data)

    steps: List[Dict[str, Any]] = []
    for idx, step in enumerate(data):
        env_state = step.get("env_state") or {}
        parsed = step.get("parsed") or {}
        observation_text = str(env_state.get("observation_text") or "")
        reasoning = str(parsed.get("reasoning") or "")
        action = _format_step_action(parsed, exo_mode=exo_mode)
        question = str(parsed.get("question") or "")
        operator_answer = _operator_answer_for_question_step(data, idx, question)
        steps.append(
            {
                "observation": observation_text,
                "observation_image": _render_observation_image(
                    env_state, texture_theme=texture_theme
                ),
                "reasoning": reasoning,
                "action": action,
                "question": question,
                "operator_answer": operator_answer,
            }
        )

    return {"id": trajectory_id, "steps": steps, "world_mode": world_mode}

