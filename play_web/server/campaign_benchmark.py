from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from oracle.prompts.prompt_generation import is_exo_world_mode, normalize_world_mode

from .campaign_mode import CRAFTAX_CAMPAIGN_TASKS, EXO_CAMPAIGN_TASKS, TaskDefinition
from .leaderboard import (
    _load_jsonl,
    COMPANION_TEST_LOG_PATH,
    LEADERBOARD_LOG_PATH,
    load_leaderboard_entries,
)
from .model_names import canonical_model_key, short_model_name

_DEPLOYMENT_TASK_KEYS: tuple[str, ...] = (
    "make_wood_pickaxe",
    "collect_stone",
    "make_stone_pickaxe",
    "collect_iron",
    "make_iron_pickaxe",
    "collect_diamond",
)

_EXO_GOALS = {task.goal for task in EXO_CAMPAIGN_TASKS}
_CRAFTAX_GOALS = {task.goal for task in CRAFTAX_CAMPAIGN_TASKS}


def _short_model_key(raw: str) -> str:
    return canonical_model_key(raw)


def parse_since_timestamp(since: str | None) -> datetime | None:
    """Parse YYYY-MM-DD or ISO datetime; returns UTC-aware cutoff (inclusive)."""
    raw = str(since or "").strip()
    if not raw:
        return None
    try:
        if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
            dt = datetime.fromisoformat(raw)
        else:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _entry_finished_at(entry: dict[str, Any]) -> datetime | None:
    raw = str(entry.get("finished_at") or entry.get("started_at") or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def filter_entries_since(entries: list[dict[str, Any]], since: datetime | None) -> list[dict[str, Any]]:
    if since is None:
        return list(entries)
    kept: list[dict[str, Any]] = []
    for entry in entries:
        finished = _entry_finished_at(entry)
        if finished is None or finished >= since:
            kept.append(entry)
    return kept


def _tasks_for_world_mode(world_mode: str) -> list[TaskDefinition]:
    return EXO_CAMPAIGN_TASKS if is_exo_world_mode(world_mode) else CRAFTAX_CAMPAIGN_TASKS


def _task_keys_for_world_mode(world_mode: str) -> list[str]:
    return [task.key for task in _tasks_for_world_mode(world_mode)]


def _task_by_key(world_mode: str) -> dict[str, TaskDefinition]:
    return {task.key: task for task in _tasks_for_world_mode(world_mode)}


def _entry_world_mode(entry: dict[str, Any]) -> str:
    raw = entry.get("world_mode")
    if raw is not None and str(raw).strip():
        return normalize_world_mode(raw)
    prompt = str(entry.get("last_prompt_excerpt") or "")
    if "MC-3" in prompt or "exo-planet" in prompt.lower():
        return "exo-planet"
    goal = str(entry.get("last_goal") or "").strip()
    if goal in _EXO_GOALS:
        return "exo-planet"
    config_name = str(entry.get("megaprompt_config_name") or "").lower()
    if config_name.startswith("exo"):
        return "exo-planet"
    if goal in _CRAFTAX_GOALS:
        return "craftax"
    return "craftax"


def _test_entry_world_mode(entry: dict[str, Any]) -> str:
    raw = entry.get("world_mode")
    if raw is not None and str(raw).strip():
        return normalize_world_mode(raw)
    title = str(entry.get("task_title") or entry.get("task_key") or "").strip()
    if title in _EXO_GOALS:
        return "exo-planet"
    if title in _CRAFTAX_GOALS:
        return "craftax"
    return "craftax"


def _max_task_key_from_levels(phase1_levels: int, phase2_level: int, task_keys: list[str]) -> str:
    level = max(int(phase1_levels or 0), int(phase2_level or 0))
    if level <= 0 or not task_keys:
        return ""
    idx = min(level, len(task_keys)) - 1
    return task_keys[idx]


def _task_key_from_goal(goal: str, world_mode: str) -> str:
    text = str(goal or "").strip()
    if not text or text == "Campaign complete — all tasks finished.":
        return ""
    task_lists = [_tasks_for_world_mode(world_mode)]
    if is_exo_world_mode(world_mode):
        task_lists.append(CRAFTAX_CAMPAIGN_TASKS)
    for tasks in task_lists:
        for task in tasks:
            if text == task.goal or text == task.title:
                return task.key
        text_lower = text.lower()
        for task in tasks:
            if text_lower == task.goal.lower() or text_lower == task.title.lower():
                return task.key
    return ""


def _entry_has_meaningful_exploration_progress(entry: dict[str, Any]) -> bool:
    return (
        int(entry.get("phase1_completed_levels") or 0) > 0
        or int(entry.get("phase2_highest_level") or 0) > 0
        or int(entry.get("phase1_questions") or 0) > 0
        or int(entry.get("phase2_questions") or 0) > 0
        or bool(entry.get("phase1_completed_keys"))
        or bool(entry.get("phase2_completed_keys"))
    )


def _max_task_key_from_entry(entry: dict[str, Any], *, world_mode: str) -> str:
    """Best exploration task reached in one run (crafting + material collection)."""
    task_keys = _task_keys_for_world_mode(world_mode)
    tasks_by_key = _task_by_key(world_mode)
    if not task_keys:
        return ""

    candidate_keys: list[str] = []
    phase1_levels = int(entry.get("phase1_completed_levels") or 0)
    phase2_level = int(entry.get("phase2_highest_level") or 0)
    level_key = _max_task_key_from_levels(phase1_levels, phase2_level, task_keys)
    if level_key:
        candidate_keys.append(level_key)

    for raw_key in entry.get("phase1_completed_keys") or []:
        key = str(raw_key or "").strip()
        if key in tasks_by_key:
            candidate_keys.append(key)
    for raw_key in entry.get("phase2_completed_keys") or []:
        key = str(raw_key or "").strip()
        if key in tasks_by_key:
            candidate_keys.append(key)

    if _entry_has_meaningful_exploration_progress(entry):
        goal_key = _task_key_from_goal(str(entry.get("last_goal") or ""), world_mode)
        if goal_key:
            candidate_keys.append(goal_key)

    if not candidate_keys:
        return ""
    return max(candidate_keys, key=lambda key: _task_level(key, task_keys))


def _task_level(task_key: str, task_keys: list[str]) -> int:
    if not task_key:
        return 0
    try:
        return task_keys.index(task_key) + 1
    except ValueError:
        return 0


def _craft_icon_path(world_mode: str, task_key: str) -> str:
    theme = "exo-planet" if is_exo_world_mode(world_mode) else "craftax"
    if not task_key:
        return ""
    return f"./assets/benchmark/{theme}/{task_key}.png"


def _aggregate_exploration_runs(
    entries: list[dict[str, Any]], *, world_mode: str
) -> dict[str, dict[str, Any]]:
    task_keys = _task_keys_for_world_mode(world_mode)
    tasks_by_key = _task_by_key(world_mode)
    per_model: dict[str, dict[str, Any]] = {}

    for entry in entries:
        if _entry_world_mode(entry) != normalize_world_mode(world_mode):
            continue
        model = str(entry.get("active_agent_model") or "unknown").strip() or "unknown"
        key = _short_model_key(model)
        phase1_levels = int(entry.get("phase1_completed_levels") or 0)
        phase1_questions = int(entry.get("phase1_questions") or 0)
        phase2_level = int(entry.get("phase2_highest_level") or 0)
        max_key = _max_task_key_from_entry(entry, world_mode=world_mode)

        row = per_model.setdefault(
            key,
            {
                "model": model,
                "model_short": short_model_name(model),
                "runs": 0,
                "solved_sum": 0.0,
                "best_phase1_levels": 0,
                "best_phase2_level": 0,
                "best_max_task_key": "",
                "exploration_q_max": 0,
            },
        )
        if model:
            row["model"] = model
            row["model_short"] = short_model_name(model)
        row["runs"] += 1
        row["solved_sum"] += float(phase1_levels)

        best_p1 = int(row["best_phase1_levels"])
        best_p2 = int(row["best_phase2_level"])
        best_level = _task_level(str(row["best_max_task_key"] or ""), task_keys)
        new_level = _task_level(max_key, task_keys)

        if new_level > best_level:
            row["best_phase1_levels"] = phase1_levels
            row["best_phase2_level"] = phase2_level
            row["best_max_task_key"] = max_key
            row["exploration_q_max"] = phase1_questions
        elif new_level == best_level and new_level > 0:
            row["best_phase1_levels"] = max(best_p1, phase1_levels)
            row["best_phase2_level"] = max(best_p2, phase2_level)
            if max_key and _task_level(max_key, task_keys) >= _task_level(row["best_max_task_key"], task_keys):
                row["best_max_task_key"] = max_key or row["best_max_task_key"]
            prev_q = int(row["exploration_q_max"])
            if prev_q == 0 or (phase1_questions > 0 and phase1_questions < prev_q):
                row["exploration_q_max"] = phase1_questions

    return per_model


def _latest_deployment_by_model_task(
    test_entries: list[dict[str, Any]], *, world_mode: str
) -> dict[str, dict[str, Any]]:
    """Return model_key -> {model, tasks: {task_key: cell}}."""
    by_model: dict[str, dict[str, Any]] = {}
    for entry in test_entries:
        if _test_entry_world_mode(entry) != normalize_world_mode(world_mode):
            continue
        model = str(entry.get("model") or "unknown").strip() or "unknown"
        task_key = str(entry.get("task_key") or "").strip()
        if not task_key:
            continue
        model_key = _short_model_key(model)
        finished = str(entry.get("finished_at") or "")
        row = by_model.setdefault(model_key, {"model": model, "tasks": {}})
        row["model"] = model
        tasks = row["tasks"]
        prev = tasks.get(task_key)
        if prev is None or finished >= str(prev.get("finished_at") or ""):
            tasks[task_key] = {
                "sr": float(entry.get("sr") or 0),
                "mean_q": float(entry.get("mean_q") or entry.get("mean_questions") or 0),
                "runs": int(entry.get("runs") or 0),
                "successes": int(entry.get("successes") or 0),
                "finished_at": finished,
                "task_title": str(entry.get("task_title") or task_key),
                "operator_call_limit": int(entry.get("operator_call_limit") or 0),
                "violation_runs": int(entry.get("violation_runs") or 0),
                "limit_violation": bool(entry.get("limit_violation"))
                or int(entry.get("violation_runs") or 0) > 0,
            }
    return by_model


def _build_compact_rows(exploration_by_model: dict[str, dict[str, Any]], *, world_mode: str) -> list[dict[str, Any]]:
    total_tasks = len(_tasks_for_world_mode(world_mode))
    tasks_by_key = _task_by_key(world_mode)
    rows: list[dict[str, Any]] = []
    for data in exploration_by_model.values():
        runs = int(data.get("runs") or 0)
        solved_avg = float(data["solved_sum"]) / runs if runs else 0.0
        max_key = str(data.get("best_max_task_key") or "")
        task = tasks_by_key.get(max_key)
        rows.append(
            {
                "model": str(data.get("model") or "unknown"),
                "model_short": str(data.get("model_short") or "unknown"),
                "runs": runs,
                "solved_avg": round(solved_avg, 1),
                "solved_display": f"{round(solved_avg, 1)}/{total_tasks}",
                "max_task_key": max_key,
                "max_item_label": task.title if task else "—",
                "max_item_icon": _craft_icon_path(world_mode, max_key),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row.get("solved_avg") or 0),
            -_task_level(str(row.get("max_task_key") or ""), _task_keys_for_world_mode(world_mode)),
            str(row.get("model_short") or "").lower(),
        )
    )
    return rows


def _build_extended_rows(
    exploration_by_model: dict[str, dict[str, Any]],
    deployment_by_model: dict[str, dict[str, Any]],
    *,
    world_mode: str,
) -> list[dict[str, Any]]:
    tasks_by_key = _task_by_key(world_mode)
    task_keys = _task_keys_for_world_mode(world_mode)
    deployment_keys = [key for key in _DEPLOYMENT_TASK_KEYS if key in tasks_by_key]
    model_keys = sorted(set(exploration_by_model.keys()) | set(deployment_by_model.keys()))
    rows: list[dict[str, Any]] = []

    for model_key in model_keys:
        exploration = exploration_by_model.get(model_key, {})
        deployment_row = deployment_by_model.get(model_key, {})
        model = str(exploration.get("model") or deployment_row.get("model") or model_key)
        max_key = str(exploration.get("best_max_task_key") or "")
        max_task = tasks_by_key.get(max_key)
        deployment = deployment_row.get("tasks") or {}
        deployment_columns: dict[str, dict[str, Any]] = {}
        for task_key in deployment_keys:
            cell = deployment.get(task_key, {})
            sr = float(cell.get("sr") or 0)
            deployment_columns[task_key] = {
                "task_title": str(cell.get("task_title") or tasks_by_key[task_key].title),
                "sr": sr,
                "sr_display": f"{round(sr * 100):d}%",
                "mean_q": float(cell.get("mean_q") or 0),
                "runs": int(cell.get("runs") or 0),
                "operator_call_limit": int(cell.get("operator_call_limit") or 0),
                "violation_runs": int(cell.get("violation_runs") or 0),
                "limit_violation": bool(cell.get("limit_violation"))
                or int(cell.get("violation_runs") or 0) > 0,
            }
        rows.append(
            {
                "model": model,
                "model_short": short_model_name(model),
                "max_task_key": max_key,
                "exploration_rt_max_label": max_task.title if max_task else "—",
                "exploration_rt_max_icon": _craft_icon_path(world_mode, max_key),
                "exploration_q_max": int(exploration.get("exploration_q_max") or 0),
                "exploration_runs": int(exploration.get("runs") or 0),
                "deployment": deployment_columns,
            }
        )

    rows.sort(
        key=lambda row: (
            -_task_level(str(row.get("max_task_key") or ""), task_keys),
            -int(row.get("exploration_q_max") or 0),
            str(row.get("model_short") or "").lower(),
        )
    )
    return rows


def get_campaign_benchmark(since: str | None = None) -> dict[str, Any]:
    since_dt = parse_since_timestamp(since)
    exploration_entries = filter_entries_since(load_leaderboard_entries(limit=500), since_dt)
    test_entries = filter_entries_since(
        _load_jsonl(COMPANION_TEST_LOG_PATH, limit=500), since_dt
    )
    payload: dict[str, Any] = {
        "ok": True,
        "since": since_dt.isoformat() if since_dt else None,
        "world_modes": {},
    }
    for world_mode in ("craftax", "exo-planet"):
        tasks = _tasks_for_world_mode(world_mode)
        exploration_by_model = _aggregate_exploration_runs(exploration_entries, world_mode=world_mode)
        deployment_by_model = _latest_deployment_by_model_task(test_entries, world_mode=world_mode)
        deployment_keys = [key for key in _DEPLOYMENT_TASK_KEYS if key in _task_by_key(world_mode)]
        payload["world_modes"][world_mode] = {
            "world_mode": world_mode,
            "total_tasks": len(tasks),
            "deployment_task_keys": deployment_keys,
            "deployment_tasks": [
                {
                    "key": key,
                    "title": _task_by_key(world_mode)[key].title,
                    "icon": _craft_icon_path(world_mode, key),
                }
                for key in deployment_keys
            ],
            "compact": _build_compact_rows(exploration_by_model, world_mode=world_mode),
            "extended": _build_extended_rows(
                exploration_by_model, deployment_by_model, world_mode=world_mode
            ),
            "exploration_runs": sum(int(row.get("runs") or 0) for row in exploration_by_model.values()),
            "deployment_tests": sum(len((row.get("tasks") or {})) for row in deployment_by_model.values()),
        }
    payload["sources"] = {
        "exploration_log": str(LEADERBOARD_LOG_PATH.name),
        "deployment_log": str(COMPANION_TEST_LOG_PATH.name),
    }
    return payload
