from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model_names import canonical_model_key, short_model_name

TALKINGHEADS_PLAYER_NAME = "TalkingHeads"

CAMPAIGN_LEVEL_STEP_PENALTY = 6
CAMPAIGN_QUESTION_PENALTY = 15
CAMPAIGN_LEVEL_SCORE_FLOOR = 50
_CAMPAIGN_TASK_KEYS: tuple[str, ...] = (
    "collect_wood",
    "place_table",
    "make_wood_pickaxe",
    "collect_stone",
    "make_stone_pickaxe",
    "collect_coal",
    "collect_iron",
    "make_furnace",
    "make_iron_pickaxe",
    "collect_diamond",
)


def campaign_task_level_index(task_key: str, world_mode: str | None = None) -> int:
    _ = world_mode
    key = str(task_key or "").strip()
    if not key:
        return 0
    try:
        return _CAMPAIGN_TASK_KEYS.index(key) + 1
    except ValueError:
        return 0


def campaign_total_levels(world_mode: str | None = None) -> int:
    _ = world_mode
    return len(_CAMPAIGN_TASK_KEYS)


_DEFAULT_LEADERBOARD_DIR = Path(__file__).resolve().parent.parent / "leaderboard"
LEADERBOARD_DIR = Path(
    os.environ.get("PLAY_WEB_LEADERBOARD_DIR", str(_DEFAULT_LEADERBOARD_DIR))
).expanduser()
LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
LEADERBOARD_LOG_PATH = LEADERBOARD_DIR / "active_agent_runs.jsonl"
COMPANION_RESEARCH_LOG_PATH = LEADERBOARD_DIR / "companion_research.jsonl"
COMPANION_TEST_LOG_PATH = LEADERBOARD_DIR / "companion_test.jsonl"
ARC_HUMAN_LOG_PATH = LEADERBOARD_DIR / "arc_human.jsonl"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_leaderboard_entry(entry: dict[str, Any]) -> None:
    with LEADERBOARD_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_leaderboard_entries(limit: int = 300) -> list[dict[str, Any]]:
    if not LEADERBOARD_LOG_PATH.exists():
        return []
    entries: list[dict[str, Any]] = []
    with LEADERBOARD_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                entries.append(obj)
    if limit > 0 and len(entries) > limit:
        return entries[-limit:]
    return entries


def aggregate_model_leaderboard(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_model: dict[str, dict[str, Any]] = {}
    for entry in entries:
        model_raw = str(entry.get("active_agent_model") or "unknown").strip()
        model = model_raw or "unknown"
        model_key = canonical_model_key(model)
        prompt_name = str(entry.get("megaprompt_config_name") or "")
        phase1_levels = int(entry.get("phase1_completed_levels") or 0)
        phase2_highest = int(entry.get("phase2_highest_level") or 0)
        phase1_questions = int(entry.get("phase1_questions") or 0)
        phase2_questions = int(entry.get("phase2_questions") or 0)

        row = per_model.setdefault(
            model_key,
            {
                "active_agent_model": model,
                "model_short": short_model_name(model),
                "attempts": 0,
                "best_phase1_completed_levels": 0,
                "best_phase2_highest_level": 0,
                "best_megaprompt_config_name": "",
                "best_phase1_questions": 0,
                "best_phase2_questions": 0,
                "last_attempt_at": "",
            },
        )
        if model:
            row["active_agent_model"] = model
            row["model_short"] = short_model_name(model)
        row["attempts"] += 1
        row["last_attempt_at"] = str(entry.get("finished_at") or row["last_attempt_at"])
        # Aggregate by model across all attempts: phase1/phase2 maxima are independent.
        if phase1_levels > row["best_phase1_completed_levels"]:
            row["best_phase1_completed_levels"] = phase1_levels
            row["best_phase1_questions"] = phase1_questions
            row["best_megaprompt_config_name"] = prompt_name
        elif phase1_levels == row["best_phase1_completed_levels"] and phase1_levels > 0:
            if row["best_phase1_questions"] == 0 or phase1_questions < row["best_phase1_questions"]:
                row["best_phase1_questions"] = phase1_questions
                if prompt_name:
                    row["best_megaprompt_config_name"] = prompt_name

        if phase2_highest > row["best_phase2_highest_level"]:
            row["best_phase2_highest_level"] = phase2_highest
            row["best_phase2_questions"] = phase2_questions
        elif phase2_highest == row["best_phase2_highest_level"] and phase2_highest > 0:
            if row["best_phase2_questions"] == 0 or phase2_questions < row["best_phase2_questions"]:
                row["best_phase2_questions"] = phase2_questions

    rows = list(per_model.values())
    rows.sort(
        key=lambda x: (
            -int(x["best_phase1_completed_levels"]),
            -int(x["best_phase2_highest_level"]),
            int(x["best_phase1_questions"]),
            int(x["best_phase2_questions"]),
            str(x.get("model_short") or x["active_agent_model"]).lower(),
        )
    )
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


def _model_key(model: str) -> str:
    return canonical_model_key(model)


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                entries.append(obj)
    if limit > 0 and len(entries) > limit:
        return entries[-limit:]
    return entries


def append_companion_research_result(
    *,
    model: str,
    max_task: int,
    total_questions: int,
    mean_questions: float,
    max_ticks_per_task: int,
    research_complete: bool,
    source: str = "bench",
) -> None:
    completed = max(0, int(max_task))
    total_q = max(0, int(total_questions))
    mean_q = float(mean_questions)
    if mean_q <= 0 and completed > 0 and total_q > 0:
        mean_q = round(total_q / completed, 2)
    _append_jsonl(
        COMPANION_RESEARCH_LOG_PATH,
        {
            "finished_at": utc_now_iso(),
            "model": str(model or "unknown").strip() or "unknown",
            "max_task": completed,
            "total_questions": total_q,
            "mean_questions": round(mean_q, 2),
            "max_ticks_per_task": max(1, int(max_ticks_per_task or 1)),
            "research_complete": bool(research_complete),
            "source": str(source or "bench").strip() or "bench",
        },
    )


def append_companion_test_result(
    *,
    model: str,
    task_key: str,
    task_title: str,
    sr: float,
    mean_q: float,
    runs: int,
    successes: int,
    world_mode: str = "",
    operator_call_limit: int = 0,
    violation_runs: int = 0,
) -> None:
    _append_jsonl(
        COMPANION_TEST_LOG_PATH,
        {
            "finished_at": utc_now_iso(),
            "model": str(model or "unknown").strip() or "unknown",
            "task_key": str(task_key or "").strip(),
            "task_title": str(task_title or task_key or "—").strip() or "—",
            "sr": float(sr),
            "mean_q": round(float(mean_q), 2),
            "mean_questions": round(float(mean_q), 2),
            "runs": max(0, int(runs)),
            "successes": max(0, int(successes)),
            "world_mode": str(world_mode or "").strip(),
            "operator_call_limit": max(0, int(operator_call_limit)),
            "violation_runs": max(0, int(violation_runs)),
            "limit_violation": int(violation_runs) > 0,
        },
    )


def _normalize_player_avatar_id(value: Any) -> int:
    try:
        avatar_id = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(9, avatar_id))


def append_arc_human_result(entry: dict[str, Any]) -> dict[str, Any]:
    player_name = str(entry.get("player_name") or "").strip()[:40]
    if not player_name:
        raise ValueError("player_name is required")
    normalized = {
        "finished_at": str(entry.get("finished_at") or utc_now_iso()),
        "player_name": player_name,
        "player_avatar_id": _normalize_player_avatar_id(entry.get("player_avatar_id")),
        "score": max(0, int(entry.get("score") or 0)),
        "game_id": str(entry.get("game_id") or "unknown").strip() or "unknown",
        "state": str(entry.get("state") or "UNKNOWN").strip() or "UNKNOWN",
        "levels_completed": max(0, int(entry.get("levels_completed") or 0)),
        "actions": max(0, int(entry.get("actions") or 0)),
        "agent_actions": max(0, int(entry.get("agent_actions") or 0)),
        "manual_actions": max(0, int(entry.get("manual_actions") or 0)),
        "questions": max(0, int(entry.get("questions") or 0)),
        "human_answers": max(0, int(entry.get("human_answers") or 0)),
        "elapsed_seconds": max(0, int(entry.get("elapsed_seconds") or 0)),
        "active_agent_model": str(entry.get("active_agent_model") or "unknown").strip() or "unknown",
        "megaprompt_config_name": str(entry.get("megaprompt_config_name") or "").strip(),
        "leaderboard_type": "arc",
    }
    _append_jsonl(ARC_HUMAN_LOG_PATH, normalized)
    return normalized


def _normalize_level_steps(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in raw.items():
        task_key = str(key or "").strip()
        if not task_key:
            continue
        try:
            steps = max(0, int(value))
        except (TypeError, ValueError):
            steps = 0
        normalized[task_key] = steps
    return normalized


def _campaign_level_points(level_index: int, steps: int) -> int:
    if level_index <= 0:
        return 0
    base = level_index * 1000
    penalty = max(0, int(steps or 0)) * CAMPAIGN_LEVEL_STEP_PENALTY
    return max(CAMPAIGN_LEVEL_SCORE_FLOOR, base - penalty)


def _campaign_human_score(entry: dict[str, Any]) -> int:
    world_mode = str(entry.get("world_mode") or "craftax").strip() or "craftax"
    level_steps = _normalize_level_steps(entry.get("level_steps"))
    questions = int(entry.get("phase1_questions") or 0) + int(entry.get("phase2_questions") or 0)

    if level_steps:
        total = 0
        for task_key, steps in level_steps.items():
            level_index = campaign_task_level_index(task_key, world_mode)
            total += _campaign_level_points(level_index, steps)
        return max(0, total - questions * CAMPAIGN_QUESTION_PENALTY)

    # Legacy rows written before per-level step tracking.
    phase1 = int(entry.get("phase1_completed_levels") or 0)
    phase2 = int(entry.get("phase2_highest_level") or 0)
    return max(0, phase1 * 1000 + phase2 * 500 - questions * 10)


def _entry_interaction_mode(entry: dict[str, Any]) -> str:
    mode = str(entry.get("interaction_mode") or "").strip().lower()
    if mode in {"oracle", "human"}:
        return mode
    return "human"


def _campaign_leaderboard_row(entry: dict[str, Any]) -> dict[str, Any]:
    world_mode = str(entry.get("world_mode") or "craftax").strip() or "craftax"
    phase1 = int(entry.get("phase1_completed_levels") or 0)
    phase2 = int(entry.get("phase2_highest_level") or 0)
    questions = int(entry.get("phase1_questions") or 0) + int(entry.get("phase2_questions") or 0)
    level_steps = _normalize_level_steps(entry.get("level_steps"))
    agent_steps = int(entry.get("agent_steps") or 0)
    if agent_steps <= 0 and level_steps:
        agent_steps = sum(level_steps.values())
    total_levels = int(entry.get("total_levels") or 0)
    if total_levels <= 0:
        total_levels = campaign_total_levels(world_mode)
    interaction_mode = _entry_interaction_mode(entry)
    is_ai_operator = interaction_mode == "oracle"
    player_name = TALKINGHEADS_PLAYER_NAME if is_ai_operator else str(entry.get("player_name") or "").strip()
    return {
        "finished_at": str(entry.get("finished_at") or ""),
        "player_name": player_name,
        "player_avatar_id": _normalize_player_avatar_id(entry.get("player_avatar_id")),
        "world_mode": world_mode,
        "game_kind": "craftax",
        "score": _campaign_human_score(entry),
        "leaderboard_type": "campaign",
        "phase1_completed_levels": phase1,
        "phase2_highest_level": phase2,
        "total_levels": total_levels,
        "agent_steps": agent_steps,
        "level_steps": level_steps,
        "questions": questions,
        "finish_reason": str(entry.get("finish_reason") or ""),
        "active_agent_model": str(entry.get("active_agent_model") or "unknown").strip() or "unknown",
        "interaction_mode": interaction_mode,
        "is_ai_operator": is_ai_operator,
    }


def _campaign_human_row(entry: dict[str, Any]) -> dict[str, Any]:
    return _campaign_leaderboard_row(entry)


def _aggregate_best_human_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_player: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        is_ai_operator = bool(row.get("is_ai_operator"))
        if is_ai_operator:
            row = {**row, "player_name": TALKINGHEADS_PLAYER_NAME, "is_ai_operator": True}
        player_name = str(row.get("player_name") or "").strip()
        if not player_name:
            continue
        world_mode = str(row.get("world_mode") or "craftax").strip().lower() or "craftax"
        leaderboard_type = str(row.get("leaderboard_type") or "").strip().lower() or "run"
        if leaderboard_type == "arc":
            game_id = str(row.get("game_id") or "unknown").strip().lower() or "unknown"
            world_mode = f"arc:{game_id}"
        operator_bucket = "ai" if is_ai_operator else "human"
        bucket_key = (player_name.lower(), world_mode, leaderboard_type, operator_bucket)
        prev = best_by_player.get(bucket_key)
        if prev is None:
            best_by_player[bucket_key] = row
            continue
        row_score = int(row.get("score") or 0)
        prev_score = int(prev.get("score") or 0)
        if row_score > prev_score:
            best_by_player[bucket_key] = row
        elif row_score == prev_score:
            row_steps = int(row.get("agent_steps") or 0)
            prev_steps = int(prev.get("agent_steps") or 0)
            if row_steps < prev_steps:
                best_by_player[bucket_key] = row
            elif row_steps == prev_steps and str(row.get("finished_at") or "") > str(prev.get("finished_at") or ""):
                best_by_player[bucket_key] = row
    return list(best_by_player.values())


def get_human_leaderboard(
    *,
    game_kind: str | None = None,
    world_mode: str | None = None,
    arc_game_id: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    normalized_kind = str(game_kind or "").strip().lower()
    normalized_world = str(world_mode or "").strip().lower()
    normalized_arc_game = str(arc_game_id or "").strip().lower()
    rows: list[dict[str, Any]] = []

    if not normalized_kind or normalized_kind == "arc_agi":
        for entry in _load_jsonl(ARC_HUMAN_LOG_PATH, limit=500):
            entry_game_id = str(entry.get("game_id") or "").strip().lower()
            if normalized_arc_game and entry_game_id != normalized_arc_game:
                continue
            row = dict(entry)
            row.setdefault("leaderboard_type", "arc")
            row.setdefault("player_avatar_id", 0)
            rows.append(row)

    if not normalized_kind or normalized_kind == "craftax":
        for entry in load_leaderboard_entries(limit=500):
            entry_world = str(entry.get("world_mode") or "craftax").strip().lower()
            if normalized_world and entry_world != normalized_world:
                continue
            interaction_mode = _entry_interaction_mode(entry)
            if interaction_mode != "oracle":
                player_name = str(entry.get("player_name") or "").strip()
                if not player_name:
                    continue
            rows.append(_campaign_leaderboard_row(entry))

    rows = _aggregate_best_human_rows(rows)
    rows.sort(
        key=lambda row: (
            -int(row.get("score") or 0),
            -int(row.get("phase1_completed_levels") or row.get("levels_completed") or 0),
            int(row.get("agent_steps") or 0),
            int(row.get("human_answers") or 0),
            str(row.get("finished_at") or ""),
        ),
    )
    capped = rows[: max(1, int(limit or 100))]
    for idx, row in enumerate(capped, 1):
        row["rank"] = idx
    return {"ok": True, "rows": capped, "total": len(rows)}


def get_arc_human_leaderboard(
    limit: int = 100,
    game_id: str | None = None,
) -> dict[str, Any]:
    safe_limit = max(1, int(limit or 100))
    normalized_game_id = str(game_id or "").strip().lower()
    load_limit = max(500, safe_limit)
    entries = _load_jsonl(ARC_HUMAN_LOG_PATH, limit=load_limit)
    if normalized_game_id:
        entries = [
            entry for entry in entries
            if str(entry.get("game_id") or "").strip().lower() == normalized_game_id
        ]
    rows = sorted(
        entries,
        key=lambda row: (
            -int(row.get("score") or 0),
            int(row.get("actions") or 0),
            int(row.get("questions") or 0),
            int(row.get("elapsed_seconds") or 0),
            str(row.get("finished_at") or ""),
        ),
    )
    rows = rows[:safe_limit]
    for idx, row in enumerate(rows, 1):
        row["rank"] = idx
    return {"ok": True, "rows": rows, "total": len(entries)}


def latest_research_by_model(
    entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_model: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = _model_key(str(entry.get("model") or ""))
        prev = by_model.get(key)
        finished = str(entry.get("finished_at") or "")
        prev_finished = str(prev.get("finished_at") or "") if prev else ""
        if prev is None or finished >= prev_finished:
            by_model[key] = entry
    return by_model


def build_companion_leaderboard_rows(
    test_entries: list[dict[str, Any]],
    research_by_model: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per canonical model + task (latest test entry wins)."""
    latest_by_model_task: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in test_entries:
        model = str(entry.get("model") or "unknown").strip() or "unknown"
        task_key = str(entry.get("task_key") or "").strip()
        model_key = _model_key(model)
        bucket_key = (model_key, task_key)
        finished = str(entry.get("finished_at") or "")
        prev = latest_by_model_task.get(bucket_key)
        if prev is None or finished >= str(prev.get("finished_at") or ""):
            latest_by_model_task[bucket_key] = {**entry, "model": model}

    rows: list[dict[str, Any]] = []
    for entry in sorted(
        latest_by_model_task.values(),
        key=lambda row: (
            str(row.get("finished_at") or ""),
            _model_key(str(row.get("model") or "")),
            str(row.get("task_key") or ""),
        ),
        reverse=True,
    ):
        model = str(entry.get("model") or "unknown").strip() or "unknown"
        research = research_by_model.get(_model_key(model), {})
        rows.append(
            {
                "model": model,
                "model_short": short_model_name(model),
                "research_max_task": int(research.get("max_task") or 0),
                "research_mean_questions": float(research.get("mean_questions") or 0),
                "task_title": str(entry.get("task_title") or entry.get("task_key") or "—"),
                "task_key": str(entry.get("task_key") or ""),
                "sr": float(entry.get("sr") or 0),
                "mean_q": float(entry.get("mean_q") or entry.get("mean_questions") or 0),
                "runs": int(entry.get("runs") or 0),
                "finished_at": str(entry.get("finished_at") or ""),
            }
        )
    return rows


def get_companion_leaderboard(
    extra_test_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    research_entries = _load_jsonl(COMPANION_RESEARCH_LOG_PATH, limit=300)
    test_entries = _load_jsonl(COMPANION_TEST_LOG_PATH, limit=500)
    if extra_test_rows:
        normalized_extra = []
        for row in extra_test_rows:
            if not isinstance(row, dict):
                continue
            normalized_extra.append(
                {
                    "finished_at": str(row.get("finished_at") or utc_now_iso()),
                    "model": str(row.get("model") or "unknown"),
                    "task_key": str(row.get("task_key") or ""),
                    "task_title": str(row.get("task_title") or row.get("task_key") or "—"),
                    "sr": float(row.get("sr") or 0),
                    "mean_q": float(row.get("mean_q") or row.get("mean_questions") or 0),
                    "runs": int(row.get("runs") or 0),
                }
            )
        test_entries = test_entries + normalized_extra
    research_by_model = latest_research_by_model(research_entries)
    rows = build_companion_leaderboard_rows(test_entries, research_by_model)
    return {
        "ok": True,
        "rows": rows,
        "total_tests": len(test_entries),
        "total_research": len(research_entries),
    }
