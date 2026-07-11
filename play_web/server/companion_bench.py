from __future__ import annotations

import asyncio
import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oracle.knowledge import (
    copy_durable_knowledge,
    load_durable_knowledge_records,
    load_knowledge,
    read_starter_revision,
    save_knowledge_records,
    use_knowledge_paths,
    write_starter_revision,
)
from oracle.prompts.prompt_generation import is_exo_world_mode, normalize_world_mode

from .campaign_mode import campaign_tasks_for_world_mode, TaskDefinition
from .deployment_operator_limits import (
    deployment_megaprompt_config_name,
    is_deployment_megaprompt,
    operator_call_limit_for_task,
    operator_call_limit_violated,
)
from .leaderboard import append_companion_research_result, append_companion_test_result
from .runtime import create_isolated_session


_REPO_ROOT = Path(__file__).resolve().parents[2]
_KNOWLEDGE_ROOT = _REPO_ROOT / "MegaPrompt" / "craftext_prompt"
_EXO_KNOWLEDGE_ROOT = _REPO_ROOT / "MegaPrompt" / "exo-planet_prompt"
DEFAULT_COMPANION_MAX_TICKS_PER_TASK = 150


def _bench_dir() -> Path:
    path = _KNOWLEDGE_ROOT / "companion_bench"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _world_slug(world_mode: str | None) -> str:
    return "exo_planet" if is_exo_world_mode(world_mode) else "craftax"


def _base_knowledge_paths(world_mode: str | None) -> tuple[Path, Path]:
    if is_exo_world_mode(world_mode):
        return (
            _EXO_KNOWLEDGE_ROOT / "knowledge_data.json",
            _EXO_KNOWLEDGE_ROOT / "knowledge_data.txt",
        )
    return (
        _KNOWLEDGE_ROOT / "knowledge_data.json",
        _KNOWLEDGE_ROOT / "knowledge_data.txt",
    )


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip()).strip("_")
    return cleaned or "unknown_model"


def _knowledge_paths(filename_base: str) -> tuple[Path, Path]:
    bench = _bench_dir()
    return bench / f"{filename_base}.json", bench / f"{filename_base}.txt"


def _starter_skills(entries: list[dict[str, Any]]) -> set[str]:
    return {str(entry.get("skill") or "") for entry in entries}


_LEGACY_STARTER_ROW_CAP = 6


def _is_replaceable_starter_row(
    entry: dict[str, Any],
    *,
    base_skills: set[str],
    max_base_id: int,
    model_revision: int,
    base_revision: int,
) -> bool:
    skill = str(entry.get("skill") or "")
    if skill in base_skills or skill == "memory_format_guide" or skill.startswith("example_"):
        return True
    if model_revision >= base_revision:
        return False
    try:
        row_id = int(entry.get("id") or 0)
    except (TypeError, ValueError):
        return False
    starter_cap = max(max_base_id, _LEGACY_STARTER_ROW_CAP)
    return 1 <= row_id <= starter_cap


def _merge_refreshed_starter_rows(
    base_entries: list[dict[str, Any]],
    model_entries: list[dict[str, Any]],
    *,
    model_revision: int,
    base_revision: int,
) -> list[dict[str, Any]]:
    base_skills = _starter_skills(base_entries)
    max_base_id = max((int(entry.get("id") or 0) for entry in base_entries), default=0)
    kept = [
        entry
        for entry in model_entries
        if not _is_replaceable_starter_row(
            entry,
            base_skills=base_skills,
            max_base_id=max_base_id,
            model_revision=model_revision,
            base_revision=base_revision,
        )
    ]
    return base_entries + kept


def _seed_knowledge_from_main_source(
    target_json: Path,
    target_txt: Path,
    *,
    world_mode: str | None,
    existing_model_entries: list[dict[str, Any]] | None = None,
    model_revision: int = 0,
) -> None:
    """Copy durable base knowledge for the active world mode (never TYPE=NOTE)."""
    source_json, source_txt = _base_knowledge_paths(world_mode)
    source_entries: list[dict[str, Any]] = []
    starter_revision = 0
    if source_json.exists() or source_txt.exists():
        with use_knowledge_paths(json_path=source_json, txt_path=source_txt):
            source_entries = load_durable_knowledge_records()
        starter_revision = read_starter_revision(source_json)
    if existing_model_entries is not None:
        merged = _merge_refreshed_starter_rows(
            source_entries,
            existing_model_entries,
            model_revision=model_revision,
            base_revision=starter_revision,
        )
    else:
        merged = source_entries
    with use_knowledge_paths(json_path=target_json, txt_path=target_txt):
        save_knowledge_records(merged)
    if starter_revision:
        write_starter_revision(target_json, starter_revision)


def ensure_model_knowledge_current(model: str, world_mode: str | None) -> None:
    """Refresh per-model starter rows when base starter_revision is newer."""
    model_json, model_txt = model_knowledge_paths(model, world_mode)
    base_json, _base_txt = _base_knowledge_paths(world_mode)
    base_revision = read_starter_revision(base_json)
    if base_revision <= 0:
        _ensure_model_seeded_knowledge(model_json, model_txt, world_mode=world_mode)
        return
    model_revision = read_starter_revision(model_json)
    if model_json.exists() and model_revision >= base_revision:
        return
    existing_model_entries: list[dict[str, Any]] = []
    if model_json.exists() or model_txt.exists():
        with use_knowledge_paths(json_path=model_json, txt_path=model_txt):
            existing_model_entries = load_durable_knowledge_records()
    _seed_knowledge_from_main_source(
        model_json,
        model_txt,
        world_mode=world_mode,
        existing_model_entries=existing_model_entries,
        model_revision=model_revision,
    )


def _ensure_model_seeded_knowledge(
    target_json: Path,
    target_txt: Path,
    *,
    world_mode: str | None,
) -> None:
    """For a new model/world pair, initialize benchmark knowledge once from base."""
    if target_json.exists() or target_txt.exists():
        return
    _seed_knowledge_from_main_source(target_json, target_txt, world_mode=world_mode)


def _has_own_knowledge(target_json: Path, target_txt: Path) -> bool:
    return target_json.exists() or target_txt.exists()


def _legacy_model_knowledge_paths(model: str) -> tuple[Path, Path]:
    model_slug = _slugify(model)
    return _knowledge_paths(f"knowlage_data_{model_slug}")


def model_knowledge_paths(model: str, world_mode: str | None = None) -> tuple[Path, Path]:
    """Per-model research knowledge for a specific world mode."""
    model_slug = _slugify(model)
    world_slug = _world_slug(world_mode)
    return _knowledge_paths(f"knowlage_data_{model_slug}__{world_slug}")


def resolve_model_knowledge_paths(model: str, world_mode: str | None = None) -> tuple[Path, Path]:
    """Prefer world-specific research files; fall back to legacy per-model paths."""
    primary = model_knowledge_paths(model, world_mode)
    if _has_own_knowledge(primary[0], primary[1]):
        return primary
    legacy = _legacy_model_knowledge_paths(model)
    if _has_own_knowledge(legacy[0], legacy[1]):
        return legacy
    return primary


def _prepare_research_knowledge(
    *,
    knowledge_source: str,
    target_json: Path,
    target_txt: Path,
    world_mode: str | None,
) -> None:
    source = str(knowledge_source or "own").strip().lower()
    if source == "base":
        _seed_knowledge_from_main_source(target_json, target_txt, world_mode=world_mode)
        return
    _ensure_model_seeded_knowledge(target_json, target_txt, world_mode=world_mode)


def _prepare_test_agent_knowledge(
    *,
    knowledge_source: str,
    world_mode: str | None,
    model: str,
    agent_json: Path,
    agent_txt: Path,
) -> None:
    """Seed an isolated test agent file from base or model research (durable rows only)."""
    source = str(knowledge_source or "base").strip().lower()
    if source == "base":
        _seed_knowledge_from_main_source(agent_json, agent_txt, world_mode=world_mode)
        return
    model_json, model_txt = resolve_model_knowledge_paths(model, world_mode)
    if not _has_own_knowledge(model_json, model_txt):
        _seed_knowledge_from_main_source(agent_json, agent_txt, world_mode=world_mode)
        return
    copy_durable_knowledge(
        source_json=model_json,
        source_txt=model_txt,
        dest_json=agent_json,
        dest_txt=agent_txt,
    )


def _campaign_tasks_snapshot(world_mode: str | None = None) -> list[dict[str, str]]:
    tasks = campaign_tasks_for_world_mode(world_mode)
    return [{"key": task.key, "title": task.title, "goal": task.goal} for task in tasks]


def knowledge_file_info_for_model(model: str, world_mode: str | None = None) -> dict[str, Any]:
    model_slug = _slugify(model)
    world_slug = _world_slug(world_mode)
    model_json, model_txt = resolve_model_knowledge_paths(model, world_mode)
    base_json, base_txt = _base_knowledge_paths(world_mode)
    return {
        "model_knowledge_file": model_json.name,
        "model_knowledge_txt_file": model_txt.name,
        "base_knowledge_file": base_json.name,
        "base_knowledge_txt_file": base_txt.name,
        "has_own_knowledge": _has_own_knowledge(model_json, model_txt),
        "model_slug": model_slug,
        "world_mode": world_slug,
    }


def _safe_copy_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(snapshot)


def _task_by_key(task_key: str, world_mode: str | None = None) -> TaskDefinition:
    for task in campaign_tasks_for_world_mode(world_mode):
        if task.key == task_key:
            return task
    raise ValueError(f"Unknown task key: {task_key}")


@dataclass
class AgentEpisodeResult:
    agent_id: int
    task_key: str
    task_title: str
    success: bool
    questions_count: int
    achieved_task_keys: list[str] = field(default_factory=list)
    knowledge_file: str = ""
    error: str = ""
    ticks_used: int = 0
    operator_call_limit: int = 0
    limit_violated: bool = False


@dataclass
class TaskAggregateResult:
    task_key: str
    task_title: str
    runs: int
    successes: int
    sr: float
    mean_q: float
    per_agent: list[AgentEpisodeResult] = field(default_factory=list)
    operator_call_limit: int = 0
    violation_runs: int = 0


@dataclass
class BenchmarkRuntimeOverrides:
    interaction_mode: str = "oracle"
    allowed_experts: list[str] = field(default_factory=list)
    forced_expert: str | None = None
    expert_models: dict[str, str] = field(default_factory=dict)
    expert_modes: dict[str, str] = field(default_factory=dict)
    active_agent_do_sample: bool = True
    active_agent_temperature: float | None = None
    active_agent_top_p: float | None = None
    hf_token: str = ""
    openrouter_api_key: str = ""


class CompanionBenchRunner:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._job: asyncio.Task | None = None
        self._state: dict[str, Any] = self._empty_state()
        self._active_sessions: list[Any] = []

    def _active_tasks(self) -> list[TaskDefinition]:
        return campaign_tasks_for_world_mode(self._state.get("world_mode"))

    def _empty_state(self) -> dict[str, Any]:
        default_world_mode = "exo-planet"
        default_tasks = campaign_tasks_for_world_mode(default_world_mode)
        return {
            "running": False,
            "world_mode": default_world_mode,
            "model": "",
            "mode": "openrouter",
            "phase": "",
            "parallel_agents": 3,
            "cycles": 1,
            "current_cycle": 0,
            "max_ticks_per_task": DEFAULT_COMPANION_MAX_TICKS_PER_TASK,
            "test_task_key": "",
            "knowledge_source": "own",
            "has_own_knowledge": False,
            "research_complete": False,
            "current_stage": "",
            "current_task_key": "",
            "current_task_title": "",
            "knowledge_preview": "",
            "knowledge_file": "",
            "agents_live": [],
            "agents_progress": [],
            "rows": [],
            "campaign_tasks": _campaign_tasks_snapshot(default_world_mode),
            "error": "",
            "completed": False,
            "progress_done": 0,
            "progress_total": 0,
            "progress_pct": 0,
            "progress_label": "",
            "baseline_task_index": 0,
            "baseline_task_total": len(default_tasks),
            "episode_tick": 0,
            "episode_max_ticks": 0,
            "research_progress_pct": 0,
            "research_observation": "",
            "research_frame": None,
            "stop_requested": False,
        }

    def _frame_snapshot(self, session: Any) -> dict[str, Any] | None:
        try:
            rendered = session.render_frame()
            png_b64 = str(rendered.get("png_b64") or "")
            if not png_b64:
                return None
            return {
                "w": int(rendered.get("w") or 0),
                "h": int(rendered.get("h") or 0),
                "png_b64": png_b64,
            }
        except Exception:
            return None

    def _sync_research_view(self, session: Any) -> None:
        frame = self._frame_snapshot(session)
        if frame is not None:
            self._state["research_frame"] = frame
        try:
            self._state["research_observation"] = session.get_agent_observation()
        except Exception:
            self._state["research_observation"] = ""

    async def _touch_state(self) -> None:
        """Yield to the event loop so /companion_bench/status can respond while running."""
        await asyncio.sleep(0)

    def _set_progress(self, done: int, total: int, label: str = "") -> None:
        safe_total = max(1, int(total))
        safe_done = max(0, min(int(done), safe_total))
        self._state["progress_done"] = safe_done
        self._state["progress_total"] = safe_total
        self._state["progress_pct"] = int(round((100.0 * safe_done) / safe_total))
        self._state["progress_label"] = str(label or "")

    def _set_research_progress(self, task_index: int, episode_tick: int, episode_max: int) -> None:
        total_tasks = max(1, int(self._state.get("baseline_task_total") or len(self._active_tasks())))
        safe_task_index = max(0, min(int(task_index), total_tasks))
        safe_episode_max = max(1, int(episode_max))
        safe_episode_tick = max(0, min(int(episode_tick), safe_episode_max))
        self._state["baseline_task_index"] = safe_task_index
        self._state["episode_tick"] = safe_episode_tick
        self._state["episode_max_ticks"] = safe_episode_max
        task_fraction = (safe_task_index - 1) + (safe_episode_tick / safe_episode_max)
        self._state["research_progress_pct"] = int(
            round((100.0 * max(0.0, task_fraction)) / total_tasks)
        )

    async def start(
        self,
        *,
        phase: str,
        model: str,
        mode: str,
        megaprompt_config_name: str,
        parallel_agents: int = 3,
        max_ticks_per_task: int = DEFAULT_COMPANION_MAX_TICKS_PER_TASK,
        cycles: int = 1,
        task_key: str = "",
        knowledge_source: str = "own",
        world_mode: str = "exo-planet",
        runtime_overrides: BenchmarkRuntimeOverrides | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            if self._job is not None and not self._job.done():
                return {"ok": False, "error": "Benchmark is already running."}
            safe_phase = str(phase or "").strip().lower()
            if safe_phase not in {"research", "test"}:
                return {"ok": False, "error": "phase must be 'research' or 'test'"}
            if safe_phase == "test" and not str(task_key or "").strip():
                return {"ok": False, "error": "task_key is required for test phase"}

            resolved_megaprompt = megaprompt_config_name
            if safe_phase == "test":
                resolved_megaprompt = deployment_megaprompt_config_name(world_mode)

            self._state = self._empty_state()
            resolved_world_mode = normalize_world_mode(str(world_mode or "exo-planet").strip())
            self._state["world_mode"] = resolved_world_mode
            campaign_tasks = self._active_tasks()
            model_slug = _slugify(model)
            model_json, model_txt = model_knowledge_paths(model, resolved_world_mode)
            has_own = _has_own_knowledge(*resolve_model_knowledge_paths(model, resolved_world_mode))
            first_task = campaign_tasks[0] if campaign_tasks else None
            safe_parallel = int(max(1, parallel_agents))
            safe_cycles = int(max(1, cycles))
            safe_max_ticks = int(max(1, max_ticks_per_task))
            safe_knowledge_source = str(knowledge_source or ("base" if safe_phase == "test" else "own")).strip().lower()
            if safe_knowledge_source not in {"base", "own"}:
                safe_knowledge_source = "base" if safe_phase == "test" else "own"

            if safe_phase == "research":
                baseline_json, baseline_txt = model_json, model_txt
                _prepare_research_knowledge(
                    knowledge_source=safe_knowledge_source,
                    target_json=baseline_json,
                    target_txt=baseline_txt,
                    world_mode=resolved_world_mode,
                )
                total_units = len(campaign_tasks)
                progress_label = (
                    f"Research: {first_task.title} (1/{len(campaign_tasks)})"
                    if first_task
                    else "Starting research"
                )
                current_stage = "baseline"
            else:
                task = _task_by_key(str(task_key).strip(), self._state.get("world_mode"))
                total_units = safe_cycles * safe_parallel
                progress_label = f"Test: {task.title} (cycle 1/{safe_cycles})"
                current_stage = f"parallel:{task.key}"
                if safe_knowledge_source == "base":
                    baseline_json, baseline_txt = _base_knowledge_paths(resolved_world_mode)
                else:
                    baseline_json, baseline_txt = resolve_model_knowledge_paths(model, resolved_world_mode)

            self._state.update(
                {
                    "running": True,
                    "model": model,
                    "mode": mode,
                    "phase": safe_phase,
                    "parallel_agents": safe_parallel,
                    "cycles": safe_cycles,
                    "current_cycle": 0,
                    "max_ticks_per_task": safe_max_ticks,
                    "test_task_key": str(task_key or "").strip(),
                    "knowledge_source": safe_knowledge_source,
                    "has_own_knowledge": has_own,
                    "current_stage": current_stage,
                    "knowledge_file": baseline_json.name,
                    "baseline_task_total": len(campaign_tasks),
                    "campaign_tasks": _campaign_tasks_snapshot(self._state.get("world_mode")),
                    "baseline_task_index": 1 if first_task and safe_phase == "research" else 0,
                    "episode_tick": 0,
                    "episode_max_ticks": safe_max_ticks,
                }
            )
            if safe_phase == "research" and first_task:
                self._state["current_task_key"] = first_task.key
                self._state["current_task_title"] = first_task.title
            elif safe_phase == "test":
                task = _task_by_key(str(task_key).strip(), self._state.get("world_mode"))
                self._state["current_task_key"] = task.key
                self._state["current_task_title"] = task.title
            self._set_progress(0, total_units, progress_label)
            if safe_phase == "research":
                self._set_research_progress(1, 0, safe_max_ticks)
            if baseline_json.exists() or baseline_txt.exists():
                with use_knowledge_paths(json_path=baseline_json, txt_path=baseline_txt):
                    self._state["knowledge_preview"] = load_knowledge()
            self._job = asyncio.create_task(
                self._run(
                    phase=safe_phase,
                    model=model,
                    mode=mode,
                    megaprompt_config_name=resolved_megaprompt,
                    parallel_agents=safe_parallel,
                    max_ticks_per_task=safe_max_ticks,
                    cycles=safe_cycles,
                    task_key=str(task_key or "").strip(),
                    knowledge_source=safe_knowledge_source,
                    runtime_overrides=runtime_overrides,
                )
            )
            return {"ok": True, "status": _safe_copy_snapshot(self._state)}

    async def status(self, *, model: str = "", world_mode: str | None = None) -> dict[str, Any]:
        snapshot = _safe_copy_snapshot(self._state)
        resolved_model = str(snapshot.get("model") or model or "").strip()
        resolved_world_mode = world_mode or snapshot.get("world_mode")
        snapshot.update(knowledge_file_info_for_model(resolved_model, resolved_world_mode))
        snapshot["campaign_tasks"] = _campaign_tasks_snapshot(resolved_world_mode)
        if resolved_world_mode:
            snapshot["world_mode"] = normalize_world_mode(resolved_world_mode)
        return {"ok": True, "status": snapshot}

    async def stop(self) -> dict[str, Any]:
        self._state["stop_requested"] = True
        self._request_stop_on_active_sessions()
        if self._state.get("running"):
            self._state["progress_label"] = "Stopping after current step..."
        return {"ok": True, "status": _safe_copy_snapshot(self._state)}

    def _register_session(self, session: Any) -> None:
        self._active_sessions.append(session)
        if self._state.get("stop_requested"):
            try:
                session.stop_agent_requested = True
            except Exception:
                pass

    def _unregister_session(self, session: Any) -> None:
        self._active_sessions = [s for s in self._active_sessions if s is not session]

    def _request_stop_on_active_sessions(self) -> None:
        for session in list(self._active_sessions):
            try:
                session.stop_agent_requested = True
            except Exception:
                continue

    async def _run(
        self,
        *,
        phase: str,
        model: str,
        mode: str,
        megaprompt_config_name: str,
        parallel_agents: int,
        max_ticks_per_task: int,
        cycles: int,
        task_key: str,
        knowledge_source: str,
        runtime_overrides: BenchmarkRuntimeOverrides | None,
    ) -> None:
        try:
            model_slug = _slugify(model)
            world_mode = self._state.get("world_mode")

            if phase == "research":
                baseline_json, baseline_txt = model_knowledge_paths(model, world_mode)
                campaign_tasks = self._active_tasks()
                if campaign_tasks:
                    self._state["current_task_key"] = campaign_tasks[0].key
                    self._state["current_task_title"] = campaign_tasks[0].title
                total_units = len(campaign_tasks)
                self._set_progress(0, total_units, "Starting research")
                await self._run_baseline(
                    model=model,
                    mode=mode,
                    megaprompt_config_name=megaprompt_config_name,
                    max_ticks_per_task=max_ticks_per_task,
                    knowledge_json=baseline_json,
                    knowledge_txt=baseline_txt,
                    total_units=total_units,
                    runtime_overrides=runtime_overrides,
                )
                if self._state.get("stop_requested"):
                    self._state["current_stage"] = "stopped"
                    self._state["progress_label"] = "Stopped by user"
                    return
                self._state["knowledge_file"] = baseline_json.name
                self._state["has_own_knowledge"] = True
                with use_knowledge_paths(json_path=baseline_json, txt_path=baseline_txt):
                    self._state["knowledge_preview"] = load_knowledge()
                self._state["current_stage"] = "research_complete"
                self._state["research_complete"] = True
                self._state["completed"] = True
                self._set_progress(total_units, total_units, "Research complete")
                return

            task = _task_by_key(task_key, self._state.get("world_mode"))
            total_units = cycles * parallel_agents
            self._set_progress(0, total_units, f"Test: {task.title} (cycle 1/{cycles})")
            test_result = await self._run_parallel_task(
                task=task,
                parallel_agents=parallel_agents,
                cycles=cycles,
                model=model,
                mode=mode,
                megaprompt_config_name=megaprompt_config_name,
                max_ticks_per_task=max_ticks_per_task,
                model_slug=model_slug,
                knowledge_source=knowledge_source,
                world_mode=world_mode,
                progress_offset=0,
                total_units=total_units,
                runtime_overrides=runtime_overrides,
            )
            self._append_row(model, test_result)
            if self._state.get("stop_requested"):
                self._state["current_stage"] = "stopped"
                self._state["progress_label"] = "Stopped by user"
                return
            self._state["current_stage"] = "finished"
            self._state["completed"] = True
            self._set_progress(total_units, total_units, "Test complete")
        except Exception as e:
            self._state["error"] = str(e)
        finally:
            self._state["running"] = False

    async def _run_baseline(
        self,
        *,
        model: str,
        mode: str,
        megaprompt_config_name: str,
        max_ticks_per_task: int,
        knowledge_json: Path,
        knowledge_txt: Path,
        total_units: int,
        runtime_overrides: BenchmarkRuntimeOverrides | None,
    ) -> None:
        self._state["current_stage"] = "baseline"
        self._state["knowledge_file"] = knowledge_json.name
        session = create_isolated_session()
        self._register_session(session)
        try:
            session.set_texture_theme(str(self._state.get("world_mode") or "craftax"))
            session.set_active_agent_model(model)
            session.set_active_agent_mode(mode)
            session.set_megaprompt_config_name(megaprompt_config_name)
            self._apply_runtime_overrides(session, runtime_overrides)
            with use_knowledge_paths(json_path=knowledge_json, txt_path=knowledge_txt):
                # Baseline is a single continuous episode:
                # one reset, then task goals advance without resetting world state.
                session.reset()
                self._state["knowledge_preview"] = load_knowledge()
                self._sync_research_view(session)
                await self._touch_state()
                per_task_budget = int(max(1, max_ticks_per_task))
                task_idx = 0
                task_initial_state = session.state
                research_questions = 0
                campaign_tasks = self._active_tasks()

                while (
                    task_idx < len(campaign_tasks)
                    and not self._state.get("stop_requested")
                ):
                    task = campaign_tasks[task_idx]
                    display_idx = task_idx + 1
                    task_ticks = 0
                    self._state["current_task_key"] = task.key
                    self._state["current_task_title"] = task.title
                    self._set_research_progress(display_idx, 0, per_task_budget)
                    self._set_progress(
                        task_idx,
                        total_units,
                        f"Research: {task.title} ({display_idx}/{len(campaign_tasks)})",
                    )
                    self._sync_research_view(session)
                    await self._touch_state()

                    while (
                        task_ticks < per_task_budget
                        and not task.check(task_initial_state, session.state)
                        and not self._state.get("stop_requested")
                    ):
                        async for payload in session.handle_agent_tick(steps=1, goal=task.goal):
                            if payload.get("type") == "agent_question_pending":
                                research_questions += 1
                            self._sync_research_view(session)
                            await self._touch_state()

                        task_ticks += 1
                        self._set_research_progress(display_idx, task_ticks, per_task_budget)
                        self._state["episode_tick"] = task_ticks
                        self._state["episode_max_ticks"] = per_task_budget
                        self._state["knowledge_preview"] = load_knowledge()
                        await self._touch_state()

                    if task.check(task_initial_state, session.state):
                        self._set_progress(
                            display_idx,
                            total_units,
                            f"Research: {task.title} complete ({display_idx}/{len(campaign_tasks)})",
                        )
                        task_idx += 1
                        task_initial_state = session.state
                        await self._touch_state()
                    else:
                        self._state["progress_label"] = (
                            f"Research: budget exhausted on {task.title} "
                            f"({task_ticks}/{per_task_budget} steps)"
                        )
                        break
                completed_tasks = min(task_idx, len(campaign_tasks))
                research_complete = task_idx >= len(campaign_tasks)
                if research_complete:
                    completed_tasks = len(campaign_tasks)
                mean_research_q = (
                    round(research_questions / max(completed_tasks, 1), 2)
                    if research_questions
                    else 0.0
                )
                append_companion_research_result(
                    model=model,
                    max_task=completed_tasks,
                    total_questions=research_questions,
                    mean_questions=mean_research_q,
                    max_ticks_per_task=per_task_budget,
                    research_complete=research_complete,
                    source="bench",
                )
                self._state["baseline_task_index"] = completed_tasks
        finally:
            self._unregister_session(session)

    async def _run_parallel_task(
        self,
        *,
        task: TaskDefinition,
        parallel_agents: int,
        cycles: int,
        model: str,
        mode: str,
        megaprompt_config_name: str,
        max_ticks_per_task: int,
        model_slug: str,
        knowledge_source: str,
        world_mode: str | None,
        progress_offset: int,
        total_units: int,
        runtime_overrides: BenchmarkRuntimeOverrides | None,
    ) -> TaskAggregateResult:
        self._state["current_stage"] = f"parallel:{task.key}"
        self._state["current_task_key"] = task.key
        self._state["current_task_title"] = task.title
        self._state["agents_live"] = []
        self._state["agents_progress"] = [
            {
                "agent_id": agent_id,
                "tick": 0,
                "max_ticks": max_ticks_per_task,
                "progress_pct": 0,
                "done": False,
            }
            for agent_id in range(1, parallel_agents + 1)
        ]
        per_agent: list[AgentEpisodeResult] = []
        completed_units = 0
        world_slug = _world_slug(world_mode)
        for cycle_idx in range(1, int(max(1, cycles)) + 1):
            if self._state.get("stop_requested"):
                break
            self._state["current_cycle"] = cycle_idx
            self._state["progress_label"] = (
                f"Test: {task.title} (cycle {cycle_idx}/{cycles})"
            )
            jobs = []
            for agent_id in range(1, parallel_agents + 1):
                if self._state.get("stop_requested"):
                    break
                file_base = f"knowlage_data_{model_slug}__{world_slug}_{agent_id}st_agent"
                json_path, txt_path = _knowledge_paths(file_base)
                jobs.append(
                    self._run_agent_worker(
                        agent_id=agent_id,
                        task=task,
                        model=model,
                        mode=mode,
                        megaprompt_config_name=megaprompt_config_name,
                        max_ticks=max_ticks_per_task,
                        knowledge_json=json_path,
                        knowledge_txt=txt_path,
                        knowledge_source=knowledge_source,
                        world_mode=world_mode,
                        progress_offset=progress_offset + completed_units,
                        total_units=total_units,
                        runtime_overrides=runtime_overrides,
                    )
                )
            cycle_results = list(await asyncio.gather(*jobs))
            per_agent.extend(cycle_results)
            completed_units += len(cycle_results)
            self._set_progress(
                progress_offset + completed_units,
                total_units,
                f"{task.title}: cycle {cycle_idx}/{cycles} complete",
            )
        successes = sum(1 for r in per_agent if r.success)
        mean_q = (
            round(
                sum(r.questions_count for r in per_agent) / len(per_agent),
                2,
            )
            if per_agent
            else 0.0
        )
        call_limit = operator_call_limit_for_task(task.key)
        violation_runs = sum(1 for r in per_agent if r.limit_violated)
        return TaskAggregateResult(
            task_key=task.key,
            task_title=task.title,
            runs=len(per_agent),
            successes=successes,
            sr=(successes / len(per_agent)) if per_agent else 0.0,
            mean_q=mean_q,
            per_agent=per_agent,
            operator_call_limit=call_limit,
            violation_runs=violation_runs,
        )

    async def _run_agent_worker(
        self,
        *,
        agent_id: int,
        task: TaskDefinition,
        model: str,
        mode: str,
        megaprompt_config_name: str,
        max_ticks: int,
        knowledge_json: Path,
        knowledge_txt: Path,
        knowledge_source: str,
        world_mode: str | None,
        progress_offset: int,
        total_units: int,
        runtime_overrides: BenchmarkRuntimeOverrides | None,
    ) -> AgentEpisodeResult:
        session = create_isolated_session()
        self._register_session(session)
        try:
            session.set_texture_theme(str(world_mode or "craftax"))
            session.set_active_agent_model(model)
            session.set_active_agent_mode(mode)
            session.set_megaprompt_config_name(megaprompt_config_name)
            if is_deployment_megaprompt(megaprompt_config_name):
                session.set_operator_call_limit(operator_call_limit_for_task(task.key))
            else:
                session.clear_operator_call_limit()
            self._apply_runtime_overrides(session, runtime_overrides)
            _prepare_test_agent_knowledge(
                knowledge_source=knowledge_source,
                world_mode=world_mode,
                model=model,
                agent_json=knowledge_json,
                agent_txt=knowledge_txt,
            )
            with use_knowledge_paths(json_path=knowledge_json, txt_path=knowledge_txt):
                result = await self._run_single_episode(
                    task=task,
                    session=session,
                    max_ticks=max_ticks,
                    agent_id=agent_id,
                    knowledge_file=knowledge_json.name,
                )
                self._state["agents_live"] = [x for x in self._state["agents_live"] if x.get("agent_id") != agent_id]
                self._state["agents_live"].append(
                    {
                        "agent_id": agent_id,
                        "task_key": task.key,
                        "success": result.success,
                        "questions_count": result.questions_count,
                        "achieved_task_keys": result.achieved_task_keys,
                        "knowledge_file": result.knowledge_file,
                        "error": result.error,
                    }
                )
                self._state["knowledge_file"] = knowledge_json.name
                self._state["knowledge_preview"] = load_knowledge()
                completed_agents = len(self._state["agents_live"])
                self._set_progress(
                    progress_offset + completed_agents,
                    total_units,
                    f"{task.title}: {completed_agents}/{self._state['parallel_agents']} agents complete",
                )
                return result
        finally:
            self._unregister_session(session)

    def _apply_runtime_overrides(
        self,
        session: Any,
        overrides: BenchmarkRuntimeOverrides | None,
    ) -> None:
        if overrides is None:
            return
        if overrides.hf_token or overrides.openrouter_api_key:
            session.set_api_secrets(
                hf_token=overrides.hf_token or None,
                openrouter_api_key=overrides.openrouter_api_key or None,
            )
        session.set_interaction_mode(overrides.interaction_mode)
        session.set_allowed_experts(list(overrides.allowed_experts))
        session.set_forced_expert(overrides.forced_expert)
        if overrides.expert_modes:
            session.set_expert_modes(dict(overrides.expert_modes))
        if overrides.expert_models:
            session.set_expert_models(dict(overrides.expert_models))
        session.set_agent_sampling(
            do_sample=bool(overrides.active_agent_do_sample),
            temperature=overrides.active_agent_temperature,
            top_p=overrides.active_agent_top_p,
        )

    async def _run_single_episode(
        self,
        *,
        task: TaskDefinition,
        session: Any,
        max_ticks: int,
        agent_id: int,
        knowledge_file: str,
        baseline_task_index: int = 0,
        baseline_total_budget: int = 0,
        baseline_used_before: int = 0,
    ) -> AgentEpisodeResult:
        def _upsert_live_agent(
            *,
            tick: int,
            max_ticks_local: int,
            done: bool,
            success: bool | None = None,
            questions_count: int | None = None,
            achieved_task_keys: list[str] | None = None,
            error: str = "",
            session: Any | None = None,
        ) -> None:
            frame = self._frame_snapshot(session) if session is not None else None
            safe_max = max(1, int(max_ticks_local or max_ticks))
            safe_tick = max(0, min(int(tick), safe_max))
            progress_pct = int(round((100.0 * safe_tick) / safe_max))
            progress_rows = list(self._state.get("agents_progress") or [])
            progress_rows = [r for r in progress_rows if int(r.get("agent_id") or 0) != int(agent_id)]
            progress_rows.append(
                {
                    "agent_id": int(agent_id),
                    "tick": safe_tick,
                    "max_ticks": safe_max,
                    "progress_pct": progress_pct,
                    "done": bool(done),
                }
            )
            progress_rows.sort(key=lambda r: int(r.get("agent_id") or 0))
            self._state["agents_progress"] = progress_rows

            live_rows = list(self._state.get("agents_live") or [])
            live_rows = [r for r in live_rows if int(r.get("agent_id") or 0) != int(agent_id)]
            live_rows.append(
                {
                    "agent_id": int(agent_id),
                    "task_key": task.key,
                    "task_title": task.title,
                    "success": success,
                    "questions_count": int(questions_count or 0),
                    "achieved_task_keys": list(achieved_task_keys or []),
                    "knowledge_file": knowledge_file,
                    "error": str(error or ""),
                    "tick": safe_tick,
                    "max_ticks": safe_max,
                    "progress_pct": progress_pct,
                    "done": bool(done),
                    "frame": frame,
                }
            )
            live_rows.sort(key=lambda r: int(r.get("agent_id") or 0))
            self._state["agents_live"] = live_rows

        try:
            session.reset()
            initial_state = session.state
            questions = 0
            achieved: list[str] = []
            achieved_set: set[str] = set()
            call_limit = (
                operator_call_limit_for_task(task.key)
                if is_deployment_megaprompt(session.megaprompt_config_name)
                else 0
            )

            def _episode_result(
                *,
                success: bool,
                ticks_used: int,
                error: str = "",
            ) -> AgentEpisodeResult:
                return AgentEpisodeResult(
                    agent_id=agent_id,
                    task_key=task.key,
                    task_title=task.title,
                    success=success,
                    questions_count=questions,
                    achieved_task_keys=achieved,
                    knowledge_file=knowledge_file,
                    error=error,
                    ticks_used=ticks_used,
                    operator_call_limit=call_limit,
                    limit_violated=bool(
                        call_limit
                        and operator_call_limit_violated(
                            questions_count=questions,
                            limit=call_limit,
                        )
                    ),
                )

            for tick_idx in range(max_ticks):
                if self._state.get("stop_requested"):
                    break
                tick_no = tick_idx + 1
                if baseline_task_index > 0:
                    total_budget = int(max(1, baseline_total_budget or max_ticks))
                    global_tick = int(baseline_used_before) + tick_no
                    self._set_research_progress(
                        baseline_task_index,
                        min(global_tick, total_budget),
                        total_budget,
                    )
                else:
                    self._state["episode_tick"] = tick_no
                    self._state["episode_max_ticks"] = max_ticks
                _upsert_live_agent(
                    tick=tick_no,
                    max_ticks_local=max_ticks,
                    done=False,
                    questions_count=questions,
                    achieved_task_keys=achieved,
                    session=session,
                )
                await self._touch_state()
                async for payload in session.handle_agent_tick(steps=1, goal=task.goal):
                    if payload.get("type") == "agent_question_pending":
                        questions += 1
                    for candidate in campaign_tasks_for_world_mode(session._campaign_world_mode()):
                        if candidate.check(initial_state, session.state) and candidate.key not in achieved_set:
                            achieved_set.add(candidate.key)
                            achieved.append(candidate.key)
                    _upsert_live_agent(
                        tick=tick_no,
                        max_ticks_local=max_ticks,
                        done=False,
                        questions_count=questions,
                        achieved_task_keys=achieved,
                        session=session,
                    )
                    await self._touch_state()
                if task.check(initial_state, session.state):
                    _upsert_live_agent(
                        tick=tick_no,
                        max_ticks_local=max_ticks,
                        done=True,
                        success=True,
                        questions_count=questions,
                        achieved_task_keys=achieved,
                        session=session,
                    )
                    return _episode_result(success=True, ticks_used=tick_no)
            _upsert_live_agent(
                tick=max_ticks,
                max_ticks_local=max_ticks,
                done=True,
                success=False,
                questions_count=questions,
                achieved_task_keys=achieved,
                session=session,
            )
            return _episode_result(success=False, ticks_used=max_ticks)
        except Exception as e:
            _upsert_live_agent(
                tick=0,
                max_ticks_local=max_ticks,
                done=True,
                success=False,
                error=str(e),
            )
            return _episode_result(success=False, ticks_used=0, error=str(e))

    def _append_row(self, model: str, result: TaskAggregateResult) -> None:
        rows = list(self._state.get("rows") or [])
        rows.append(
            {
                "model": model,
                "task_key": result.task_key,
                "task_title": result.task_title,
                "sr": result.sr,
                "mean_q": result.mean_q,
                "mean_questions": result.mean_q,
                "runs": result.runs,
                "successes": result.successes,
                "operator_call_limit": result.operator_call_limit,
                "violation_runs": result.violation_runs,
                "limit_violation": int(result.violation_runs) > 0,
                "per_agent": [
                    {
                        "agent_id": r.agent_id,
                        "success": r.success,
                        "questions_count": r.questions_count,
                        "achieved_task_keys": r.achieved_task_keys,
                        "knowledge_file": r.knowledge_file,
                        "error": r.error,
                        "operator_call_limit": r.operator_call_limit,
                        "limit_violated": r.limit_violated,
                    }
                    for r in result.per_agent
                ],
            }
        )
        self._state["rows"] = rows
        append_companion_test_result(
            model=model,
            task_key=result.task_key,
            task_title=result.task_title,
            sr=result.sr,
            mean_q=result.mean_q,
            runs=result.runs,
            successes=result.successes,
            world_mode=str(self._state.get("world_mode") or ""),
            operator_call_limit=result.operator_call_limit,
            violation_runs=result.violation_runs,
        )
