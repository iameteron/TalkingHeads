from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from oracle.prompts.prompt_generation import is_exo_world_mode


WORLD_MODE_ARC = "arc_agi"
ARC_CAMPAIGN_LEVEL_COUNT = 7

CRAFTING_TABLE_BLOCK_ID = 11
FURNACE_BLOCK_ID = 12


CheckerFn = Callable[[Any, Any], bool]


@dataclass(frozen=True)
class TaskDefinition:
    key: str
    title: str
    goal: str
    check: CheckerFn


def _inventory_count(state: Any, attr: str) -> int:
    return int(getattr(getattr(state, "inventory", None), attr, 0) or 0)


def _inventory_increased(initial_state: Any, current_state: Any, attr: str) -> bool:
    return _inventory_count(current_state, attr) > _inventory_count(initial_state, attr)


def _count_crafting_tables(state: Any) -> int:
    game_map = getattr(state, "map", None)
    if game_map is None:
        return 0
    height, width = game_map.shape
    count = 0
    for x in range(height):
        for y in range(width):
            if int(game_map[x, y]) == CRAFTING_TABLE_BLOCK_ID:
                count += 1
    return count


def _count_furnaces(state: Any) -> int:
    game_map = getattr(state, "map", None)
    if game_map is None:
        return 0
    height, width = game_map.shape
    count = 0
    for x in range(height):
        for y in range(width):
            if int(game_map[x, y]) == FURNACE_BLOCK_ID:
                count += 1
    return count


def check_collect_wood(initial_state: Any, current_state: Any) -> bool:
    return _inventory_increased(initial_state, current_state, "wood")


def check_place_table(initial_state: Any, current_state: Any) -> bool:
    return _count_crafting_tables(current_state) > _count_crafting_tables(initial_state)


def check_make_wood_pickaxe(initial_state: Any, current_state: Any) -> bool:
    return _inventory_increased(initial_state, current_state, "wood_pickaxe")


def check_collect_stone(initial_state: Any, current_state: Any) -> bool:
    return _inventory_increased(initial_state, current_state, "stone")


def check_make_stone_pickaxe(initial_state: Any, current_state: Any) -> bool:
    return _inventory_increased(initial_state, current_state, "stone_pickaxe")


def check_collect_coal(initial_state: Any, current_state: Any) -> bool:
    return _inventory_increased(initial_state, current_state, "coal")


def check_collect_iron(initial_state: Any, current_state: Any) -> bool:
    return _inventory_increased(initial_state, current_state, "iron")


def check_make_furnace(initial_state: Any, current_state: Any) -> bool:
    return _count_furnaces(current_state) > _count_furnaces(initial_state)


def check_make_iron_pickaxe(initial_state: Any, current_state: Any) -> bool:
    return _inventory_increased(initial_state, current_state, "iron_pickaxe")


def check_collect_diamond(initial_state: Any, current_state: Any) -> bool:
    return _inventory_increased(initial_state, current_state, "diamond")


CRAFTAX_CAMPAIGN_TASKS: list[TaskDefinition] = [
    TaskDefinition(
        key="collect_wood",
        title="Collect wood",
        goal="Collect wood",
        check=check_collect_wood,
    ),
    TaskDefinition(
        key="place_table",
        title="Place table",
        goal="Place table",
        check=check_place_table,
    ),
    TaskDefinition(
        key="make_wood_pickaxe",
        title="Make wood pickaxe",
        goal="Make wood pickaxe",
        check=check_make_wood_pickaxe,
    ),
    TaskDefinition(
        key="collect_stone",
        title="Collect stone",
        goal="Collect stone",
        check=check_collect_stone,
    ),
    TaskDefinition(
        key="make_stone_pickaxe",
        title="Make stone pickaxe",
        goal="Make stone pickaxe",
        check=check_make_stone_pickaxe,
    ),
    TaskDefinition(
        key="collect_coal",
        title="Collect coal",
        goal="Collect coal",
        check=check_collect_coal,
    ),
    TaskDefinition(
        key="collect_iron",
        title="Collect iron",
        goal="Collect iron",
        check=check_collect_iron,
    ),
    TaskDefinition(
        key="make_furnace",
        title="Make furnace",
        goal="Make furnace",
        check=check_make_furnace,
    ),
    TaskDefinition(
        key="make_iron_pickaxe",
        title="Make iron pickaxe",
        goal="Make iron pickaxe",
        check=check_make_iron_pickaxe,
    ),
    TaskDefinition(
        key="collect_diamond",
        title="Collect diamond",
        goal="Collect diamond",
        check=check_collect_diamond,
    ),
]

# Backward-compatible alias used by older imports.
CAMPAIGN_TASKS = CRAFTAX_CAMPAIGN_TASKS

_EXO_CAMPAIGN_LABELS: dict[str, tuple[str, str]] = {
    "collect_wood": ("Collect Biomass", "Collect Biomass"),
    "place_table": ("Deploy Replicator", "Deploy Replicator"),
    "make_wood_pickaxe": ("Make Bone Drill", "Make Bone Drill"),
    "collect_stone": ("Collect Basalt Shard", "Collect Basalt Shard"),
    "make_stone_pickaxe": ("Make Rock Drill", "Make Rock Drill"),
    "collect_coal": ("Collect Energy Ore", "Collect Energy Ore"),
    "collect_iron": ("Collect Titanite Ore", "Collect Titanite Ore"),
    "make_furnace": ("Deploy Thermal Oven", "Deploy Thermal Oven"),
    "make_iron_pickaxe": ("Make Titan Drill", "Make Titan Drill"),
    "collect_diamond": ("Collect Core Ore", "Collect Core Ore"),
}

EXO_CAMPAIGN_TASKS: list[TaskDefinition] = [
    TaskDefinition(
        key=task.key,
        title=_EXO_CAMPAIGN_LABELS[task.key][0],
        goal=_EXO_CAMPAIGN_LABELS[task.key][1],
        check=task.check,
    )
    for task in CRAFTAX_CAMPAIGN_TASKS
]


def normalize_campaign_world_mode(world_mode: str | None) -> str:
    token = str(world_mode or "craftax").strip().lower()
    if token in {"exo", "exo-planet", "exo_planet"}:
        return "exo-planet"
    if token in {"arc", "arc_agi", "arc-agi", "arc-agi-3", "arc_agi_3"}:
        return WORLD_MODE_ARC
    return "craftax"


def is_arc_world_mode(world_mode: str | None) -> bool:
    return normalize_campaign_world_mode(world_mode) == WORLD_MODE_ARC


def arc_levels_completed(state: Any) -> int:
    if state is None:
        return 0
    if hasattr(state, "levels_completed"):
        try:
            return max(0, int(getattr(state, "levels_completed") or 0))
        except (TypeError, ValueError):
            return 0
    if isinstance(state, dict):
        try:
            return max(0, int(state.get("levels_completed") or 0))
        except (TypeError, ValueError):
            return 0
    return 0


def _make_arc_level_check(required_levels: int) -> CheckerFn:
    def check(_initial_state: Any, current_state: Any) -> bool:
        return arc_levels_completed(current_state) >= required_levels

    return check


def _build_arc_campaign_tasks(level_count: int = ARC_CAMPAIGN_LEVEL_COUNT) -> list[TaskDefinition]:
    tasks: list[TaskDefinition] = []
    for level in range(1, max(1, int(level_count or 1)) + 1):
        tasks.append(
            TaskDefinition(
                key=f"level_{level}",
                title=f"Level-{level}",
                goal=f"Complete Level-{level}",
                check=_make_arc_level_check(level),
            )
        )
    return tasks


ARC_CAMPAIGN_TASKS: list[TaskDefinition] = _build_arc_campaign_tasks()


def campaign_tasks_for_world_mode(world_mode: str | None) -> list[TaskDefinition]:
    if is_arc_world_mode(world_mode):
        return ARC_CAMPAIGN_TASKS
    if is_exo_world_mode(world_mode):
        return EXO_CAMPAIGN_TASKS
    return CRAFTAX_CAMPAIGN_TASKS


def campaign_task_level_index(task_key: str, world_mode: str | None = None) -> int:
    key = str(task_key or "").strip()
    if not key:
        return 0
    for idx, task in enumerate(campaign_tasks_for_world_mode(world_mode), 1):
        if task.key == key:
            return idx
    return 0


def campaign_task_level_weight(task_key: str, world_mode: str | None = None) -> int:
    return max(1, campaign_task_level_index(task_key, world_mode))


def campaign_total_levels(world_mode: str | None = None) -> int:
    return len(campaign_tasks_for_world_mode(world_mode))


def detect_new_episode_achievements(
    *,
    world_mode: str | None,
    episode_initial_state: Any,
    current_state: Any,
    already_discovered: set[str],
) -> list[TaskDefinition]:
    """Return campaign tasks newly satisfied since episode start (first time only)."""
    newly: list[TaskDefinition] = []
    for task in campaign_tasks_for_world_mode(world_mode):
        if task.key in already_discovered:
            continue
        if task.check(episode_initial_state, current_state):
            already_discovered.add(task.key)
            newly.append(task)
    return newly


@dataclass
class CampaignState:
    enabled: bool = False
    current_index: int = 0
    completed_keys: list[str] = field(default_factory=list)
    current_task_initial_state: Any = None
    phase2_selected_key: str | None = None
    phase2_initial_state: Any = None
    phase2_completed_keys: list[str] = field(default_factory=list)
    world_mode: str = "exo-planet"

    def _tasks(self) -> list[TaskDefinition]:
        return campaign_tasks_for_world_mode(self.world_mode)

    def set_world_mode(self, world_mode: str | None) -> None:
        self.world_mode = normalize_campaign_world_mode(world_mode)

    def reset_progress(self, initial_state: Any) -> None:
        self.current_index = 0
        self.completed_keys = []
        self.current_task_initial_state = initial_state
        self.phase2_selected_key = None
        self.phase2_initial_state = None
        self.phase2_completed_keys = []

    def set_enabled(self, enabled: bool, initial_state: Any) -> None:
        self.enabled = bool(enabled)
        self.reset_progress(initial_state)

    def current_task(self) -> TaskDefinition | None:
        tasks = self._tasks()
        if self.current_index >= len(tasks):
            return None
        return tasks[self.current_index]

    def task_by_key(self, key: str) -> TaskDefinition | None:
        for task in self._tasks():
            if task.key == key:
                return task
        return None

    def is_finished(self) -> bool:
        return self.current_index >= len(self._tasks())

    def phase2_active(self) -> bool:
        return self.phase2_selected_key is not None

    def phase2_selected_task(self) -> TaskDefinition | None:
        if not self.phase2_selected_key:
            return None
        return self.task_by_key(self.phase2_selected_key)

    def maybe_advance(self, current_state: Any) -> bool:
        if not self.enabled:
            return False
        if self.phase2_active():
            return self.maybe_advance_phase2(current_state)
        changed = False
        while not self.is_finished():
            task = self.current_task()
            if task is None:
                break
            if not task.check(self.current_task_initial_state, current_state):
                break
            self.completed_keys.append(task.key)
            self.current_index += 1
            self.current_task_initial_state = current_state
            changed = True
        return changed

    def start_phase2(self, level_key: str, initial_state: Any) -> None:
        if not self.enabled:
            raise ValueError("Campaign mode is disabled")
        task = self.task_by_key(level_key)
        if task is None:
            raise ValueError(f"Unknown campaign level key: {level_key}")
        self.phase2_selected_key = task.key
        self.phase2_initial_state = initial_state

    def maybe_advance_phase2(self, current_state: Any) -> bool:
        if not self.enabled or not self.phase2_active():
            return False
        task = self.phase2_selected_task()
        if task is None:
            self.phase2_selected_key = None
            self.phase2_initial_state = None
            return False
        if self.phase2_initial_state is None:
            self.phase2_initial_state = current_state
            return False
        if not task.check(self.phase2_initial_state, current_state):
            return False
        if task.key not in self.phase2_completed_keys:
            self.phase2_completed_keys.append(task.key)
        self.phase2_selected_key = None
        self.phase2_initial_state = None
        return True

    def snapshot(self) -> dict[str, Any]:
        tasks = self._tasks()
        task = self.current_task()
        phase2_task = self.phase2_selected_task()
        return {
            "enabled": self.enabled,
            "world_mode": self.world_mode,
            "current_task_key": task.key if task else None,
            "current_task_title": task.title if task else None,
            "current_task_goal": task.goal if task else None,
            "completed_count": len(self.completed_keys),
            "total_count": len(tasks),
            "completed_keys": list(self.completed_keys),
            "is_finished": self.is_finished(),
            "tasks": [
                {"key": t.key, "title": t.title, "goal": t.goal}
                for t in tasks
            ],
            "phase": "phase2" if self.phase2_active() else "phase1",
            "phase1": {
                "current_task_key": task.key if task else None,
                "current_task_title": task.title if task else None,
                "current_task_goal": task.goal if task else None,
                "completed_count": len(self.completed_keys),
                "total_count": len(tasks),
                "completed_keys": list(self.completed_keys),
                "is_finished": self.is_finished(),
            },
            "phase2": {
                "active": self.phase2_active(),
                "selected_task_key": phase2_task.key if phase2_task else None,
                "selected_task_title": phase2_task.title if phase2_task else None,
                "selected_task_goal": phase2_task.goal if phase2_task else None,
                "completed_keys": list(self.phase2_completed_keys),
            },
        }
