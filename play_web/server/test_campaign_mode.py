from types import SimpleNamespace
import importlib.util
from pathlib import Path
import sys

import numpy as np

_CAMPAIGN_MODE_PATH = Path(__file__).resolve().parent / "campaign_mode.py"
_SPEC = importlib.util.spec_from_file_location("campaign_mode_local", _CAMPAIGN_MODE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load campaign_mode module from {_CAMPAIGN_MODE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
CampaignState = _MODULE.CampaignState


def _make_state(
    *,
    wood: int = 0,
    stone: int = 0,
    coal: int = 0,
    iron: int = 0,
    wood_pickaxe: int = 0,
    stone_pickaxe: int = 0,
    iron_pickaxe: int = 0,
    diamond: int = 0,
    table_on_map: bool = False,
    furnace_on_map: bool = False,
):
    block = 12 if furnace_on_map else (11 if table_on_map else 0)
    return SimpleNamespace(
        inventory=SimpleNamespace(
            wood=wood,
            stone=stone,
            coal=coal,
            iron=iron,
            wood_pickaxe=wood_pickaxe,
            stone_pickaxe=stone_pickaxe,
            iron_pickaxe=iron_pickaxe,
            diamond=diamond,
        ),
        map=np.array([[block]]),
    )


def test_campaign_progression_happy_path():
    state = CampaignState()
    state.set_enabled(True, _make_state())

    assert state.snapshot()["current_task_key"] == "collect_wood"

    state.maybe_advance(_make_state(wood=1))
    assert state.snapshot()["current_task_key"] == "place_table"

    state.maybe_advance(_make_state(wood=1, table_on_map=True))
    assert state.snapshot()["current_task_key"] == "make_wood_pickaxe"

    state.maybe_advance(_make_state(wood_pickaxe=1, table_on_map=True))
    assert state.snapshot()["current_task_key"] == "collect_stone"

    state.maybe_advance(_make_state(stone=1, wood_pickaxe=1, table_on_map=True))
    assert state.snapshot()["current_task_key"] == "make_stone_pickaxe"

    state.maybe_advance(_make_state(stone_pickaxe=1, stone=1, wood_pickaxe=1, table_on_map=True))
    assert state.snapshot()["current_task_key"] == "collect_coal"

    state.maybe_advance(_make_state(stone_pickaxe=1, coal=1, stone=1, wood_pickaxe=1, table_on_map=True))
    assert state.snapshot()["current_task_key"] == "collect_iron"

    state.maybe_advance(
        _make_state(stone_pickaxe=1, coal=1, iron=1, stone=1, wood_pickaxe=1, table_on_map=True)
    )
    assert state.snapshot()["current_task_key"] == "make_furnace"

    state.maybe_advance(
        _make_state(
            stone_pickaxe=1,
            coal=1,
            iron=1,
            stone=1,
            wood_pickaxe=1,
            table_on_map=True,
            furnace_on_map=True,
        )
    )
    assert state.snapshot()["current_task_key"] == "make_iron_pickaxe"

    state.maybe_advance(
        _make_state(
            stone_pickaxe=1,
            coal=1,
            iron=1,
            stone=1,
            wood_pickaxe=1,
            iron_pickaxe=1,
            table_on_map=True,
            furnace_on_map=True,
        )
    )
    assert state.snapshot()["current_task_key"] == "collect_diamond"


def test_campaign_progression_finishes_after_diamond():
    state = CampaignState()
    state.set_enabled(True, _make_state())

    state.maybe_advance(
        _make_state(
            wood=1,
            stone=1,
            coal=1,
            iron=1,
            wood_pickaxe=1,
            stone_pickaxe=1,
            iron_pickaxe=1,
            table_on_map=True,
            furnace_on_map=True,
        )
    )
    snapshot = state.snapshot()
    assert snapshot["is_finished"] is True
    assert snapshot["completed_count"] == snapshot["total_count"]
    assert snapshot["current_task_key"] is None


def test_make_stone_pickaxe_not_completed_without_new_pickaxe():
    state = CampaignState()
    state.set_enabled(True, _make_state())
    state.maybe_advance(_make_state(wood=1, table_on_map=True, wood_pickaxe=1, stone=1))
    assert state.snapshot()["current_task_key"] == "make_stone_pickaxe"

    # Already had a stone pickaxe before this task started — should not auto-complete.
    state.maybe_advance(
        _make_state(wood=1, table_on_map=True, wood_pickaxe=1, stone=1, stone_pickaxe=1)
    )
    assert state.snapshot()["current_task_key"] == "make_stone_pickaxe"

    # Pickaxe acquired during this task.
    state.maybe_advance(
        _make_state(wood=1, table_on_map=True, wood_pickaxe=1, stone=1, stone_pickaxe=2)
    )
    assert state.snapshot()["current_task_key"] == "collect_coal"


def test_campaign_reset_progress_keeps_enabled_flag():
    state = CampaignState()
    state.set_enabled(True, _make_state())
    state.maybe_advance(_make_state(wood=1))
    assert state.snapshot()["completed_count"] == 1

    state.reset_progress(_make_state())
    snapshot = state.snapshot()
    assert snapshot["enabled"] is True
    assert snapshot["completed_count"] == 0
    assert snapshot["current_task_key"] == "collect_wood"


def test_phase2_level_can_be_started_and_completed():
    state = CampaignState()
    state.set_enabled(True, _make_state())
    state.start_phase2("collect_stone", _make_state())

    snap = state.snapshot()
    assert snap["phase"] == "phase2"
    assert snap["phase2"]["selected_task_key"] == "collect_stone"

    state.maybe_advance(_make_state(stone=1))
    snap_done = state.snapshot()
    assert snap_done["phase2"]["active"] is False
    assert "collect_stone" in snap_done["phase2"]["completed_keys"]


def test_campaign_exo_tasks_use_exo_vocabulary():
    state = CampaignState()
    state.set_world_mode("exo-planet")
    state.set_enabled(True, _make_state())
    snapshot = state.snapshot()
    assert snapshot["current_task_goal"] == "Collect Biomass"
    assert snapshot["tasks"][1]["title"] == "Deploy Replicator"
    assert snapshot["tasks"][0]["key"] == "collect_wood"


def test_campaign_world_mode_switch_updates_goals():
    state = CampaignState()
    state.set_enabled(True, _make_state())
    assert state.snapshot()["current_task_goal"] == "Collect wood"
    state.set_world_mode("exo-planet")
    assert state.snapshot()["current_task_goal"] == "Collect Biomass"
    state.set_world_mode("craftax")
    assert state.snapshot()["current_task_goal"] == "Collect wood"


def test_phase2_level_does_not_progress_phase1():
    state = CampaignState()
    state.set_enabled(True, _make_state())
    state.start_phase2("collect_wood", _make_state())

    state.maybe_advance(_make_state(wood=1))
    snap = state.snapshot()
    assert snap["phase2"]["active"] is False
    assert snap["completed_count"] == 0


def test_detect_new_episode_achievements_first_time_only():
    detect = _MODULE.detect_new_episode_achievements
    initial = _make_state()
    discovered: set[str] = set()

    first = detect(
        world_mode="craftax",
        episode_initial_state=initial,
        current_state=_make_state(wood=1),
        already_discovered=discovered,
    )
    assert [task.key for task in first] == ["collect_wood"]
    assert "collect_wood" in discovered

    second = detect(
        world_mode="craftax",
        episode_initial_state=initial,
        current_state=_make_state(wood=2, table_on_map=True),
        already_discovered=discovered,
    )
    assert [task.key for task in second] == ["place_table"]

    repeat = detect(
        world_mode="craftax",
        episode_initial_state=initial,
        current_state=_make_state(wood=3),
        already_discovered=discovered,
    )
    assert repeat == []


def test_detect_new_episode_achievements_exo_titles():
    detect = _MODULE.detect_new_episode_achievements
    initial = _make_state()
    discovered: set[str] = set()
    tasks = detect(
        world_mode="exo-planet",
        episode_initial_state=initial,
        current_state=_make_state(wood=1),
        already_discovered=discovered,
    )
    assert len(tasks) == 1
    assert tasks[0].key == "collect_wood"
    assert tasks[0].title == "Collect Biomass"


def test_arc_campaign_progression_by_levels_completed():
    state = CampaignState()
    state.set_world_mode("arc_agi")
    state.set_enabled(True, SimpleNamespace(levels_completed=0))

    assert state.snapshot()["current_task_key"] == "level_1"
    assert state.snapshot()["total_count"] == _MODULE.ARC_CAMPAIGN_LEVEL_COUNT

    state.maybe_advance(SimpleNamespace(levels_completed=1))
    assert state.snapshot()["completed_count"] == 1
    assert state.snapshot()["current_task_key"] == "level_2"

    state.maybe_advance(SimpleNamespace(levels_completed=2))
    assert state.snapshot()["completed_count"] == 2
