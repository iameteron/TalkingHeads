import importlib.util
import sys
import types
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
_server_pkg = types.ModuleType("server")
_server_pkg.__path__ = [str(_SERVER_DIR)]
sys.modules.setdefault("server", _server_pkg)

_SPEC = importlib.util.spec_from_file_location(
    "server.model_names",
    _SERVER_DIR / "model_names.py",
)
_NAMES_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _NAMES_MOD
_SPEC.loader.exec_module(_NAMES_MOD)

_SPEC = importlib.util.spec_from_file_location(
    "server.campaign_benchmark",
    _SERVER_DIR / "campaign_benchmark.py",
    submodule_search_locations=[str(_SERVER_DIR)],
)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

short_model_name = _NAMES_MOD.short_model_name
canonical_model_key = _NAMES_MOD.canonical_model_key
parse_since_timestamp = _MOD.parse_since_timestamp
filter_entries_since = _MOD.filter_entries_since
get_campaign_benchmark = _MOD.get_campaign_benchmark
_entry_world_mode = _MOD._entry_world_mode


def test_short_model_name_strips_family_and_suffix():
    assert short_model_name("Qwen/Qwen3-4B-Instruct-2507:nscale") == "Qwen3-4B-Instruct-2507"
    assert short_model_name("qwen/qwen3-235b-a22b-2507") == "qwen3-235b-a22b-2507"
    assert canonical_model_key("Qwen/Qwen3-4B-Instruct-2507:novita") == "qwen3-4b-instruct-2507"
    assert canonical_model_key("qwen/qwen3-4b-instruct-2507:nscale") == "qwen3-4b-instruct-2507"


def test_entry_world_mode_detects_exo_from_prompt():
    assert _entry_world_mode({"last_prompt_excerpt": "Survey Unit MC-3"}) == "exo-planet"
    assert _entry_world_mode({"last_goal": "Collect Biomass"}) == "exo-planet"
    assert _entry_world_mode({"last_goal": "Collect wood"}) == "craftax"


def test_campaign_benchmark_payload_shape():
    payload = get_campaign_benchmark()
    assert payload["ok"] is True
    assert "since" in payload
    assert payload["since"] is None
    for mode in ("craftax", "exo-planet"):
        block = payload["world_modes"][mode]
        assert "compact" in block
        assert "extended" in block
        assert block["total_tasks"] == 10
        if block["compact"]:
            row = block["compact"][0]
            assert "model_short" in row
            assert "solved_display" in row
            assert "max_item_icon" in row
        if block.get("deployment_tasks"):
            task = block["deployment_tasks"][0]
            assert "icon" in task
            assert task["icon"].endswith(f"{task['key']}.png")


def test_parse_since_timestamp():
    dt = parse_since_timestamp("2026-06-16")
    assert dt is not None
    assert dt.year == 2026 and dt.month == 6 and dt.day == 16


def test_filter_entries_since():
    entries = [
        {"finished_at": "2026-06-15T12:00:00", "active_agent_model": "a"},
        {"finished_at": "2026-06-16T00:00:00", "active_agent_model": "b"},
        {"finished_at": "2026-06-17T08:00:00", "active_agent_model": "c"},
    ]
    since = parse_since_timestamp("2026-06-16")
    filtered = filter_entries_since(entries, since)
    assert len(filtered) == 2
    assert filtered[0]["active_agent_model"] == "b"


def test_campaign_benchmark_since_param():
    payload = get_campaign_benchmark(since="2099-01-01")
    assert payload["since"] is not None
    for mode in ("craftax", "exo-planet"):
        assert payload["world_modes"][mode]["exploration_runs"] == 0


def test_max_task_key_from_entry_uses_last_goal_for_phase2_progress():
    _max_task_key_from_entry = _MOD._max_task_key_from_entry
    entry = {
        "phase1_completed_levels": 0,
        "phase2_highest_level": 0,
        "phase2_questions": 5,
        "last_goal": "Collect iron",
    }
    assert _max_task_key_from_entry(entry, world_mode="craftax") == "collect_iron"
    exo_entry = {**entry, "last_goal": "Collect Titanite Ore"}
    assert _max_task_key_from_entry(exo_entry, world_mode="exo-planet") == "collect_iron"
    mixed_entry = {
        "phase1_completed_levels": 0,
        "phase2_highest_level": 0,
        "phase2_questions": 5,
        "last_goal": "Collect iron",
    }
    assert _max_task_key_from_entry(mixed_entry, world_mode="exo-planet") == "collect_iron"


def test_max_task_key_from_entry_prefers_completed_keys():
    _max_task_key_from_entry = _MOD._max_task_key_from_entry
    entry = {
        "phase1_completed_levels": 0,
        "phase2_highest_level": 0,
        "phase2_completed_keys": ["collect_coal"],
        "last_goal": "Collect wood",
    }
    assert _max_task_key_from_entry(entry, world_mode="craftax") == "collect_coal"


if __name__ == "__main__":
    test_short_model_name_strips_family_and_suffix()
    test_entry_world_mode_detects_exo_from_prompt()
    test_campaign_benchmark_payload_shape()
    test_parse_since_timestamp()
    test_filter_entries_since()
    test_campaign_benchmark_since_param()
    test_max_task_key_from_entry_uses_last_goal_for_phase2_progress()
    test_max_task_key_from_entry_prefers_completed_keys()
    print("Campaign benchmark tests passed.")
