import importlib.util
from pathlib import Path
import sys
import types

_SERVER_DIR = Path(__file__).resolve().parent
_DASHBOARD_PATH = _SERVER_DIR / "trajectory_dashboard.py"

_server_pkg = types.ModuleType("server")
_server_pkg.__path__ = [str(_SERVER_DIR)]
sys.modules.setdefault("server", _server_pkg)

_SPEC = importlib.util.spec_from_file_location(
    "server.trajectory_dashboard",
    _DASHBOARD_PATH,
    submodule_search_locations=[str(_SERVER_DIR)],
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load trajectory_dashboard module from {_DASHBOARD_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_process_trajectory_steps = _MODULE._process_trajectory_steps
_count_operator_questions = _MODULE._count_operator_questions
_is_exo_trajectory = _MODULE._is_exo_trajectory
_render_observation_image = _MODULE._render_observation_image
_ordered_campaign_goals = _MODULE._ordered_campaign_goals

from server.trajectory_logger import (  # noqa: E402
    build_trajectory_filename,
    display_name_from_filename,
    label_from_display_name,
    sanitize_filename_label,
)


def _run_stats(data, *, exo_mode=False):
    metrics, _action_counts = _run_stats_with_counts(data, exo_mode=exo_mode)
    return metrics


def _run_stats_with_counts(data, *, exo_mode=False):
    per_goal = {}
    action_counts = {}
    if exo_mode:
        valid_actions = _MODULE._exo_valid_actions()
    else:
        valid_actions = {"LEFT", "RIGHT", "UP", "DOWN", "DO"}
    metrics = _process_trajectory_steps(
        data,
        per_goal=per_goal,
        action_counts=action_counts,
        valid_actions=valid_actions,
        exo_mode=exo_mode,
    )
    return metrics, action_counts


def test_operator_questions_ignore_superseded_long_reasoning_step():
    goal = "Collect wood"
    data = [
        {
            "parsed": {"question": "This long reasoning was mis-parsed as a question."},
            "meta": {"step_index": 0, "goal": goal},
            "oracle_dialog": [],
        },
        {
            "parsed": {"action": "RIGHT"},
            "meta": {"step_index": 0, "goal": goal, "operator_retry_kind": "long_reasoning"},
            "oracle_dialog": [{"question": "System", "answer": "Please reply with an action only."}],
        },
        {
            "parsed": {"question": "Where is the nearest tree?"},
            "meta": {"step_index": 1, "goal": goal},
            "oracle_dialog": [],
        },
        {
            "parsed": {"action": "LEFT"},
            "meta": {"step_index": 2, "goal": goal},
            "oracle_dialog": [
                {"question": "Where is the nearest tree?", "answer": "Go east."},
            ],
        },
    ]

    actions, questions, db_records, *_rest = _run_stats(data)

    assert questions == 1
    assert actions == 2
    assert db_records == 0


def test_operator_questions_count_pending_unanswered_on_last_step():
    goal = "Place table"
    data = [
        {
            "parsed": {"question": "Should I place the table here?"},
            "meta": {"step_index": 0, "goal": goal},
            "oracle_dialog": [],
        },
    ]

    assert _count_operator_questions(data) == 1


def test_database_records_skip_superseded_step():
    goal = "Collect stone"
    data = [
        {
            "parsed": {"knowledge_updated": True, "to_database": "TYPE: note\nOP: UPSERT\n"},
            "meta": {"step_index": 0, "goal": goal},
            "oracle_dialog": [],
            "raw_answer": "<to_database>TYPE: note\nOP: UPSERT\n</to_database>",
        },
        {
            "parsed": {"action": "DOWN"},
            "meta": {"step_index": 0, "goal": goal, "operator_retry_kind": "invalid_format"},
            "oracle_dialog": [{"question": "System", "answer": "Invalid format."}],
        },
    ]

    _actions, _questions, db_records, db_updates, *_rest = _run_stats(data)

    assert db_records == 0
    assert db_updates == 0


def test_is_exo_trajectory_detects_mc3_in_agent_prompt():
    data = [
        {"agent_prompt": "You are Survey Unit MC-3 on an exo-planet.", "parsed": {}},
    ]
    assert _is_exo_trajectory(data) is True
    assert _is_exo_trajectory([{"agent_prompt": "Craftax agent", "parsed": {}}]) is False


def test_exo_action_distribution_uses_exo_labels():
    goal = "Collect biomass"
    data = [
        {
            "agent_prompt": "Survey Unit MC-3",
            "parsed": {"action": "DO", "action_raw": "EXTRACT"},
            "meta": {"step_index": 0, "goal": goal},
            "oracle_dialog": [],
        },
        {
            "agent_prompt": "Survey Unit MC-3",
            "parsed": {"action": "RIGHT", "action_raw": "RIGHT"},
            "meta": {"step_index": 1, "goal": goal},
            "oracle_dialog": [],
        },
    ]

    _metrics, action_counts = _run_stats_with_counts(data, exo_mode=True)

    assert action_counts.get("EXTRACT") == 1
    assert action_counts.get("RIGHT") == 1
    assert "DO" not in action_counts


def test_exo_render_uses_exo_texture_theme():
    env_state = {
        "map": [[2] * 9 for _ in range(7)],
        "player_position": [3, 4],
        "player_direction": 4,
    }
    craftax_image = _render_observation_image(env_state, texture_theme="craftax")
    exo_image = _render_observation_image(env_state, texture_theme="exo-planet")
    assert craftax_image.startswith("data:image/png;base64,")
    assert exo_image.startswith("data:image/png;base64,")
    assert craftax_image != exo_image


def test_ordered_campaign_goals_includes_exo_labels():
    seen = {"Collect Biomass", "Deploy Replicator", "Make Bone Drill"}
    ordered = _ordered_campaign_goals(seen)
    assert ordered == [
        "Collect Biomass",
        "Deploy Replicator",
        "Make Bone Drill",
    ]


def test_stats_for_exo_trajectory_populates_campaign_goals():
    goal = "Collect Biomass"
    data = [
        {
            "agent_prompt": "Survey Unit MC-3",
            "parsed": {"action": "RIGHT", "action_raw": "RIGHT"},
            "meta": {"step_index": 0, "goal": goal},
            "oracle_dialog": [],
        },
        {
            "agent_prompt": "Survey Unit MC-3",
            "parsed": {"question": "Where is biomass?"},
            "meta": {"step_index": 1, "goal": goal},
            "oracle_dialog": [],
        },
    ]
    per_goal = {}
    _process_trajectory_steps(
        data,
        per_goal=per_goal,
        action_counts={},
        valid_actions=_MODULE._exo_valid_actions(),
        exo_mode=True,
    )
    ordered = _ordered_campaign_goals(set(per_goal.keys()))
    assert "Collect Biomass" in ordered
    assert "Collect wood" not in ordered


def test_display_name_from_filename_includes_surname_name_and_date():
    assert (
        display_name_from_filename("trajectory_20260617_Smith_John.pkl")
        == "Smith John 2026-06-17"
    )
    assert (
        display_name_from_filename("trajectory_20260617_143052_Curie.pkl")
        == "Curie 2026-06-17"
    )
    assert (
        display_name_from_filename("trajectory_20260609_113437_Qwen-Big.pkl")
        == "Qwen-Big 2026-06-09"
    )


def test_build_trajectory_filename_uses_random_english_name_and_date():
    from datetime import datetime
    from unittest.mock import patch

    with patch(
        "server.trajectory_logger.generate_random_person_name",
        return_value=("Smith", "John"),
    ):
        filename = build_trajectory_filename(when=datetime(2026, 6, 17, 14, 30, 52))
    assert filename == "trajectory_20260617_Smith_John.pkl"
    assert display_name_from_filename(filename) == "Smith John 2026-06-17"


def test_label_from_display_name_strips_trailing_date():
    assert label_from_display_name("Smith John 2026-06-17", "20260617") == "Smith John"
    assert sanitize_filename_label("Smith John") == "Smith_John"


if __name__ == "__main__":
    test_operator_questions_ignore_superseded_long_reasoning_step()
    test_operator_questions_count_pending_unanswered_on_last_step()
    test_database_records_skip_superseded_step()
    test_is_exo_trajectory_detects_mc3_in_agent_prompt()
    test_exo_action_distribution_uses_exo_labels()
    test_exo_render_uses_exo_texture_theme()
    test_ordered_campaign_goals_includes_exo_labels()
    test_stats_for_exo_trajectory_populates_campaign_goals()
    test_display_name_from_filename_includes_surname_name_and_date()
    test_build_trajectory_filename_uses_random_english_name_and_date()
    test_label_from_display_name_strips_trailing_date()
    print("All trajectory dashboard tests passed.")
