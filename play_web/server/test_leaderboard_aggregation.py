import importlib.util
import sys
import types
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
_server_pkg = types.ModuleType("server")
_server_pkg.__path__ = [str(_SERVER_DIR)]
sys.modules.setdefault("server", _server_pkg)

for module_name, filename in (
    ("server.model_names", "model_names.py"),
    ("server.leaderboard", "leaderboard.py"),
):
    spec = importlib.util.spec_from_file_location(module_name, _SERVER_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

leaderboard = sys.modules["server.leaderboard"]


def test_aggregate_model_leaderboard_merges_provider_variants():
    entries = [
        {
            "active_agent_model": "Qwen/Qwen3-4B-Instruct-2507:nscale",
            "phase1_completed_levels": 3,
            "phase1_questions": 4,
            "phase2_highest_level": 0,
            "phase2_questions": 0,
            "finished_at": "2026-06-01T10:00:00",
        },
        {
            "active_agent_model": "qwen/qwen3-4b-instruct-2507:novita",
            "phase1_completed_levels": 5,
            "phase1_questions": 2,
            "phase2_highest_level": 0,
            "phase2_questions": 0,
            "finished_at": "2026-06-02T10:00:00",
        },
    ]
    rows = leaderboard.aggregate_model_leaderboard(entries)
    assert len(rows) == 1
    assert rows[0]["model_short"].lower() == "qwen3-4b-instruct-2507"
    assert rows[0]["attempts"] == 2
    assert rows[0]["best_phase1_completed_levels"] == 5


def test_build_companion_leaderboard_rows_dedupes_by_canonical_model():
    test_entries = [
        {
            "finished_at": "2026-06-01T10:00:00",
            "model": "Qwen/Qwen3-4B-Instruct-2507:nscale",
            "task_key": "collect_stone",
            "task_title": "Collect stone",
            "sr": 0.0,
            "mean_q": 2.0,
            "runs": 3,
        },
        {
            "finished_at": "2026-06-02T10:00:00",
            "model": "qwen/qwen3-4b-instruct-2507:novita",
            "task_key": "collect_stone",
            "task_title": "Collect stone",
            "sr": 0.5,
            "mean_q": 1.0,
            "runs": 4,
        },
    ]
    rows = leaderboard.build_companion_leaderboard_rows(test_entries, {})
    assert len(rows) == 1
    assert rows[0]["model_short"].lower() == "qwen3-4b-instruct-2507"
    assert rows[0]["sr"] == 0.5
    assert rows[0]["runs"] == 4


def test_campaign_human_row_score():
    row = leaderboard._campaign_human_row(
        {
            "finished_at": "2026-06-03T10:00:00",
            "player_name": "Zoya",
            "player_avatar_id": 4,
            "world_mode": "craftax",
            "phase1_completed_levels": 2,
            "phase2_highest_level": 0,
            "phase1_questions": 1,
            "phase2_questions": 0,
            "level_steps": {"collect_wood": 12, "place_table": 8},
            "finish_reason": "level_complete",
        }
    )
    assert row["leaderboard_type"] == "campaign"
    # L1: 1000 - 12*6 = 928; L2: 2000 - 8*6 = 1952; questions: 15
    assert row["score"] == 2865
    assert row["agent_steps"] == 20
    assert row["total_levels"] == 10
    assert row["player_avatar_id"] == 4


def test_campaign_human_row_legacy_score_without_level_steps():
    row = leaderboard._campaign_human_row(
        {
            "finished_at": "2026-06-03T10:00:00",
            "player_name": "Zoya",
            "world_mode": "craftax",
            "phase1_completed_levels": 2,
            "phase2_highest_level": 1,
            "phase1_questions": 1,
            "phase2_questions": 0,
        }
    )
    assert row["score"] == 2490


def test_aggregate_best_human_rows_keeps_best_score_per_player():
    rows = leaderboard._aggregate_best_human_rows(
        [
            {
                "player_name": "Zoya",
                "world_mode": "craftax",
                "leaderboard_type": "campaign",
                "score": 1200,
                "finished_at": "2026-06-01T10:00:00",
            },
            {
                "player_name": "Zoya",
                "world_mode": "craftax",
                "leaderboard_type": "campaign",
                "score": 2400,
                "finished_at": "2026-06-02T10:00:00",
            },
            {
                "player_name": "Alex",
                "world_mode": "exo-planet",
                "leaderboard_type": "campaign",
                "score": 900,
                "finished_at": "2026-06-02T10:00:00",
            },
        ]
    )
    assert len(rows) == 2
    zoya = next(row for row in rows if row["player_name"] == "Zoya")
    assert zoya["score"] == 2400


def test_aggregate_best_human_rows_keeps_arc_games_separate():
    rows = leaderboard._aggregate_best_human_rows(
        [
            {
                "player_name": "Zoya",
                "leaderboard_type": "arc",
                "game_id": "ls20",
                "score": 1200,
                "actions": 12,
                "finished_at": "2026-06-01T10:00:00",
            },
            {
                "player_name": "Zoya",
                "leaderboard_type": "arc",
                "game_id": "lp85",
                "score": 900,
                "actions": 10,
                "finished_at": "2026-06-01T11:00:00",
            },
            {
                "player_name": "Zoya",
                "leaderboard_type": "arc",
                "game_id": "ls20",
                "score": 1400,
                "actions": 9,
                "finished_at": "2026-06-01T12:00:00",
            },
        ]
    )
    assert len(rows) == 2
    by_game = {row["game_id"]: row for row in rows}
    assert by_game["ls20"]["score"] == 1400
    assert by_game["lp85"]["score"] == 900


def test_oracle_campaign_row_maps_to_talkingheads():
    row = leaderboard._campaign_leaderboard_row(
        {
            "finished_at": "2026-06-03T10:00:00",
            "player_name": "Babycar",
            "player_avatar_id": 4,
            "world_mode": "craftax",
            "interaction_mode": "oracle",
            "phase1_completed_levels": 2,
            "phase2_highest_level": 0,
            "phase1_questions": 1,
            "level_steps": {"collect_wood": 3, "place_table": 4},
            "active_agent_model": "qwen/qwen3-next-80b-a3b-instruct",
        }
    )
    assert row["player_name"] == leaderboard.TALKINGHEADS_PLAYER_NAME
    assert row["is_ai_operator"] is True
    assert row["interaction_mode"] == "oracle"


def test_oracle_runs_aggregate_best_across_models():
    rows = leaderboard._aggregate_best_human_rows(
        [
            leaderboard._campaign_leaderboard_row(
                {
                    "finished_at": "2026-06-01T10:00:00",
                    "player_name": "Babycar",
                    "world_mode": "craftax",
                    "interaction_mode": "oracle",
                    "phase1_completed_levels": 2,
                    "phase1_questions": 0,
                    "level_steps": {"collect_wood": 10, "place_table": 10},
                    "active_agent_model": "model-a",
                }
            ),
            leaderboard._campaign_leaderboard_row(
                {
                    "finished_at": "2026-06-02T10:00:00",
                    "player_name": "Babycar",
                    "world_mode": "craftax",
                    "interaction_mode": "oracle",
                    "phase1_completed_levels": 3,
                    "phase1_questions": 0,
                    "level_steps": {"collect_wood": 1, "place_table": 1, "make_wood_pickaxe": 1},
                    "active_agent_model": "model-b",
                }
            ),
        ]
    )
    assert len(rows) == 1
    assert rows[0]["player_name"] == leaderboard.TALKINGHEADS_PLAYER_NAME
    assert rows[0]["phase1_completed_levels"] == 3
    assert rows[0]["active_agent_model"] == "model-b"


if __name__ == "__main__":
    test_aggregate_model_leaderboard_merges_provider_variants()
    test_build_companion_leaderboard_rows_dedupes_by_canonical_model()
    test_campaign_human_row_score()
    test_campaign_human_row_legacy_score_without_level_steps()
    test_aggregate_best_human_rows_keeps_best_score_per_player()
    test_aggregate_best_human_rows_keeps_arc_games_separate()
    test_oracle_campaign_row_maps_to_talkingheads()
    test_oracle_runs_aggregate_best_across_models()
    print("Leaderboard aggregation tests passed.")
