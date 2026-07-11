from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SERVER_DIR = Path(__file__).resolve().parent
_server_pkg = types.ModuleType("server")
_server_pkg.__path__ = [str(_SERVER_DIR)]
sys.modules.setdefault("server", _server_pkg)

for name, fname in [
    ("server.model_names", "model_names.py"),
    ("server.campaign_mode", "campaign_mode.py"),
    ("server.deployment_operator_limits", "deployment_operator_limits.py"),
    ("server.leaderboard", "leaderboard.py"),
]:
    spec = importlib.util.spec_from_file_location(
        name, _SERVER_DIR / fname, submodule_search_locations=[str(_SERVER_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

_runtime = types.ModuleType("server.runtime")
_runtime.create_isolated_session = MagicMock()
sys.modules["server.runtime"] = _runtime

spec = importlib.util.spec_from_file_location(
    "server.companion_bench",
    _SERVER_DIR / "companion_bench.py",
    submodule_search_locations=[str(_SERVER_DIR)],
)
bench_mod = importlib.util.module_from_spec(spec)
bench_mod.__package__ = "server"
bench_mod.__name__ = "server.companion_bench"
sys.modules["server.companion_bench"] = bench_mod
assert spec.loader is not None
spec.loader.exec_module(bench_mod)

spec = importlib.util.spec_from_file_location(
    "server.knowledge_paths",
    _SERVER_DIR / "knowledge_paths.py",
    submodule_search_locations=[str(_SERVER_DIR)],
)
kp_mod = importlib.util.module_from_spec(spec)
kp_mod.__package__ = "server"
kp_mod.__name__ = "server.knowledge_paths"
sys.modules["server.knowledge_paths"] = kp_mod
assert spec.loader is not None
spec.loader.exec_module(kp_mod)


def _mock_session(
    *,
    play_session_id: str = "client-a",
    texture_theme: str = "craftax",
    campaign_enabled: bool = False,
    companion_research_active: bool = False,
    active_agent_model: str = "gpt-test",
) -> MagicMock:
    sess = MagicMock()
    sess.play_session_id = play_session_id
    sess.texture_theme = texture_theme
    sess.active_agent_model = active_agent_model
    sess.companion_research_active = companion_research_active
    sess.is_arc_game = MagicMock(return_value=False)
    campaign = MagicMock()
    campaign.enabled = campaign_enabled
    sess.campaign_state = campaign
    return sess


def test_session_knowledge_slug_is_stable_hash() -> None:
    slug_a = kp_mod.session_knowledge_slug(_mock_session(play_session_id="abc-123"))
    slug_b = kp_mod.session_knowledge_slug(_mock_session(play_session_id="abc-123"))
    slug_c = kp_mod.session_knowledge_slug(_mock_session(play_session_id="other-id"))
    assert slug_a == slug_b
    assert slug_a != slug_c
    assert len(slug_a) == 16


def test_play_knowledge_paths_are_isolated_per_session() -> None:
    sess_a = _mock_session(play_session_id="user-a", campaign_enabled=True)
    sess_b = _mock_session(play_session_id="user-b", campaign_enabled=True)
    json_a, _txt_a = kp_mod.play_knowledge_paths_for_session(sess_a)
    json_b, _txt_b = kp_mod.play_knowledge_paths_for_session(sess_b)
    assert "sessions" in str(json_a)
    assert "sessions" in str(json_b)
    assert json_a.parent != json_b.parent
    assert kp_mod.session_knowledge_slug(sess_a) in str(json_a)
    assert kp_mod.session_knowledge_slug(sess_b) in str(json_b)


def test_play_knowledge_paths_do_not_use_global_model_store() -> None:
    sess = _mock_session(play_session_id="solo-user", campaign_enabled=True)
    json_path, _txt_path = kp_mod.play_knowledge_paths_for_session(sess)
    global_json, _global_txt = bench_mod.model_knowledge_paths("gpt-test", "craftax")
    assert json_path != global_json
