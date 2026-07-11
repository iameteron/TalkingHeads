from types import SimpleNamespace
import importlib.util
import numpy as np
from pathlib import Path
import sys
import unittest

_MOD_PATH = Path(__file__).resolve().with_name("arc_agi_adapter.py")
_SPEC = importlib.util.spec_from_file_location("arc_agi_adapter_for_test", _MOD_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)
ArcAgiAdapter = _MOD.ArcAgiAdapter
parse_arc_action_text = _MOD.parse_arc_action_text
normalize_arc_game_id = _MOD.normalize_arc_game_id


class _FakeAction:
    def __init__(self, name):
        self.name = name


class _FakeGameAction:
    ACTION1 = _FakeAction("ACTION1")
    ACTION2 = _FakeAction("ACTION2")
    ACTION6 = _FakeAction("ACTION6")


class _FakeEnv:
    def __init__(self):
        self.action_space = [_FakeGameAction.ACTION1, _FakeGameAction.ACTION6]
        self.info = SimpleNamespace(title="Fake ARC")
        self.calls = []
        self.obs = SimpleNamespace(
            state=SimpleNamespace(name="PLAYING"),
            levels_completed=0,
            text="fake observation",
            frame=[np.array([[0, 1], [14, 15]], dtype=np.int8)],
        )

    def reset(self):
        self.calls.append(("reset",))
        return self.obs

    def step(self, action, **kwargs):
        self.calls.append(("step", action.name, kwargs))
        return self.obs


class _FakeArcade:
    def __init__(self, env):
        self.env = env

    def make(self, game_id, seed=0):
        self.calls = [("make", game_id, seed)]
        return self.env


def _adapter():
    env = _FakeEnv()
    arcade = _FakeArcade(env)
    adapter = ArcAgiAdapter(
        "ls20",
        arcade_factory=lambda: arcade,
        game_action_cls=_FakeGameAction,
    )
    return adapter, env


def test_reset_returns_frame_with_available_actions():
    adapter, _env = _adapter()
    frame = adapter.reset()
    assert frame["arc"]["game_id"] == "ls20"
    assert frame["arc"]["available_actions"] == ["ACTION1", "ACTION6"]
    assert frame["arc"]["frame_grid"] == "01\nef"
    assert frame["png_b64"]
    assert "Game: ls20 (Fake ARC)" in frame["agent_observation"]
    assert "Current frame grid:" in frame["agent_observation"]
    assert "01\nef" in frame["agent_observation"]


def test_new_arc_game_ids_are_supported():
    assert normalize_arc_game_id("ar25") == "ar25"
    assert normalize_arc_game_id("bp35") == "bp35"


def test_action1_calls_env_step():
    adapter, env = _adapter()
    adapter.step("ACTION1")
    assert env.calls[-1] == ("step", "ACTION1", {})


def test_action6_calls_env_step_with_coordinates():
    adapter, env = _adapter()
    adapter.step("ACTION6 32 31")
    assert env.calls[-1] == (
        "step",
        "ACTION6",
        {"data": {"x": 32, "y": 31}},
    )


def test_parse_action6_bracket_coordinates():
    assert parse_arc_action_text("ACTION6 [12, 40]") == (
        "ACTION6",
        {"x": 12, "y": 40},
    )


def test_parse_action6_compact_bracket_coordinates():
    assert parse_arc_action_text("ACTION6 [32,31]") == (
        "ACTION6",
        {"x": 32, "y": 31},
    )


def test_get_arc_game_preview_uses_rendered_frame(monkeypatch=None):
    adapter, _env = _adapter()
    original_cls = _MOD.ArcAgiAdapter

    class _PreviewAdapter:
        def __init__(self, game_id, **kwargs):
            self.game_id = game_id

        def render_frame(self):
            return original_cls("ls20", arcade_factory=lambda: _FakeArcade(_FakeEnv())).render_frame()

    _MOD.ArcAgiAdapter = _PreviewAdapter  # type: ignore[assignment]
    try:
        preview = _MOD.get_arc_game_preview("ls20")
        assert preview["game_id"] == "ls20"
        assert preview["png_b64"]
        assert preview["human_operator"] is True
        assert preview["ai_operator"] is False
    finally:
        _MOD.ArcAgiAdapter = original_cls


def test_invalid_action_and_missing_coords_raise_clear_errors():
    with unittest.TestCase().assertRaisesRegex(ValueError, "ACTION1..ACTION7"):
        parse_arc_action_text("LEFT")
    with unittest.TestCase().assertRaisesRegex(ValueError, "requires x and y"):
        parse_arc_action_text("ACTION6")


if __name__ == "__main__":
    test_reset_returns_frame_with_available_actions()
    test_action1_calls_env_step()
    test_action6_calls_env_step_with_coordinates()
    test_parse_action6_bracket_coordinates()
    test_get_arc_game_preview_uses_rendered_frame()
    test_invalid_action_and_missing_coords_raise_clear_errors()
