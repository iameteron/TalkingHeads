import os
from unittest.mock import patch

from .features import apply_demo_runtime_defaults, get_app_features


class _FakeSession:
    def __init__(self, *, model: str = "default-model", arc_game: bool = False) -> None:
        self.active_agent_model = model
        self.active_agent_mode = "hub"
        self._arc_game = arc_game
        self.arc_game_id = "ls20"
        self.megaprompt_config_name = "database_formulation"
        self.arc_prompt_extra = "custom"

    def is_arc_game(self) -> bool:
        return self._arc_game

    def set_active_agent_model(self, model_name: str) -> None:
        self.active_agent_model = model_name

    def set_active_agent_mode(self, mode: str) -> None:
        self.active_agent_mode = mode

    def set_megaprompt_config_name(self, name: str) -> None:
        self.megaprompt_config_name = name

    def set_game_kind(self, _game_kind: str, *, arc_game_id: str | None = None) -> bool:
        self.arc_game_id = arc_game_id or self.arc_game_id
        return True

    def set_arc_prompt_extra(self, extra: str) -> None:
        self.arc_prompt_extra = extra


def test_demo_profile_allows_model_selection() -> None:
    with patch.dict(os.environ, {"TALKINGHEADS_APP_PROFILE": "demo"}, clear=False):
        features = get_app_features()
    assert features.is_demo
    assert features.model_selection is True


def test_demo_runtime_defaults_apply_model_only_when_requested() -> None:
    sess = _FakeSession(model="user-selected-model")
    with patch.dict(
        os.environ,
        {
            "TALKINGHEADS_APP_PROFILE": "demo",
            "TALKINGHEADS_DEMO_AGENT_MODEL": "demo-default-model",
        },
        clear=False,
    ):
        apply_demo_runtime_defaults(sess)
    assert sess.active_agent_model == "user-selected-model"
    assert sess.active_agent_mode == "openrouter"

    with patch.dict(
        os.environ,
        {
            "TALKINGHEADS_APP_PROFILE": "demo",
            "TALKINGHEADS_DEMO_AGENT_MODEL": "demo-default-model",
        },
        clear=False,
    ):
        apply_demo_runtime_defaults(sess, apply_model_default=True)
    assert sess.active_agent_model == "demo-default-model"


def test_demo_runtime_defaults_use_sonnet_when_demo_model_env_is_empty() -> None:
    sess = _FakeSession(model="user-selected-model")
    with patch.dict(
        os.environ,
        {
            "TALKINGHEADS_APP_PROFILE": "demo",
            "TALKINGHEADS_DEMO_AGENT_MODEL": "",
        },
        clear=False,
    ):
        apply_demo_runtime_defaults(sess, apply_model_default=True)
    assert sess.active_agent_model == "anthropic/claude-sonnet-4.5"


def test_demo_profile_exposes_read_only_prompt_viewer() -> None:
    with patch.dict(os.environ, {"TALKINGHEADS_APP_PROFILE": "demo"}, clear=False):
        features = get_app_features()
    assert features.is_demo
    assert features.agent_prompt_debug is True
    assert features.arc_prompt_override is False


def test_demo_profile_enables_arc_multi_level() -> None:
    with patch.dict(os.environ, {"TALKINGHEADS_APP_PROFILE": "demo"}, clear=False):
        from .features import app_capabilities_payload, arc_multi_level_progression

        assert arc_multi_level_progression() is True
        payload = app_capabilities_payload()
    assert payload["arc_multi_level"] is True


def test_demo_profile_allows_craftax_companion_bench() -> None:
    sess = _FakeSession(arc_game=False)
    with patch.dict(os.environ, {"TALKINGHEADS_APP_PROFILE": "demo"}, clear=False):
        from .features import companion_bench_allowed

        assert companion_bench_allowed(sess) is True


def test_demo_profile_blocks_arc_companion_bench() -> None:
    sess = _FakeSession(arc_game=True)
    with patch.dict(os.environ, {"TALKINGHEADS_APP_PROFILE": "demo"}, clear=False):
        from .features import companion_bench_allowed

        assert companion_bench_allowed(sess) is False


def test_dev_profile_disables_arc_multi_level() -> None:
    with patch.dict(os.environ, {"TALKINGHEADS_APP_PROFILE": "dev"}, clear=False):
        from .features import arc_multi_level_progression

        assert arc_multi_level_progression() is False


def test_demo_runtime_defaults_lock_arc_obs_format() -> None:
    sess = _FakeSession(arc_game=True)
    with patch.dict(
        os.environ,
        {
            "TALKINGHEADS_APP_PROFILE": "demo",
            "TALKINGHEADS_DEMO_ARC_OBS_FORMAT": "arc_grid",
        },
        clear=False,
    ):
        apply_demo_runtime_defaults(sess)
    assert sess.megaprompt_config_name == "arc_grid"
    assert sess.arc_prompt_extra == ""


def test_demo_profile_excludes_lp85_from_arc_game_options() -> None:
    with patch.dict(os.environ, {"TALKINGHEADS_APP_PROFILE": "demo"}, clear=False):
        from .features import filter_arc_game_options
        from .arc_agi_adapter import ARC_GAME_OPTIONS

        options = filter_arc_game_options([dict(option) for option in ARC_GAME_OPTIONS])
    assert [option["id"] for option in options] == ["ar25", "bp35", "ls20"]


def test_dev_profile_keeps_lp85_in_arc_game_options() -> None:
    with patch.dict(os.environ, {"TALKINGHEADS_APP_PROFILE": "dev"}, clear=False):
        from .features import filter_arc_game_options
        from .arc_agi_adapter import ARC_GAME_OPTIONS

        options = filter_arc_game_options([dict(option) for option in ARC_GAME_OPTIONS])
    assert [option["id"] for option in options] == ["ar25", "bp35", "lp85", "ls20"]


def test_demo_profile_rejects_lp85_arc_game_selection() -> None:
    with patch.dict(os.environ, {"TALKINGHEADS_APP_PROFILE": "demo"}, clear=False):
        from .features import assert_arc_game_allowed_for_profile

        try:
            assert_arc_game_allowed_for_profile("lp85")
        except ValueError as exc:
            assert "not available in demo mode" in str(exc)
        else:
            raise AssertionError("expected lp85 to be rejected in demo mode")
        assert assert_arc_game_allowed_for_profile("ls20") == "ls20"


if __name__ == "__main__":
    test_demo_profile_allows_model_selection()
    test_demo_runtime_defaults_apply_model_only_when_requested()
    test_demo_profile_enables_arc_multi_level()
    test_demo_profile_allows_craftax_companion_bench()
    test_demo_profile_blocks_arc_companion_bench()
    test_dev_profile_disables_arc_multi_level()
    test_demo_runtime_defaults_lock_arc_obs_format()
    test_demo_profile_excludes_lp85_from_arc_game_options()
    test_dev_profile_keeps_lp85_in_arc_game_options()
    test_demo_profile_rejects_lp85_arc_game_selection()
    print("ok")
