import os
from dataclasses import dataclass
from typing import Any


APP_PROFILE_ENV = "TALKINGHEADS_APP_PROFILE"
DEMO_AGENT_MODEL_ENV = "TALKINGHEADS_DEMO_AGENT_MODEL"
DEMO_ARC_OBS_FORMAT_ENV = "TALKINGHEADS_DEMO_ARC_OBS_FORMAT"
DEFAULT_DEMO_AGENT_MODEL = "anthropic/claude-sonnet-4.5"
DEMO_EXCLUDED_ARC_GAME_IDS = frozenset({"lp85"})


@dataclass(frozen=True)
class AppFeatures:
    profile: str
    settings_api_keys: bool
    model_selection: bool
    expert_model_settings: bool
    observation_format_selection: bool
    arc_prompt_override: bool
    agent_prompt_debug: bool
    setup_wizard: bool
    leaderboard: bool
    human_operator: bool
    companion_bench: bool

    @property
    def is_demo(self) -> bool:
        return self.profile == "demo"

    def as_dict(self) -> dict[str, bool]:
        return {
            "settings_api_keys": self.settings_api_keys,
            "model_selection": self.model_selection,
            "expert_model_settings": self.expert_model_settings,
            "observation_format_selection": self.observation_format_selection,
            "arc_prompt_override": self.arc_prompt_override,
            "agent_prompt_debug": self.agent_prompt_debug,
            "setup_wizard": self.setup_wizard,
            "leaderboard": self.leaderboard,
            "human_operator": self.human_operator,
            "companion_bench": self.companion_bench,
        }


def app_profile() -> str:
    value = os.environ.get(APP_PROFILE_ENV, "dev").strip().lower()
    if value in {"demo", "public"}:
        return "demo"
    return "dev"


def get_app_features() -> AppFeatures:
    profile = app_profile()
    if profile == "demo":
        return AppFeatures(
            profile=profile,
            settings_api_keys=False,
            model_selection=True,
            expert_model_settings=False,
            observation_format_selection=False,
            arc_prompt_override=False,
            agent_prompt_debug=True,
            setup_wizard=True,
            leaderboard=True,
            human_operator=True,
            companion_bench=False,
        )
    return AppFeatures(
        profile=profile,
        settings_api_keys=True,
        model_selection=True,
        expert_model_settings=True,
        observation_format_selection=True,
        arc_prompt_override=True,
        agent_prompt_debug=True,
        setup_wizard=True,
        leaderboard=True,
        human_operator=True,
        companion_bench=True,
    )


def demo_agent_model() -> str:
    return os.environ.get(DEMO_AGENT_MODEL_ENV, DEFAULT_DEMO_AGENT_MODEL).strip() or DEFAULT_DEMO_AGENT_MODEL


def demo_arc_obs_format() -> str:
    value = os.environ.get(DEMO_ARC_OBS_FORMAT_ENV, "arc_image").strip()
    return value or "arc_image"


def demo_excluded_arc_game_ids() -> frozenset[str]:
    if get_app_features().is_demo:
        return DEMO_EXCLUDED_ARC_GAME_IDS
    return frozenset()


def filter_arc_game_options(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded = demo_excluded_arc_game_ids()
    if not excluded:
        return options
    return [option for option in options if str(option.get("id") or "") not in excluded]


def assert_arc_game_allowed_for_profile(game_id: str | None) -> str:
    from .arc_agi_adapter import normalize_arc_game_id

    normalized = normalize_arc_game_id(game_id)
    if normalized in demo_excluded_arc_game_ids():
        raise ValueError(
            f"ARC-AGI-3 game '{game_id}' is not available in demo mode."
        )
    return normalized


def coerce_demo_arc_game_selection(sess: Any) -> None:
    if not get_app_features().is_demo or not sess.is_arc_game():
        return
    if sess.arc_game_id not in DEMO_EXCLUDED_ARC_GAME_IDS:
        return
    sess.set_game_kind(sess.game_kind, arc_game_id="ls20")


def arc_multi_level_progression() -> bool:
    """Demo visitors play through ARC in-game levels instead of one-level episodes."""
    return get_app_features().is_demo


def companion_bench_allowed(sess: Any) -> bool:
    """Companion research/test is always available for Crafter and Exo-planet."""
    features = get_app_features()
    if features.companion_bench:
        return True
    if features.is_demo and not sess.is_arc_game():
        return True
    return False


def apply_demo_runtime_defaults(sess: Any, *, apply_model_default: bool = False) -> None:
    features = get_app_features()
    if not features.is_demo:
        return
    if apply_model_default:
        model = demo_agent_model()
        if model:
            sess.set_active_agent_model(model)
    sess.set_active_agent_mode("openrouter")
    coerce_demo_arc_game_selection(sess)
    if sess.is_arc_game():
        sess.set_megaprompt_config_name(demo_arc_obs_format())
        sess.set_arc_prompt_extra("")


def app_capabilities_payload() -> dict[str, Any]:
    features = get_app_features()
    return {
        "app_profile": features.profile,
        "features": features.as_dict(),
        "demo_defaults": {
            "agent_model": demo_agent_model(),
            "arc_obs_format": demo_arc_obs_format(),
        } if features.is_demo else {},
        "arc_multi_level": arc_multi_level_progression(),
    }
