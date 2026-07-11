from dataclasses import dataclass
from typing import Optional, Literal


ExpertMode = Literal["local", "hub", "openrouter"]



@dataclass
class GenConfig:
    """Text generation configuration shared across local and hub experts."""

    max_new_tokens: int = 1024
    do_sample: bool = False          # greedy by default (fast)
    temperature: float = 0.7         # only used if do_sample=True
    top_p: float = 0.9               # only used if do_sample=True
    use_cache: bool = True


@dataclass
class ExpertConfig:
    """Configuration for a single expert (map, mechanics, question, goal)."""

    mode: str                  # "local" | "hub" | "openrouter"
    model_path: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    provider: Optional[str] = None


@dataclass
class OracleConfig:
    """Top-level configuration for Oracle, Intent, and all experts."""

    map_expert_mode: ExpertMode = "hub"
    mechanics_expert_mode: ExpertMode = "hub"
    action_expert_mode: ExpertMode = "hub"
    question_expert_mode: ExpertMode = "hub"
    goal_expert_mode: ExpertMode = "hub"
    path_expert_helper_mode: ExpertMode = "hub"
    path_expert_mode: ExpertMode = "hub"

    map_expert_model_path: str = "Qwen/Qwen2.5-Coder-32B"
    mechanics_expert_model_path: str = "Qwen/Qwen2.5-Coder-32B"
    action_expert_model_path: str = "Qwen/Qwen2.5-32B-Instruct"
    question_expert_model_path: str = "Qwen/Qwen2.5-Coder-32B"
    goal_expert_model_path: str = "Qwen/Qwen2.5-Coder-32B"
    path_expert_helper_model_path: str = "Qwen/Qwen2.5-32B-Instruct"
    path_expert_model_path: str = "Qwen/Qwen2.5-32B-Instruct"

    # Optional: Hub inference (used when expert mode is "hub")
    hub_api_key: Optional[str] = None
    hub_base_url: Optional[str] = None
    hub_provider: Optional[str] = None
    hub_default_system_message: str = "You are a helpful assistant."

    # Optional: OpenRouter inference (used when expert mode is "openrouter")
    openrouter_api_key: Optional[str] = None

    # Optional: generation config for experts
    gen_config: Optional[GenConfig] = None

    # Path pipeline: max characters for a single operator reply to the agent (Craftext rollouts).
    # None disables enforcement. When exceeded, TalkingHeads asks the path expert to shorten (same
    # pattern as the “no arrows” repair); if still too long, a short fallback message is returned.
    operator_max_answer_chars: Optional[int] = None


def oracle_config_to_expert_config(oracle_config: OracleConfig):
    def api_key_for_mode(mode: str) -> Optional[str]:
        if mode == "openrouter":
            return oracle_config.openrouter_api_key or oracle_config.hub_api_key
        return oracle_config.hub_api_key

    experts_configs_dict = {}
    experts_configs_dict["map"] = ExpertConfig(
        mode=oracle_config.map_expert_mode,
        model_path=oracle_config.map_expert_model_path,
        api_key=api_key_for_mode(oracle_config.map_expert_mode),
        base_url=oracle_config.hub_base_url,
        provider=oracle_config.hub_provider,
    )
    experts_configs_dict["mechanics"] = ExpertConfig(
        mode=oracle_config.mechanics_expert_mode,
        model_path=oracle_config.mechanics_expert_model_path,
        api_key=api_key_for_mode(oracle_config.mechanics_expert_mode),
        base_url=oracle_config.hub_base_url,
        provider=oracle_config.hub_provider,
    )
    experts_configs_dict["action"] = ExpertConfig(
        mode=oracle_config.action_expert_mode,
        model_path=oracle_config.action_expert_model_path,
        api_key=api_key_for_mode(oracle_config.action_expert_mode),
        base_url=oracle_config.hub_base_url,
        provider=oracle_config.hub_provider,
    )
    experts_configs_dict["question"] = ExpertConfig(
        mode=oracle_config.question_expert_mode,
        model_path=oracle_config.question_expert_model_path,
        api_key=api_key_for_mode(oracle_config.question_expert_mode),
        base_url=oracle_config.hub_base_url,
        provider=oracle_config.hub_provider,
    )
    experts_configs_dict["goal"] = ExpertConfig(
        mode=oracle_config.goal_expert_mode,
        model_path=oracle_config.goal_expert_model_path,
        api_key=api_key_for_mode(oracle_config.goal_expert_mode),
        base_url=oracle_config.hub_base_url,
        provider=oracle_config.hub_provider,
    )
    experts_configs_dict["path_expert_helper"] = ExpertConfig(
        mode=oracle_config.path_expert_helper_mode,
        model_path=oracle_config.path_expert_helper_model_path,
        api_key=api_key_for_mode(oracle_config.path_expert_helper_mode),
        base_url=oracle_config.hub_base_url,
        provider=oracle_config.hub_provider,
    )
    experts_configs_dict["path_expert"] = ExpertConfig(
        mode=oracle_config.path_expert_mode,
        model_path=oracle_config.path_expert_model_path,
        api_key=api_key_for_mode(oracle_config.path_expert_mode),
        base_url=oracle_config.hub_base_url,
        provider=oracle_config.hub_provider,
    )
    return experts_configs_dict