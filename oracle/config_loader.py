"""
Load Oracle, Intent, and expert configuration from YAML.
"""
from pathlib import Path
from typing import Any, Optional

import yaml

from oracle.configs import OracleConfig, GenConfig


def load_config(path: Optional[str] = None) -> OracleConfig:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to YAML config file. If None, uses config/oracle_config.yaml
              relative to the project root (parent of oracle package).

    Returns:
        OracleConfig populated from the YAML file.
    """
    if path is None:
        # Default: config/oracle_config.yaml next to project root
        pkg_dir = Path(__file__).resolve().parent
        root = pkg_dir.parent
        path = str(root / "config" / "oracle_config.yaml")

    config_path = Path(path).resolve()
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return _build_oracle_config(data)


def _build_oracle_config(data: dict) -> OracleConfig:
    """Build OracleConfig from parsed YAML dict."""
    oracle = data.get("oracle", {})

    map_expert = oracle.get("map_expert", {})
    mechanics_expert = oracle.get("mechanics_expert", {})
    action_expert = oracle.get("action_expert", {})
    question_expert = oracle.get("question_expert", {})
    goal_expert = oracle.get("goal_expert", {})
    path_expert_helper = oracle.get("path_expert_helper", {})
    path_expert = oracle.get("path_expert", {})
    hub = oracle.get("hub", {})
    gen_data = oracle.get("gen", {})
    path_pipeline = oracle.get("path_pipeline", {})

    map_expert_mode = map_expert.get("mode", "hub")
    map_expert_model_path = map_expert.get("model_path", "Qwen/Qwen2.5-Coder-32B")
    mechanics_expert_mode = mechanics_expert.get("mode", "hub")
    mechanics_expert_model_path = mechanics_expert.get("model_path", "Qwen/Qwen2.5-Coder-32B")
    action_expert_mode = action_expert.get("mode", "hub")
    action_expert_model_path = action_expert.get("model_path", "Qwen/Qwen2.5-32B-Instruct")
    question_expert_mode = question_expert.get("mode", "hub")
    question_expert_model_path = question_expert.get("model_path", "Qwen/Qwen2.5-Coder-32B")
    goal_expert_mode = goal_expert.get("mode", "hub")
    goal_expert_model_path = goal_expert.get("model_path", "Qwen/Qwen2.5-Coder-32B")
    path_expert_helper_mode = path_expert_helper.get("mode", path_expert.get("mode", "hub"))
    path_expert_helper_model_path = path_expert_helper.get(
        "model_path",
        path_expert.get("model_path", "Qwen/Qwen2.5-32B-Instruct"),
    )
    path_expert_mode = path_expert.get("mode", "hub")
    path_expert_model_path = path_expert.get("model_path", "Qwen/Qwen2.5-32B-Instruct")

    hub_api_key = hub.get("api_key")
    hub_base_url = hub.get("base_url")
    hub_provider = hub.get("provider")
    hub_default_system_message = hub.get("default_system_message", "You are a helpful assistant.")

    gen_config = GenConfig(
        max_new_tokens=gen_data.get("max_new_tokens", 1024),
        do_sample=gen_data.get("do_sample", False),
        temperature=gen_data.get("temperature", 0.7),
        top_p=gen_data.get("top_p", 0.9),
        use_cache=gen_data.get("use_cache", True),
    )

    operator_max_answer_chars = path_pipeline.get("operator_max_answer_chars")

    return OracleConfig(
        map_expert_mode=map_expert_mode,
        mechanics_expert_mode=mechanics_expert_mode,
        action_expert_mode=action_expert_mode,
        question_expert_mode=question_expert_mode,
        goal_expert_mode=goal_expert_mode,
        path_expert_helper_mode=path_expert_helper_mode,
        path_expert_mode=path_expert_mode,
        map_expert_model_path=map_expert_model_path,
        mechanics_expert_model_path=mechanics_expert_model_path,
        action_expert_model_path=action_expert_model_path,
        question_expert_model_path=question_expert_model_path,
        goal_expert_model_path=goal_expert_model_path,
        path_expert_helper_model_path=path_expert_helper_model_path,
        path_expert_model_path=path_expert_model_path,
        hub_api_key=hub_api_key,
        hub_base_url=hub_base_url,
        hub_provider=hub_provider,
        hub_default_system_message=hub_default_system_message,
        gen_config=gen_config,
        operator_max_answer_chars=operator_max_answer_chars,
    )
