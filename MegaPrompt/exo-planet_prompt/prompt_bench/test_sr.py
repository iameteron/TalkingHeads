from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

MEGAPROMPT_ROOT = Path(__file__).resolve().parents[2]
EXO_ROOT = MEGAPROMPT_ROOT / "exo-planet_prompt"
for p in (MEGAPROMPT_ROOT, EXO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from megaprompt.renders import Renderer

try:
    from .exo_fixture import EXO_SR_SUCCESS_ACTIONS, extract_state_from_entry, load_exo_trajectory_entry
    from .predict import EXO_ALLOWED_ACTIONS_CALL, EXO_ENV_ACTIONS, LLMConfig, predict_action_from_prompt
except Exception:  # pragma: no cover
    from exo_fixture import EXO_SR_SUCCESS_ACTIONS, extract_state_from_entry, load_exo_trajectory_entry
    from predict import EXO_ALLOWED_ACTIONS_CALL, EXO_ENV_ACTIONS, LLMConfig, predict_action_from_prompt

DEFAULT_CONFIG = EXO_ROOT / "templates" / "no_dialog" / "exo-planet.yaml"
DEFAULT_STEP_IDX = -1


def render_prompt(
    entry: dict[str, Any],
    state: Any,
    *,
    render_config_path: str | Path = DEFAULT_CONFIG,
    allow_ask_operator: bool = False,
    extra_meta: dict[str, Any] | None = None,
) -> str:
    meta = entry.get("meta")
    goal = str(meta.get("goal", "goal: Deploy Replicator")) if isinstance(meta, dict) else "goal: Deploy Replicator"
    dialog_history = entry.get("oracle_dialog")
    if not isinstance(dialog_history, list):
        dialog_history = []

    renderer = Renderer(config_path=render_config_path)
    action_space = EXO_ALLOWED_ACTIONS_CALL if allow_ask_operator else EXO_ENV_ACTIONS
    payload: dict[str, Any] = {
        "goal": goal,
        "obs": state,
        "act": action_space,
        "dialog": dialog_history,
        "action_history": [],
        "state_history": [],
        "knowledge": "(none yet)",
    }
    if extra_meta:
        payload.update(extra_meta)
    return renderer.render(payload)


def test(
    *,
    step_idx: int = DEFAULT_STEP_IDX,
    run_count: int = 10,
    render_config_path: str | Path = DEFAULT_CONFIG,
    llm_cfg: LLMConfig | None = None,
) -> dict[str, Any]:
    entry = load_exo_trajectory_entry(step_idx=step_idx)
    state = extract_state_from_entry(entry)
    act_prompt = render_prompt(entry=entry, state=state, render_config_path=render_config_path)

    cfg_llm = llm_cfg or LLMConfig(
        model_name_or_path=os.environ.get("CRAFTEXT_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        max_new_tokens=int(os.environ.get("CRAFTEXT_LLM_MAX_NEW_TOKENS", "64")),
    )

    success = 0
    reasoning_tags = 0
    errors = 0
    all_answers: list[str] = []

    for _ in range(run_count):
        try:
            action, raw = predict_action_from_prompt(act_prompt, allow_ask_operator=False, cfg=cfg_llm)
            all_answers.append(raw)
        except Exception as exc:
            errors += 1
            all_answers.append(f"__ERROR__: {exc}")
            continue

        if action in EXO_SR_SUCCESS_ACTIONS:
            success += 1
        reasoning_tags += raw.count("</reasoning>")

    success_rate = success / run_count
    print(f"Success rate: {success_rate}")
    print(f"Errors: {errors}/{run_count}")

    return {
        "success_rate": success_rate,
        "errors": errors,
        "all_answers": all_answers,
        "prompt": act_prompt,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exo-planet prompt success-rate bench")
    parser.add_argument("--render-config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--step-idx", type=int, default=DEFAULT_STEP_IDX)
    parser.add_argument("--run-count", type=int, default=10)
    parser.add_argument("--llm-model", type=str, default=os.environ.get("CRAFTEXT_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    args = parser.parse_args()
    test(
        step_idx=args.step_idx,
        run_count=args.run_count,
        render_config_path=args.render_config,
        llm_cfg=LLMConfig(model_name_or_path=args.llm_model),
    )
