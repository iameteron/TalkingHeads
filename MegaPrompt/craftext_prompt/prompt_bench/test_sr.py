from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

MEGAPROMPTS_ROOT = Path(__file__).resolve().parents[2]
if str(MEGAPROMPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGAPROMPTS_ROOT))

try:
    from .llm import (
        DEFAULT_ALLOWED_ACTIONS,
        DEFAULT_ALLOWED_ACTIONS_CALL,
        LLMConfig,
        predict_action_from_prompt,
    )
    from .trajectory_fixture import extract_state_from_entry, load_trajectory_entry
except Exception:  # pragma: no cover
    from llm import DEFAULT_ALLOWED_ACTIONS, DEFAULT_ALLOWED_ACTIONS_CALL, LLMConfig, predict_action_from_prompt
    from trajectory_fixture import extract_state_from_entry, load_trajectory_entry
from megaprompt.renders import Renderer

RENDER_CONFIG_PATH = Path(__file__).resolve().parents[1] / "templates" / "base_exmple" / "craftext.yaml"
DEFAULT_STEP_IDX = -1


def render_prompt(
    entry: dict[str, Any],
    state: Any,
    *,
    render_config_path: str | Path = RENDER_CONFIG_PATH,
    allow_ask_operator: bool = False,
) -> str:
    meta = entry.get("meta")
    if isinstance(meta, dict):
        goal = str(meta.get("goal", "goal: Place a crafting table"))
    else:
        goal = "goal: Place a crafting table"

    dialog_history = entry.get("oracle_dialog")
    if not isinstance(dialog_history, list):
        dialog_history = []

    renderer = Renderer(render_config_path)
    action_space = DEFAULT_ALLOWED_ACTIONS_CALL if allow_ask_operator else DEFAULT_ALLOWED_ACTIONS
    return renderer.render(
        {
            "goal": goal,
            "obs": state,
            "act": action_space,
            "dialog": dialog_history,
        }
    )


def test(
    *,
    step_idx: int = DEFAULT_STEP_IDX,
    run_count: int = 10,
    render_config_path: str | Path = RENDER_CONFIG_PATH,
    llm_cfg: LLMConfig | None = None,
) -> dict[str, Any]:
    entry = load_trajectory_entry(step_idx=step_idx)
    state = extract_state_from_entry(entry)
    act_prompt = render_prompt(entry=entry, state=state, render_config_path=render_config_path)

    print(act_prompt)
    cfg_llm = llm_cfg or LLMConfig(
        model_name_or_path=os.environ.get("CRAFTEXT_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        max_new_tokens=int(os.environ.get("CRAFTEXT_LLM_MAX_NEW_TOKENS", "64")),
        temperature=1.0,
        top_p=0.9,
    )

    place_table_success = 0
    count_of_answers = 0
    all_answers: list[str] = []
    errors = 0

    for _ in range(run_count):
        try:
            action, raw = predict_action_from_prompt(act_prompt, allow_ask_operator=False, cfg=cfg_llm)
            all_answers.append(raw)
        except Exception as exc:
            errors += 1
            all_answers.append(f"__ERROR__: {exc}")
            continue

        if action in {"PLACE_STONE", "DO (TO GATHER SOMETHING)", "PLACE_TABLE"}:
            place_table_success += 1

        count_of_answers += raw.count("</reasoning>")

    success_rate = place_table_success / run_count
    avg_reasoning = count_of_answers / max(1, (run_count - errors))

    print(f"Success rate: {success_rate}")
    print(f"Average number of reasoning tags: {avg_reasoning}")
    print(f"Errors: {errors}/{run_count}")

    return {
        "success_rate": success_rate,
        "average_number_of_reasoning_tags": avg_reasoning,
        "errors": errors,
        "all_answers": all_answers,
        "prompt": act_prompt,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run craftext prompt bench on place_a_table trajectory.")
    parser.add_argument("--render-config", type=str, default=str(RENDER_CONFIG_PATH))
    parser.add_argument("--step-idx", type=int, default=DEFAULT_STEP_IDX)
    parser.add_argument("--run-count", type=int, default=10)
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.environ.get("CRAFTEXT_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        help="Model name or local path for HF model.",
    )
    parser.add_argument(
        "--llm-max-new-tokens",
        type=int,
        default=int(os.environ.get("CRAFTEXT_LLM_MAX_NEW_TOKENS", "64")),
    )
    parser.add_argument("--llm-temperature", type=float, default=1.0)
    parser.add_argument("--llm-top-p", type=float, default=0.9)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    llm_cfg = LLMConfig(
        model_name_or_path=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
    )
    test(
        step_idx=args.step_idx,
        run_count=args.run_count,
        render_config_path=args.render_config,
        llm_cfg=llm_cfg,
    )


