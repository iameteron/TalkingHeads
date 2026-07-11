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
    from .llm import DEFAULT_ALLOWED_ACTIONS_CALL, LLMConfig, predict_action_from_prompt
    from .trajectory_fixture import extract_state_from_entry, load_trajectory_entry
    from .test_sr import DEFAULT_STEP_IDX
except Exception:  # pragma: no cover
    from llm import DEFAULT_ALLOWED_ACTIONS_CALL, LLMConfig, predict_action_from_prompt
    from trajectory_fixture import extract_state_from_entry, load_trajectory_entry
    from test_sr import DEFAULT_STEP_IDX
from megaprompt.renders import Renderer

RENDER_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "templates" / "reasoning_or_ask_path" / "craftext.yaml"
)
 
def render_prompt(
    entry: dict[str, Any],
    state: Any,
    *,
    render_config_path: str | Path = RENDER_CONFIG_PATH,
    allow_ask_operator: bool = False,
) -> str:
    meta = entry.get("meta")
    # if isinstance(meta, dict):
    #     goal = str(meta.get("goal", "goal: Place a plant"))
    # else:
    #     goal = "goal: Place a plant"

    goal = "goal: Reach hidden exo-planet destination by landmark hints"
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
    step_idx: int = -1,
    run_count: int = 10,
    render_config_path: str | Path = RENDER_CONFIG_PATH,
    llm_cfg: LLMConfig | None = None,
) -> dict[str, Any]:
    entry = load_trajectory_entry(step_idx=step_idx)
    state = extract_state_from_entry(entry)
    prompt = render_prompt(
        entry=entry,
        state=state,
        render_config_path=render_config_path,
        allow_ask_operator=True,
    )

    print(prompt)
    cfg_llm = llm_cfg or LLMConfig(
        model_name_or_path=os.environ.get("CRAFTEXT_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        max_new_tokens=int(os.environ.get("CRAFTEXT_LLM_MAX_NEW_TOKENS", "64")),
        temperature=1.0,
        top_p=0.9,
    )

    asks_count = 0
    errors = 0
    all_answers: list[str] = []

    for _ in range(run_count):
        try:
            action, raw = predict_action_from_prompt(prompt, allow_ask_operator=True, cfg=cfg_llm)
            all_answers.append(raw)
        except Exception as exc:
            errors += 1
            all_answers.append(f"__ERROR__: {exc}")
            continue

        if action == "ASK_OPERATOR":
            asks_count += 1

    valid_runs = max(1, run_count - errors)
    ask_rate = asks_count / valid_runs

    print(f"ASK_OPERATOR count: {asks_count}")
    print(f"Ask rate: {ask_rate}")
    print(f"Errors: {errors}/{run_count}")

    return {
        "ask_count": asks_count,
        "ask_rate": ask_rate,
        "errors": errors,
        "all_answers": all_answers,
        "prompt": prompt,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure ASK_OPERATOR rate in exo-planet scenarios.")
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
