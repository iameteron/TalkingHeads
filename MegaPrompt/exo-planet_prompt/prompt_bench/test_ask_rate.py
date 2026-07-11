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

try:
    from .exo_fixture import extract_state_from_entry, load_exo_trajectory_entry
    from .predict import LLMConfig, predict_action_from_prompt
    from .test_sr import DEFAULT_STEP_IDX, render_prompt
except Exception:  # pragma: no cover
    from exo_fixture import extract_state_from_entry, load_exo_trajectory_entry
    from predict import LLMConfig, predict_action_from_prompt
    from test_sr import DEFAULT_STEP_IDX, render_prompt

DEFAULT_CONFIG = EXO_ROOT / "templates" / "reasoning_or_ask_path" / "exo-planet.yaml"


def test(
    *,
    step_idx: int = -1,
    run_count: int = 10,
    render_config_path: str | Path = DEFAULT_CONFIG,
    llm_cfg: LLMConfig | None = None,
) -> dict[str, Any]:
    entry = load_exo_trajectory_entry(step_idx=step_idx)
    state = extract_state_from_entry(entry)
    prompt = render_prompt(
        entry=entry,
        state=state,
        render_config_path=render_config_path,
        allow_ask_operator=True,
    )

    cfg_llm = llm_cfg or LLMConfig(
        model_name_or_path=os.environ.get("CRAFTEXT_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        max_new_tokens=int(os.environ.get("CRAFTEXT_LLM_MAX_NEW_TOKENS", "64")),
    )

    asks = 0
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
            asks += 1

    valid = max(1, run_count - errors)
    ask_rate = asks / valid
    print(f"Ask rate: {ask_rate} ({asks}/{valid})")
    print(f"Errors: {errors}/{run_count}")

    return {"ask_rate": ask_rate, "ask_count": asks, "errors": errors, "all_answers": all_answers}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exo-planet ASK_OPERATOR rate bench")
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
