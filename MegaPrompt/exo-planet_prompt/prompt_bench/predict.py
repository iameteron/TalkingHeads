from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional, Tuple

MEGAPROMPT_ROOT = Path(__file__).resolve().parents[2]
EXO_ROOT = MEGAPROMPT_ROOT / "exo-planet_prompt"
BENCH_DIR = Path(__file__).resolve().parent
CRAFTEXT_LLM_PATH = MEGAPROMPT_ROOT / "craftext_prompt" / "prompt_bench" / "llm.py"

for p in (EXO_ROOT, BENCH_DIR):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from action_bridge import to_engine_action  # noqa: E402
from actions import EXO_ALLOWED_ACTIONS_CALL, EXO_ALLOWED_ACTIONS_META, EXO_ENV_ACTIONS  # noqa: E402
from contract import (  # noqa: E402
    EXO_LEGACY_TERMS,
    allowed_actions_for,
    extract_action,
    extract_question,
    validate_response_contract,
)


def _load_craftext_llm():
    spec = importlib.util.spec_from_file_location("_craftext_prompt_bench_llm", CRAFTEXT_LLM_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load craftext llm from {CRAFTEXT_LLM_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_craftext_llm = _load_craftext_llm()
LLMConfig = _craftext_llm.LLMConfig
llm_complete = _craftext_llm.llm_complete


def predict_action_from_prompt(
    prompt: str,
    *,
    allow_ask_operator: bool = False,
    allow_update_database: bool = False,
    cfg: Optional[LLMConfig] = None,
) -> Tuple[str, str]:
    raw = llm_complete(prompt, cfg=cfg)
    action = extract_action(raw)
    question = extract_question(raw) if allow_ask_operator else None
    if allow_ask_operator and action == "ASK_OPERATOR" and not question:
        strict_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT: If you choose ASK_OPERATOR, your output MUST include "
            "both <action>ASK_OPERATOR</action> and a non-empty "
            "<question>...</question>."
        )
        raw = llm_complete(strict_prompt, cfg=cfg)
        action = extract_action(raw)
        question = extract_question(raw)

    validate_response_contract(
        raw,
        allowed_actions=allowed_actions_for(
            allow_ask_operator=allow_ask_operator,
            allow_update_database=allow_update_database,
        ),
        allow_ask_operator=allow_ask_operator,
        allow_update_database=allow_update_database,
    )
    return action, raw


__all__ = [
    "EXO_ALLOWED_ACTIONS_CALL",
    "EXO_ALLOWED_ACTIONS_META",
    "EXO_ENV_ACTIONS",
    "EXO_LEGACY_TERMS",
    "LLMConfig",
    "extract_action",
    "predict_action_from_prompt",
    "to_engine_action",
    "validate_response_contract",
]
