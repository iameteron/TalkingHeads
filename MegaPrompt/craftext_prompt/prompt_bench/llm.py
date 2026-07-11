from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE)
_QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL | re.IGNORECASE)
_TO_DATABASE_RE = re.compile(r"<to_database>\s*(.*?)\s*</to_database>", re.DOTALL | re.IGNORECASE)
_HF_CACHE: dict[str, object] = {}


DEFAULT_ALLOWED_ACTIONS: list[str] = [
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "NOOP",
    "DO (TO GATHER SOMETHING)",
    "DO (TO FIGHT)",
    "DO (DRINK WATER)",
    "SLEEP",
    "REST",
    "PLACE_STONE",
    "PLACE_TABLE",
    "PLACE_FURNACE",
    "PLACE_PLANT",
    "MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE",
    "MAKE_IRON_PICKAXE",
    "MAKE_WOOD_SWORD",
    "MAKE_STONE_SWORD",
    "MAKE_IRON_SWORD",
]

DEFAULT_ALLOWED_ACTIONS_CALL: list[str] = [*DEFAULT_ALLOWED_ACTIONS, "ASK_OPERATOR"]
DEFAULT_ALLOWED_ACTIONS_META: list[str] = [*DEFAULT_ALLOWED_ACTIONS_CALL, "UPDATE_DATABASE"]
LEGACY_TERMS: tuple[str, ...] = ("craftax", "minecraft", "crafting table")


def extract_action(text: str) -> Optional[str]:
    if not text:
        return None
    m = _ACTION_RE.search(text)
    if not m:
        return None
    action = (m.group(1) or "").strip()
    action = action.replace("\n", " ").strip()
    return action or None


def extract_question(text: str) -> Optional[str]:
    if not text:
        return None
    m = _QUESTION_RE.search(text)
    if not m:
        return None
    question = (m.group(1) or "").strip()
    return question or None


def extract_to_database(text: str) -> Optional[str]:
    if not text:
        return None
    m = _TO_DATABASE_RE.search(text)
    if not m:
        return None
    payload = (m.group(1) or "").strip()
    return payload or None


def _contains_legacy_terms(text: str) -> Optional[str]:
    lowered = str(text or "").lower()
    for term in LEGACY_TERMS:
        if term in lowered:
            return term
    return None


def validate_response_contract(
    raw: str,
    *,
    allowed_actions: Sequence[str],
    allow_ask_operator: bool,
    allow_update_database: bool,
) -> tuple[str, Optional[str]]:
    action = extract_action(raw)
    question = extract_question(raw)
    to_database = extract_to_database(raw)

    if action is None and question is None:
        raise ValueError(f"LLM output did not contain contract tags. Output was:\n{raw}")
    if action is None and question is not None:
        raise ValueError("Question is present but <action>...</action> is missing.")
    if action is None:
        raise ValueError(f"LLM output did not contain <action>...</action>. Output was:\n{raw}")

    if action == "ASK_OPERATOR":
        if not allow_ask_operator:
            raise ValueError("ASK_OPERATOR is forbidden in this benchmark configuration.")
        if not question:
            raise ValueError("ASK_OPERATOR requires non-empty <question>...</question>.")
        if to_database:
            raise ValueError("ASK_OPERATOR response must not include <to_database>.")
    else:
        if question:
            raise ValueError("Mixed output is forbidden: question is allowed only with ASK_OPERATOR.")

    if action == "UPDATE_DATABASE":
        if not allow_update_database:
            raise ValueError("UPDATE_DATABASE is forbidden in this benchmark configuration.")
        if not to_database:
            raise ValueError("UPDATE_DATABASE requires non-empty <to_database>...</to_database>.")
    elif to_database:
        raise ValueError("Mixed output is forbidden: <to_database> is allowed only with UPDATE_DATABASE.")

    if action not in set(allowed_actions):
        raise ValueError(
            "LLM predicted an action outside the allowed action space. "
            f"Got: {action!r}. Allowed: {sorted(set(allowed_actions))}\n\nRaw output:\n{raw}"
        )

    bad_term = _contains_legacy_terms(raw)
    if bad_term:
        raise ValueError(f"Legacy term '{bad_term}' is forbidden for exo-planet prompts.")

    return action, question


@dataclass
class LLMConfig:
    model_name_or_path: str = os.environ.get("CRAFTEXT_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    max_new_tokens: int = int(os.environ.get("CRAFTEXT_LLM_MAX_NEW_TOKENS", "512"))
    temperature: float = float(os.environ.get("CRAFTEXT_LLM_TEMPERATURE", "1.0"))
    top_p: float = float(os.environ.get("CRAFTEXT_LLM_TOP_P", "0.9"))


def _get_hf_llm(cfg: LLMConfig):
    key = f"hf::{cfg.model_name_or_path}"
    cached = _HF_CACHE.get(key)
    if cached is not None:
        return cached

    tok = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    _HF_CACHE[key] = (tok, model)
    return tok, model


class StopOnSubsequence(StoppingCriteria):
    def __init__(self, stop_sequences_ids: Sequence[Sequence[int]]):
        super().__init__()
        self.stop_sequences_ids = [list(s) for s in stop_sequences_ids]

    def __call__(self, input_ids, scores, **kwargs):  # type: ignore[override]
        sequence = input_ids[0].tolist()
        for stop_ids in self.stop_sequences_ids:
            if len(sequence) >= len(stop_ids) and sequence[-len(stop_ids) :] == stop_ids:
                return True
        return False


def llm_complete(prompt: str, *, cfg: Optional[LLMConfig] = None) -> str:
    cfg = cfg or LLMConfig()
    tok, model = _get_hf_llm(cfg)

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompt_lower = prompt.lower()
    expects_question = "<question>" in prompt_lower or "</question>" in prompt_lower
    stop_strings = ["</action>", "\n<reasoning>", "<reasoning>"]
    if expects_question:
        # In ask mode, let generation continue through </question> so we can return
        # both <action>ASK_OPERATOR</action> and <question>...</question>.
        stop_strings = ["</question>", "\n<reasoning>", "<reasoning>"]
    stop_sequences_ids = [tok.encode(s, add_special_tokens=False) for s in stop_strings]
    stopping_criteria = StoppingCriteriaList([StopOnSubsequence(stop_sequences_ids)])

    gen_kwargs = {
        **inputs,
        "max_new_tokens": int(cfg.max_new_tokens),
        "do_sample": float(cfg.temperature) > 0.0,
        "pad_token_id": tok.eos_token_id if tok.eos_token_id is not None else 0,
        "eos_token_id": tok.eos_token_id,
        "stopping_criteria": stopping_criteria,
    }
    if float(cfg.temperature) > 0.0:
        gen_kwargs["temperature"] = float(cfg.temperature)
        gen_kwargs["top_p"] = float(cfg.top_p)

    with torch.no_grad():
        out = model.generate(**gen_kwargs)

    gen_ids = out[0][inputs["input_ids"].shape[-1] :]
    text = tok.decode(gen_ids, skip_special_tokens=True)

    if expects_question:
        question_end_tag = "</question>"
        question_end_pos = text.find(question_end_tag)
        if question_end_pos != -1:
            text = text[: question_end_pos + len(question_end_tag)]
        else:
            # If the model chose a normal env action, keep only the action block.
            action = extract_action(text)
            if action is not None and action != "ASK_OPERATOR":
                action_end_tag = "</action>"
                action_end_pos = text.find(action_end_tag)
                if action_end_pos != -1:
                    text = text[: action_end_pos + len(action_end_tag)]
            else:
                next_reasoning_pos = text.find("<reasoning>")
                if next_reasoning_pos != -1:
                    text = text[:next_reasoning_pos]
    else:
        action_end_tag = "</action>"
        action_end_pos = text.find(action_end_tag)
        if action_end_pos != -1:
            text = text[: action_end_pos + len(action_end_tag)]
        else:
            next_reasoning_pos = text.find("<reasoning>")
            if next_reasoning_pos != -1:
                text = text[:next_reasoning_pos]
    return text.strip()


def predict_action_from_prompt(
    prompt: str,
    *,
    allow_ask_operator: bool = False,
    allow_update_database: bool = False,
    cfg: Optional[LLMConfig] = None,
) -> Tuple[str, str]:
    raw = llm_complete(prompt, cfg=cfg)
    action = extract_action(raw)
    question = extract_question(raw)
    if allow_ask_operator and action == "ASK_OPERATOR" and question is None:
        strict_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT: If you choose ASK_OPERATOR, your output MUST include "
            "both <action>ASK_OPERATOR</action> and a non-empty "
            "<question>...</question>."
        )
        raw = llm_complete(strict_prompt, cfg=cfg)
        action = extract_action(raw)
        question = extract_question(raw)
    print("--------------------------------")
    print("Raw:")
    print(raw)
    print("Action:")
    print(action)
    print("--------------------------------")
    if allow_update_database:
        allowed = DEFAULT_ALLOWED_ACTIONS_META
    elif allow_ask_operator:
        allowed = DEFAULT_ALLOWED_ACTIONS_CALL
    else:
        allowed = DEFAULT_ALLOWED_ACTIONS

    validate_response_contract(
        raw,
        allowed_actions=allowed,
        allow_ask_operator=allow_ask_operator,
        allow_update_database=allow_update_database,
    )

    return action, raw
