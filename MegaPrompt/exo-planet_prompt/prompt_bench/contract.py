from __future__ import annotations

import re
from typing import Optional, Sequence

from actions import EXO_ALLOWED_ACTIONS_CALL, EXO_ALLOWED_ACTIONS_META, EXO_ENV_ACTIONS

_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE)
_QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL | re.IGNORECASE)
_TO_DATABASE_RE = re.compile(r"<to_database>\s*(.*?)\s*</to_database>", re.DOTALL | re.IGNORECASE)

EXO_LEGACY_TERMS: tuple[str, ...] = (
    "craftax",
    "minecraft",
    "crafting table",
    "pickaxe",
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "PLACE_TABLE",
    "PLACE_STONE",
    "MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE",
    "MAKE_IRON_PICKAXE",
    "DO (TO GATHER SOMETHING)",
    "zombie",
    "skeleton",
    "cow",
)


def extract_action(text: str) -> Optional[str]:
    if not text:
        return None
    m = _ACTION_RE.search(text)
    if not m:
        return None
    action = (m.group(1) or "").strip().replace("\n", " ").strip()
    return action or None


def extract_question(text: str) -> Optional[str]:
    if not text:
        return None
    m = _QUESTION_RE.search(text)
    if not m:
        return None
    return (m.group(1) or "").strip() or None


def extract_to_database(text: str) -> Optional[str]:
    if not text:
        return None
    m = _TO_DATABASE_RE.search(text)
    if not m:
        return None
    return (m.group(1) or "").strip() or None


def _contains_legacy_terms(text: str) -> Optional[str]:
    lowered = str(text or "").lower()
    for term in EXO_LEGACY_TERMS:
        if term.lower() in lowered:
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
            "Action outside exo-planet action space. "
            f"Got: {action!r}. Allowed: {sorted(set(allowed_actions))}\n\nRaw output:\n{raw}"
        )

    bad_term = _contains_legacy_terms(raw)
    if bad_term:
        raise ValueError(f"Legacy term '{bad_term}' is forbidden for exo-planet prompts.")

    return action, question


def allowed_actions_for(
    *,
    allow_ask_operator: bool,
    allow_update_database: bool,
) -> list[str]:
    if allow_update_database:
        return list(EXO_ALLOWED_ACTIONS_META)
    if allow_ask_operator:
        return list(EXO_ALLOWED_ACTIONS_CALL)
    return list(EXO_ENV_ACTIONS)
