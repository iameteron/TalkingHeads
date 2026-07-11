"""
Helpers for active agent in play_web: observation formatting, answer parsing, action execution.
Uses craftax_classic constants (same as Craftax-Classic-Symbolic-v1).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from craftax.craftax_classic.constants import Action
from oracle.knowledge import apply_knowledge_from_response
from oracle.utils.observation_formatting import (
    format_observation_from_env_state as _format_observation_from_env_state,
    render_symbolic_map_from_env_state,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MEGAPROMPT_ROOT = _REPO_ROOT / "MegaPrompt"
if str(_MEGAPROMPT_ROOT) not in sys.path:
    sys.path.append(str(_MEGAPROMPT_ROOT))
try:
    from megaprompt.obs.map_and_coords import render as _render_megaprompt_observation  # type: ignore
    from megaprompt.obs.map_and_coords_exo import render as _render_megaprompt_observation_exo  # type: ignore
except Exception:
    _render_megaprompt_observation = None
    _render_megaprompt_observation_exo = None

# EnvState from craftax_classic has map and player_position
EnvState = Any


def render_symbolic_map(
    state: EnvState,
    player_symbol: str = "P",
    show_top_axis: bool = True,
    show_bottom_axis: bool = False,
    radius: int = 5,
) -> str:
    """Render local map with tiles and mobs (zombies, cows, skeletons)."""
    return render_symbolic_map_from_env_state(
        state,
        player_symbol=player_symbol,
        show_top_axis=show_top_axis,
        show_bottom_axis=show_bottom_axis,
        radius=radius,
    )


def format_inventory_from_state(state: EnvState) -> str:
    """
    Build text description of the agent's inventory.
    Only lists items with count > 0.
    """
    inv = state.inventory
    items: List[str] = []
    if inv.wood > 0:
        items.append(f"wood: {int(inv.wood)}")
    if inv.stone > 0:
        items.append(f"stone: {int(inv.stone)}")
    if inv.coal > 0:
        items.append(f"coal: {int(inv.coal)}")
    if inv.iron > 0:
        items.append(f"iron: {int(inv.iron)}")
    if inv.diamond > 0:
        items.append(f"diamond: {int(inv.diamond)}")
    if inv.sapling > 0:
        items.append(f"sapling: {int(inv.sapling)}")
    if inv.wood_pickaxe > 0:
        items.append(f"wood_pickaxe: {int(inv.wood_pickaxe)}")
    if inv.stone_pickaxe > 0:
        items.append(f"stone_pickaxe: {int(inv.stone_pickaxe)}")
    if inv.iron_pickaxe > 0:
        items.append(f"iron_pickaxe: {int(inv.iron_pickaxe)}")
    if inv.wood_sword > 0:
        items.append(f"wood_sword: {int(inv.wood_sword)}")
    if inv.stone_sword > 0:
        items.append(f"stone_sword: {int(inv.stone_sword)}")
    if inv.iron_sword > 0:
        items.append(f"iron_sword: {int(inv.iron_sword)}")
    if not items:
        return "Empty"
    return ", ".join(items)


def format_observation_from_state(state: EnvState, k: int = 5, radius: int = 5) -> str:
    """Build text observation including tiles and mobs (zombies, cows, skeletons)."""
    return _format_observation_from_env_state(state, k=k, radius=radius)


def format_agent_observation_text_from_state(
    state: EnvState,
    k: int = 5,
    radius: int = 5,
    *,
    world_mode: str | None = None,
) -> str:
    """
    Build the exact text observation shown to the agent in TalkingHeads.
    """
    from oracle.prompts.prompt_generation import is_exo_world_mode
    from oracle.utils.observation_formatting import format_inventory_from_env_state

    exo_mode = is_exo_world_mode(world_mode)
    render_fn = _render_megaprompt_observation_exo if exo_mode else _render_megaprompt_observation
    if render_fn is not None and state is not None:
        try:
            megaprompt_obs = str(render_fn(state)).strip()
            if megaprompt_obs:
                if exo_mode:
                    inv = format_inventory_from_env_state(state, world_mode=world_mode)
                    return f"{megaprompt_obs}\n\n## Your inventory\n{inv}"
                return megaprompt_obs
        except Exception:
            pass
    obs = format_observation_from_state(state, k=k, radius=radius)
    inv = (
        format_inventory_from_env_state(state, world_mode=world_mode)
        if exo_mode
        else format_inventory_from_state(state)
    )
    return f"{obs}\n\n## Your inventory\n{inv}"


def _assign_parsed_action(result: Dict[str, Any], raw_content: str) -> None:
    raw = str(raw_content or "").strip()
    result["action_raw"] = raw
    result["action"] = _sanitize_action_content(raw)


def format_action_for_ui(
    parsed: Dict[str, Any],
    *,
    world_mode: str | None = None,
) -> str:
    """Return the action string shown in chat / action history for the active world mode."""
    from oracle.prompts.prompt_generation import is_exo_world_mode

    engine_action = str(parsed.get("action") or "").strip()
    raw_action = str(parsed.get("action_raw") or engine_action).strip()
    if not is_exo_world_mode(world_mode):
        return engine_action or raw_action
    try:
        exo_root = _MEGAPROMPT_ROOT / "exo-planet_prompt"
        if str(exo_root) not in sys.path:
            sys.path.insert(0, str(exo_root))
        from action_bridge import to_display_action  # type: ignore

        return to_display_action(raw_action, engine_action)
    except Exception:
        return raw_action or engine_action


def _sanitize_action_content(content: str) -> str:
    """
    Normalize the action string produced by the agent so it is parsable by Craftax.

    Maps exo-planet action tokens to engine actions, then strips DO helper hints.
    """
    cleaned = content.strip().upper()
    try:
        exo_root = _MEGAPROMPT_ROOT / "exo-planet_prompt"
        if str(exo_root) not in sys.path:
            sys.path.insert(0, str(exo_root))
        from action_bridge import to_engine_action  # type: ignore

        cleaned = to_engine_action(cleaned)
    except Exception:
        cleaned = content
    # Allow both variants: with and without closing parenthesis.
    patterns = [
        r"\(TO GATHER SOMETHING\)?",
        r"\(TO FIGHT\)?",
        r"\(DRINK WATER\)?",
    ]
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    # Collapse extra whitespace.
    return " ".join(cleaned.split())


_STRUCTURED_OUTPUT_TAG = re.compile(
    r"<\s*/?\s*(?:action|act|question|ask|q|to_database)\b",
    flags=re.IGNORECASE,
)


def _strip_orphan_reasoning_markers(text: str) -> str:
    """Remove stray </reasoning> left when the opening tag was omitted or stripped."""
    return re.sub(r"</\s*reasoning\s*>", "", text, flags=re.IGNORECASE).strip()


def _extract_dash_block(text: str, label: str) -> str:
    match = re.search(
        rf"---\s*{re.escape(label)}\s*---\s*(.*?)(?:---\s*{re.escape(label)}\s*---|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match and match.group(1).strip() else ""


def _extract_arc_action_candidate(text: str) -> str:
    match = re.search(
        r"\bACTION[1-7]\b(?:\s*(?:\[\s*\d{1,2}\s*,\s*\d{1,2}\s*\]|\d{1,2}\s+\d{1,2}))?",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(0).strip().upper() if match else ""


def _split_reasoning_and_body(text: str) -> tuple[str, str]:
    """
    Extract optional <reasoning> block and return (reasoning_text, remainder_for_action_parse).
    Unclosed <reasoning> stops at the next structured tag so <action>/<question> are preserved.
    """
    if "--- REASONING ---" in text:
        parts = text.split("--- REASONING ---")
        if len(parts) >= 3:
            reasoning = parts[1].strip()
            suffix = "--- REASONING ---".join(parts[2:]).strip()
            return reasoning, suffix
        return "", text

    closed = re.search(
        r"<reasoning>\s*(.*?)\s*</reasoning>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if closed and closed.group(1).strip():
        body = (text[: closed.start()] + text[closed.end() :]).strip()
        return closed.group(1).strip(), _strip_orphan_reasoning_markers(body)

    open_tag = re.search(r"<reasoning>\s*", text, flags=re.IGNORECASE)
    if not open_tag:
        return "", _strip_orphan_reasoning_markers(text)

    tail = text[open_tag.end() :]
    next_tag = _STRUCTURED_OUTPUT_TAG.search(tail)
    if next_tag:
        reasoning = tail[: next_tag.start()].strip()
        body = tail[next_tag.start() :].strip()
        return reasoning, _strip_orphan_reasoning_markers(body)

    return tail.strip(), ""


def _extract_xml_question(text: str) -> str:
    match = re.search(
        r"<(?:question|ask|q)>\s*(.*?)\s*</(?:question|ask|q)>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match and match.group(1).strip() else ""


def _extract_ask_operator_question(text: str, reasoning: str = "") -> str:
    """
    Recover operator question when the model declared ASK_OPERATOR but omitted <question>.
    """
    for candidate in (text, reasoning):
        if not candidate:
            continue
        q = _extract_xml_question(candidate)
        if q:
            return q
        after_action = re.search(
            r"</\s*(?:action|act)\s*>\s*(.+)",
            candidate,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if after_action:
            tail = _strip_orphan_reasoning_markers(after_action.group(1).strip())
            tail = re.sub(
                r"^<\s*(?:action|act)\s*>.*?</\s*(?:action|act)\s*>\s*",
                "",
                tail,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
            if tail and not re.fullmatch(r"<\s*/?\s*\w+[^>]*>", tail, flags=re.IGNORECASE):
                return tail
        for line in candidate.splitlines():
            s = line.strip()
            if s.upper().startswith("QUESTION:"):
                body = s.split(":", 1)[1].strip()
                if body:
                    return body
        ask_q = _extract_question_from_ask_operator(candidate)
        if ask_q:
            return ask_q
        compact = " ".join(candidate.split())
        if compact and "ASK_OPERATOR" not in compact.upper():
            if "?" in compact or len(compact.split()) >= 6:
                return compact
    return ""


def _extract_question_from_ask_operator(content: str) -> str:
    """
    Convert ASK_OPERATOR action payload into plain question text.
    Examples:
      - "ASK_OPERATOR Where is coal?" -> "Where is coal?"
      - "ASK_OPERATOR: Where is coal?" -> "Where is coal?"
    """
    normalized = str(content or "").strip()
    if not normalized:
        return ""
    if not re.match(r"^\s*ASK_OPERATOR\b", normalized, flags=re.IGNORECASE):
        return ""
    question = re.sub(
        r"^\s*ASK_OPERATOR\s*[:\-]?\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip()
    return question


def parse_agent_answer(raw_answer: str) -> Dict[str, Any]:
    """
    Parse ActiveAgent answer. Returns one of:
    - {"action": "<ACTION STRING>", "reasoning": "<TEXT>?", ...}
    - {"question": "<QUESTION STRING>", "reasoning": "<TEXT>?", ...}
    - {}

    When the model emits <to_database>, blocks are merged into knowledge_data.json
    and knowledge_data.txt is re-rendered as a markdown table
    and optional keys are set: ``to_database`` (this turn's text), ``knowledge_updated`` (bool).
    """
    if not isinstance(raw_answer, str):
        raw_answer = str(raw_answer)
    text, knowledge_updated, to_database_block = apply_knowledge_from_response(raw_answer)
    text = text.strip()
    result: Dict[str, Any] = {}
    if to_database_block:
        result["to_database"] = to_database_block
    if knowledge_updated:
        result["knowledge_updated"] = True
    reasoning_block, text_without_reasoning = _split_reasoning_and_body(text)
    if reasoning_block:
        result["reasoning"] = reasoning_block

    # Remove legacy reasoning block from content considered as question/action.
    text_without_reasoning = re.sub(
        r"---\s*REASONING\s*---.*?---\s*REASONING\s*---",
        "",
        text_without_reasoning,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    act_block = _extract_dash_block(text_without_reasoning, "Act")
    if act_block:
        ask_q = _extract_question_from_ask_operator(act_block)
        if ask_q:
            result["question"] = ask_q
            return result
        _assign_parsed_action(result, act_block)
        return result

    q_block = _extract_dash_block(text_without_reasoning, "Q")
    if q_block:
        result["question"] = q_block
        return result

    # Megaprompt/XML-like formats (strict handling first):
    # <action>ASK_OPERATOR</action> + <question>...</question>
    action_tag_match = re.search(
        r"<(?:action|act)>\s*(.*?)\s*</(?:action|act)>",
        text_without_reasoning,
        flags=re.IGNORECASE | re.DOTALL,
    )
    question_tag_match = re.search(
        r"<(?:question|ask|q)>\s*(.*?)\s*</(?:question|ask|q)>",
        text_without_reasoning,
        flags=re.IGNORECASE | re.DOTALL,
    )
    action_tag_value = action_tag_match.group(1).strip() if action_tag_match and action_tag_match.group(1).strip() else ""
    question_tag_value = question_tag_match.group(1).strip() if question_tag_match and question_tag_match.group(1).strip() else ""

    if action_tag_value:
        if action_tag_value.upper() == "ASK_OPERATOR":
            question = question_tag_value or _extract_ask_operator_question(
                text_without_reasoning,
                reasoning=reasoning_block,
            )
            if not question:
                question = _extract_ask_operator_question(text, reasoning=reasoning_block)
            if question:
                result["question"] = question
            return result
        _assign_parsed_action(result, action_tag_value)
        return result

    if question_tag_value:
        result["question"] = question_tag_value
        return result

    # Fallback line-based formats.
    for line in text_without_reasoning.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.upper().startswith("ACTION:"):
            candidate = s.split(":", 1)[1].strip()
            if candidate:
                ask_q = _extract_question_from_ask_operator(candidate)
                if ask_q:
                    result["question"] = ask_q
                    return result
                _assign_parsed_action(result, candidate)
                return result
        if s.upper().startswith("QUESTION:"):
            candidate = s.split(":", 1)[1].strip()
            if candidate:
                result["question"] = candidate
                return result

    # Last-resort fallback for plain-text outputs:
    # - if text contains clear action tokens, execute those
    # - otherwise treat as a question for operator/oracle flow
    compact = " ".join(text_without_reasoning.split())
    if compact:
        arc_action = _extract_arc_action_candidate(compact)
        if arc_action:
            _assign_parsed_action(result, arc_action)
            return result
        tokens = [tok for tok in re.split(r"[^A-Za-z0-9_]+", compact.upper()) if tok]
        action_tokens = [tok for tok in tokens if tok in Action.__members__]
        if action_tokens:
            _assign_parsed_action(result, " ".join(action_tokens))
            return result
        result["question"] = compact

    return result
