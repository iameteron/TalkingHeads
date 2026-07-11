from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent.parent
MEGAPROMPT_ROOT = REPO_ROOT / "MegaPrompt"
EXO_PROMPT_ROOT = MEGAPROMPT_ROOT / "exo-planet_prompt"
ARC_PROMPT_ROOT = MEGAPROMPT_ROOT / "arc_agi_prompt"
EXO_EXPERT_PROMPT_DIR = BASE_DIR / "texts" / "exo"

WORLD_MODE_CRAFTAX = "craftax"
WORLD_MODE_EXO = "exo-planet"
GAME_KIND_ARC_AGI = "arc_agi"

ALLOWED_AGENT_ACTIONS = [
    "NOOP",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "DO (comment: TO GATHER SOMETHING: WOOD | STONE | IRON | PLANT | WATER| ETC, dont use this comment in the output)",
    "SLEEP",
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
    "REST",
]

if str(MEGAPROMPT_ROOT) not in sys.path:
    sys.path.append(str(MEGAPROMPT_ROOT))
if str(EXO_PROMPT_ROOT) not in sys.path:
    sys.path.append(str(EXO_PROMPT_ROOT))

try:
    from megaprompt import Renderer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Renderer = None  # type: ignore

try:
    from actions import EXO_ALLOWED_ACTIONS_CALL, EXO_ALLOWED_ACTIONS_META  # type: ignore
except Exception:  # pragma: no cover
    EXO_ALLOWED_ACTIONS_CALL = []
    EXO_ALLOWED_ACTIONS_META = []

from oracle.knowledge import load_knowledge, use_knowledge_paths, load_knowledge_with_world_fallback

import logging

logger = logging.getLogger(__name__)


EXO_PROMPT_CONFIGS: Dict[str, Path] = {
    "exo_database_formulation": EXO_PROMPT_ROOT / "templates/database_formulation/exo-planet.yaml",
    "exo_database_formulation_deployment": EXO_PROMPT_ROOT / "templates/database_formulation_deployment/exo-planet.yaml",
    "exo_reasoning_or_ask_path": EXO_PROMPT_ROOT / "templates/reasoning_or_ask_path/exo-planet.yaml",
    "exo_reasoning_or_ask_help": EXO_PROMPT_ROOT / "templates/reasoning_or_ask_help/exo-planet.yaml",
    "exo_no_dialog": EXO_PROMPT_ROOT / "templates/no_dialog/exo-planet.yaml",
}
ARC_PROMPT_CONFIGS: Dict[str, Path] = {
    "arc_2_image": ARC_PROMPT_ROOT / "templates/2_image/arc.yaml",
    "arc_grid": ARC_PROMPT_ROOT / "templates/grid/arc.yaml",
    "arc_grid_image": ARC_PROMPT_ROOT / "templates/grid_image/arc.yaml",
    "arc_image": ARC_PROMPT_ROOT / "templates/image/arc.yaml",
}

EXO_KNOWLEDGE_JSON = EXO_PROMPT_ROOT / "knowledge_data.json"
EXO_KNOWLEDGE_TXT = EXO_PROMPT_ROOT / "knowledge_data.txt"

# Legacy megaprompt ids from before the KARA/DUSA → exploration/deployment rename.
LEGACY_MEGAPROMPT_ALIASES: Dict[str, str] = {
    "database_formulation_dusa": "database_formulation_deployment",
    "exo_database_formulation_dusa": "exo_database_formulation_deployment",
}


def _is_exo_megaprompt_config(config_name: str) -> bool:
    return str(config_name).startswith("exo_")


def _is_arc_megaprompt_config(config_name: str) -> bool:
    return str(config_name) in ARC_PROMPT_CONFIGS


CRAFTAX_TO_EXO_MEGAPROMPT: Dict[str, str] = {
    "dialog": "exo_reasoning_or_ask_help",
    "no_dialog": "exo_no_dialog",
    "reasoning_or_ask_path": "exo_reasoning_or_ask_path",
    "reasoning_or_ask_help": "exo_reasoning_or_ask_help",
    "database_formulation": "exo_database_formulation",
    "database_formulation_deployment": "exo_database_formulation_deployment",
}

EXO_TO_CRAFTAX_MEGAPROMPT: Dict[str, str] = {
    exo_name: craftax_name for craftax_name, exo_name in CRAFTAX_TO_EXO_MEGAPROMPT.items()
}

CRAFTAX_MEGAPROMPT_DEFAULTS = [
    "dialog",
    "no_dialog",
    "reasoning_or_ask_path",
    "reasoning_or_ask_help",
    "database_formulation",
    "database_formulation_deployment",
]

EXO_MEGAPROMPT_DEFAULTS = list(EXO_PROMPT_CONFIGS.keys())
ARC_MEGAPROMPT_DEFAULTS = list(ARC_PROMPT_CONFIGS.keys())


def coerce_megaprompt_config_for_world_mode(
    config_name: str,
    world_mode: str | None,
    game_kind: str | None = None,
) -> str:
    """
    Map a megaprompt config to the equivalent template for ``world_mode``.

    Unknown craftax-only configs fall back to ``database_formulation``;
    unknown exo configs fall back to ``exo_database_formulation``.
    """
    normalized = str(config_name or "").strip()
    normalized = LEGACY_MEGAPROMPT_ALIASES.get(normalized, normalized)
    if not normalized:
        normalized = "dialog"
    if str(game_kind or "").strip().lower() in {"arc", "arc_agi", "arc-agi", "arc_agi_3", "arc-agi-3"}:
        return normalized if _is_arc_megaprompt_config(normalized) else "arc_grid"
    exo_mode = is_exo_world_mode(world_mode)
    if _is_arc_megaprompt_config(normalized):
        return "exo_database_formulation" if exo_mode else "database_formulation"
    is_exo_config = _is_exo_megaprompt_config(normalized)
    if exo_mode and not is_exo_config:
        mapped = CRAFTAX_TO_EXO_MEGAPROMPT.get(normalized)
        if mapped:
            return mapped
        candidate = f"exo_{normalized}"
        if candidate in EXO_PROMPT_CONFIGS:
            return candidate
        return "exo_database_formulation"
    if not exo_mode and is_exo_config:
        mapped = EXO_TO_CRAFTAX_MEGAPROMPT.get(normalized)
        if mapped:
            return mapped
        base = normalized[4:] if normalized.startswith("exo_") else normalized
        if base in CRAFTAX_MEGAPROMPT_DEFAULTS:
            return base
        return "database_formulation"
    return normalized


def list_megaprompt_configs_for_world_mode(
    world_mode: str | None,
    game_kind: str | None = None,
) -> List[str]:
    """Return megaprompt config names available for the given world mode."""
    if str(game_kind or "").strip().lower() in {"arc", "arc_agi", "arc-agi", "arc_agi_3", "arc-agi-3"}:
        return sorted(ARC_MEGAPROMPT_DEFAULTS)
    exo_mode = is_exo_world_mode(world_mode)
    defaults = EXO_MEGAPROMPT_DEFAULTS if exo_mode else CRAFTAX_MEGAPROMPT_DEFAULTS
    options = {
        name
        for name in list_megaprompt_configs()
        if _is_exo_megaprompt_config(name) == exo_mode
    }
    options.update(defaults)
    return sorted(options)


def _renderer_for_config(config_name: str):
    if Renderer is None:
        raise RuntimeError("MegaPrompt Renderer is unavailable")
    if config_name in ARC_PROMPT_CONFIGS:
        return Renderer(config_path=ARC_PROMPT_CONFIGS[config_name])
    if config_name in EXO_PROMPT_CONFIGS:
        return Renderer(config_path=EXO_PROMPT_CONFIGS[config_name])
    return Renderer(config_name=config_name)


DEFAULT_MEGAPROMPT_CONFIGS = [
    "dialog",
    "no_dialog",
    "reasoning_or_ask_path",
    "database_formulation",
    "database_formulation_deployment",
    "exo_database_formulation",
    "exo_database_formulation_deployment",
    "exo_reasoning_or_ask_path",
    "exo_reasoning_or_ask_help",
    "exo_no_dialog",
    "arc_2_image",
    "arc_grid",
    "arc_grid_image",
    "arc_image",
]


def list_megaprompt_configs() -> List[str]:
    """
    Best-effort discovery of available MegaPrompt config names.
    Always returns defaults at minimum.
    """
    options = set(DEFAULT_MEGAPROMPT_CONFIGS)
    templates_dir = MEGAPROMPT_ROOT / "craftext_prompt" / "templates"
    exo_templates_dir = EXO_PROMPT_ROOT / "templates"

    # Try to read config names exposed by the Renderer API.
    if Renderer is not None:
        for attr_name in (
            "list_configs",
            "get_available_configs",
            "available_configs",
            "config_names",
            "AVAILABLE_CONFIGS",
            "CONFIGS",
            "SUPPORTED_CONFIGS",
        ):
            try:
                attr = getattr(Renderer, attr_name, None)
                if attr is None:
                    continue
                value = attr() if callable(attr) else attr
                if isinstance(value, (list, tuple, set)):
                    options.update(str(x).strip() for x in value if str(x).strip())
            except Exception:
                continue

    # Primary source: template directories in MegaPrompt checkout.
    # Example: MegaPrompt/craftext_prompt/templates/dialog
    if templates_dir.exists() and templates_dir.is_dir():
        for item in templates_dir.iterdir():
            if item.is_dir():
                name = item.name.strip()
                if name:
                    options.add(name)

    if exo_templates_dir.exists() and exo_templates_dir.is_dir():
        for key in EXO_PROMPT_CONFIGS:
            options.add(key)

    for key in ARC_PROMPT_CONFIGS:
        options.add(key)

    # Fallback: discover config-like files in MegaPrompt repository checkout.
    # This keeps the UI in sync even if renderer API changes.
    if MEGAPROMPT_ROOT.exists():
        for pattern in ("**/*config*.yaml", "**/*config*.yml", "**/*config*.json"):
            for cfg_path in MEGAPROMPT_ROOT.glob(pattern):
                stem = cfg_path.stem.strip()
                if stem:
                    options.add(stem)

    return sorted(options)


def compute_previous_actions_analysis(
    actions_history: List[str],
    min_repeats: int = 2,
) -> str:
    """
    If the same action pattern was repeated in the last min_repeats (or more) ticks,
    return a message for the "Previous actions analysis" prompt section.
    Otherwise return an empty string.
    """
    if len(actions_history) < min_repeats:
        return ""
    last_n = actions_history[-min_repeats:]
    pattern = last_n[0].strip()
    if not pattern:
        return ""
    if not all(s.strip() == pattern for s in last_n):
        return ""
    count = min_repeats
    for i in range(len(actions_history) - min_repeats - 1, -1, -1):
        if actions_history[i].strip() == pattern:
            count += 1
        else:
            break
    return (
        f"In previous ticks I already made actions \"{pattern}\" (repeated {count} time(s)) "
        "and it did not lead to progress. I need to ask the Operator or change actions."
    )


def build_previous_actions_analysis(
    actions_history: List[str],
    consecutive_questions_count: int = 0,
    min_action_repeats: int = 2,
    max_consecutive_questions: int = 5,
) -> str:
    """
    Build the full "Previous actions analysis" text from repeated actions and/or too many questions.
    Returns combined message (may be empty).
    """
    parts: List[str] = []
    repeated_msg = compute_previous_actions_analysis(actions_history, min_repeats=min_action_repeats)
    if repeated_msg:
        parts.append(repeated_msg)
    if consecutive_questions_count > max_consecutive_questions:
        parts.append(
            "I already ask too much questions, I need to predict an actions."
        )
    return "\n\n".join(parts) if parts else ""


def _arc_action_descriptions(available_actions: Any) -> List[str]:
    raw_actions = available_actions or []
    actions = [str(action).strip().upper() for action in raw_actions if str(action).strip()]
    descriptions = {
        "ACTION1": "ACTION1: Up arrow on the game controller.",
        "ACTION2": "ACTION2: Down arrow on the game controller.",
        "ACTION3": "ACTION3: Left arrow on the game controller.",
        "ACTION4": "ACTION4: Right arrow on the game controller.",
        "ACTION5": "ACTION5: Spacebar / interact / select button when available.",
        "ACTION6": "ACTION6 x y: click on the game frame at coordinate x,y. Use exactly `ACTION6 32 31` or `ACTION6 [32,31]`; x and y must be integers in 0..63.",
        "ACTION7": "ACTION7: undo when available",
    }
    return [descriptions.get(action, action) for action in actions]


def _arc_repeated_action_tip(
    action_history: List[str] | None,
    *,
    min_repeats: int = 4,
) -> str:
    actions = [" ".join(str(action or "").strip().upper().split()) for action in (action_history or [])]
    actions = [action for action in actions if action]
    if len(actions) < min_repeats:
        return ""
    repeated_action = actions[-1]
    count = 0
    for action in reversed(actions):
        if action != repeated_action:
            break
        count += 1
    if count < min_repeats:
        return ""
    return (
        f"TIP!!!! You have executed {repeated_action} for {count} consecutive environment actions. "
        "This may mean you are stuck, not moving, or not making progress. Strongly consider asking "
        "the human operator about your current progress or which actions can get you out of this "
        "problem situation instead of repeating the same action again."
    )


def generate_arc_agent_prompt(
    *,
    goal: str,
    arc_observation: Dict[str, Any],
    dialog: List[Dict[str, Any]] | str | None = None,
    action_history: List[str] | None = None,
    previous_reasoning: str = "",
    current_step: int = 1,
    megaprompt_config_name: str = "arc_grid",
    operator_call_budget: str = "",
) -> str:
    """
    Generate active-agent prompt for ARC-AGI-3 games via the ARC MegaPrompt family.

    ``arc_observation`` is a structured dict derived from ARC ``FrameDataRaw``:
    metadata, available actions, a 64x64 frame grid, and a PNG data URL payload.
    """
    config_name = coerce_megaprompt_config_for_world_mode(
        megaprompt_config_name,
        WORLD_MODE_CRAFTAX,
        game_kind=GAME_KIND_ARC_AGI,
    )
    action_descriptions = _arc_action_descriptions(arc_observation.get("available_actions"))
    repeat_action_tip = _arc_repeated_action_tip(action_history)
    meta: Dict[str, Any] = {
        "goal": goal,
        "obs": arc_observation,
        "dialog": dialog or [],
        "action_history": action_history or [],
        "previous_reasoning": previous_reasoning,
        "repeat_action_tip": repeat_action_tip,
        "current_step": current_step,
        "act": action_descriptions,
    }
    if operator_call_budget.strip():
        meta["operator_call_budget"] = operator_call_budget.strip()
    try:
        return _renderer_for_config(config_name).render(meta)
    except Exception as exc:
        logger.warning("ARC MegaPrompt render failed for %s: %s", config_name, exc)
        from megaprompt.obs.arc_grid import render as render_arc_grid  # type: ignore

        return "\n".join(
            [
                "# Agent Instructions",
                "",
                "You are playing an ARC-AGI-3 interactive game.",
                "",
                "## Current goal",
                goal,
                "",
                "## Current step",
                str(max(1, int(current_step or 1))),
                "",
                "## Available actions",
                "\n".join(f"- {item}" for item in action_descriptions),
                "",
                "## Current observation",
                render_arc_grid(arc_observation),
                "",
                "## Recent action history",
                "\n".join(f"{idx}. {action}" for idx, action in enumerate((action_history or [])[-30:], 1))
                or "No actions have been executed yet.",
                "",
                "## Previous agent reasoning",
                previous_reasoning.strip() or "No previous reasoning is available.",
                "",
                "## Repeated action tip",
                repeat_action_tip,
            ]
        )


def parse_operator_qa_transcript(message_from_operator: str) -> List[Dict[str, str]]:
    """
    Parse the Q:/A: transcript produced by Session._format_chat_history_for_agent
    into dialog turns for MegaPrompt's last_five renderer.

    Each turn looks like::

        Q: <question>
        A: <answer>

    Turns are separated by a blank line. Any other shape returns [] so callers
    can fall back to treating the message as a single blob.
    """
    text = message_from_operator.strip().replace("\r\n", "\n")
    if not text.startswith("Q:"):
        return []
    turns: List[Dict[str, str]] = []
    pos = 0
    n = len(text)
    while pos < n:
        m = re.match(r"Q:\s*", text[pos:])
        if not m:
            return []
        pos += m.end()
        q_end = text.find("\nA:", pos)
        if q_end == -1:
            return []
        q = text[pos:q_end].strip()
        pos = q_end + len("\nA:")
        ws = re.match(r"\s*", text[pos:])
        if ws:
            pos += ws.end()
        next_block = text.find("\n\nQ:", pos)
        if next_block == -1:
            a = text[pos:].strip()
            if q or a:
                turns.append({"question": q, "answer": a})
            break
        a = text[pos:next_block].strip()
        if q or a:
            turns.append({"question": q, "answer": a})
        pos = next_block + 2
    return turns


def _build_megaprompt_dialog(
    message_from_operator: str,
    operator_context: str,
) -> List[Dict[str, Any]]:
    """
    Build dialog items for MegaPrompt last_five: Agent = question, Operator = answer.

    When message_from_operator is a Q:/A: transcript (play_web), split into turns
    so oracle replies are not merged into the Agent line. Inventory / hints /
    previous-actions context is appended to the last Operator reply.
    """
    msg = message_from_operator.strip()
    ctx = operator_context.strip()
    turns = parse_operator_qa_transcript(msg)
    if turns:
        if ctx:
            last = turns[-1]
            prev_a = str(last.get("answer", "")).strip()
            last["answer"] = f"{prev_a}\n\n{ctx}".strip() if prev_a else ctx
        return turns
    if msg or ctx:
        return [{"question": msg, "answer": ctx}]
    return []


def normalize_world_mode(world_mode: str | None) -> str:
    token = str(world_mode or WORLD_MODE_CRAFTAX).strip().lower()
    if token in {"exo", "exo-planet", "exo_planet"}:
        return WORLD_MODE_EXO
    return WORLD_MODE_CRAFTAX


def is_exo_world_mode(world_mode: str | None) -> bool:
    return normalize_world_mode(world_mode) == WORLD_MODE_EXO


def _prompt_template_path(relative_template_path: str, *, world_mode: str | None = None) -> Path:
    rel = str(relative_template_path).replace("\\", "/").lstrip("/")
    if rel.startswith("texts/"):
        rel = rel[len("texts/") :]
    if is_exo_world_mode(world_mode):
        exo_path = EXO_EXPERT_PROMPT_DIR / Path(rel).name
        if exo_path.is_file():
            return exo_path
    return BASE_DIR / "texts" / rel


def load_prompt(
    relative_template_path: str,
    params: Dict[str, Any],
    *,
    world_mode: str | None = None,
) -> str:
    """
    Load a text prompt template and substitute placeholder tokens.

    When ``world_mode`` is exo-planet, loads from ``texts/exo/`` if present.
    """
    template_path = _prompt_template_path(relative_template_path, world_mode=world_mode)
    text = template_path.read_text(encoding="utf-8")

    for key, value in params.items():
        text = text.replace(str(key), str(value))

    return text


def generate_agent_prompt(
    goal: str,
    observation: Any,
    message_from_operator: str,
    inventory: str = "",
    hints: str = "",
    previous_actions_analysis: str = "",
    action_history: List[str] | None = None,
    state_history: List[Any] | None = None,
    megaprompt_config_name: str = "dialog",
    world_mode: str | None = None,
    operator_call_budget: str = "",
) -> str:
    """
    Generate active-agent prompt.

    Preferred path: render with MegaPrompt (if available and observation is EnvState-like).
    Fallback path: render with local `agent_prompt.txt` template.

    Parameters
    ----------
    goal: str
        The current goal for the agent.
    observation: str
        The latest observation text describing the environment.
    message_from_operator: str
        Any additional message or hint from the remote operator.
    inventory: str
        Human-readable description of the agent's current inventory.
    hints: str
        Optional hints for the agent.
    previous_actions_analysis: str
        If non-empty, text for the "Previous actions analysis" section (e.g. when
        the agent repeated the same actions without progress).
    action_history: list[str] | None
        List of the most recent environment actions. For prompts that support it,
        this history is rendered as "actions after the latest question".
    state_history: list[Any] | None
        Only for ``reasoning_or_ask_help``: at most one transition dict
        ``{before, after, action}`` — the last primitive step only (two states + action).
    megaprompt_config_name: str
        MegaPrompt template config name (e.g. "dialog", "no_dialog",
        "reasoning_or_ask_path", "database_formulation_deployment").
    operator_call_budget: str
        Pre-rendered deployment operator call budget text for database_formulation_deployment
        templates; ignored for other configs.
    world_mode: str | None
        Active play texture theme (``craftax`` or ``exo-planet``). When set, selects
        the matching megaprompt template and fallback ``agent_prompt.txt``.
    """
    resolved_world_mode = normalize_world_mode(
        world_mode
        if world_mode is not None
        else (WORLD_MODE_EXO if _is_exo_megaprompt_config(megaprompt_config_name) else WORLD_MODE_CRAFTAX)
    )
    megaprompt_config_name = coerce_megaprompt_config_for_world_mode(
        megaprompt_config_name,
        resolved_world_mode,
    )
    section = ""
    if previous_actions_analysis.strip():
        section = (
            "### Previous actions analysis\n\n"
            f"{previous_actions_analysis.strip()}\n\n"
        )

    # MegaPrompt requires EnvState-like observation object.
    can_use_megaprompt = (
        Renderer is not None
        and observation is not None
        and hasattr(observation, "map")
        and hasattr(observation, "player_position")
    )
    if can_use_megaprompt:
        context_lines: List[str] = []
        if inventory.strip():
            context_lines.append(f"Inventory: {inventory.strip()}")
        if hints.strip():
            context_lines.append(f"Hints:\n{hints.strip()}")
        if section.strip():
            context_lines.append(section.strip())
        operator_context = "\n\n".join(context_lines).strip()
        dialog = _build_megaprompt_dialog(message_from_operator, operator_context)

        try:
            renderer = _renderer_for_config(megaprompt_config_name)
            is_exo = _is_exo_megaprompt_config(megaprompt_config_name)
            meta: Dict[str, Any] = {
                "goal": goal,
                "obs": observation,
                "dialog": dialog,
                "act": EXO_ALLOWED_ACTIONS_CALL if is_exo else ALLOWED_AGENT_ACTIONS,
                "action_history": action_history or [],
            }
            if megaprompt_config_name in {"reasoning_or_ask_help", "exo_reasoning_or_ask_help"}:
                meta["state_history"] = state_history or []
            if megaprompt_config_name in {
                "database_formulation",
                "exo_database_formulation",
                "database_formulation_deployment",
                "exo_database_formulation_deployment",
            }:
                meta["knowledge"] = load_knowledge_with_world_fallback(resolved_world_mode)
                meta["state_history"] = state_history or []
                meta["emergency_message"] = {
                    "dialog": dialog,
                    "action_history": action_history or [],
                }
            if megaprompt_config_name in {"database_formulation_deployment", "exo_database_formulation_deployment"}:
                meta["operator_call_budget"] = operator_call_budget.strip() or (
                    "Operator call budget is not configured for this task."
                )
            return renderer.render(meta)
        except Exception as exc:
            logger.warning(
                "MegaPrompt render failed for %s (%s): %s",
                megaprompt_config_name,
                resolved_world_mode,
                exc,
            )

    return load_prompt(
        "texts/agent_prompt.txt",
        {
            "GOAL": goal,
            "OBS": str(observation),
            "MSG": message_from_operator,
            "INVENTORY": inventory,
            "HINTS": hints,
            "PREVIOUS_ACTIONS_ANALYSIS": section,
        },
        world_mode=resolved_world_mode,
    )


def create_map_prompt(question: str, *, world_mode: str | None = None) -> str:
    return load_prompt(
        "map_prompt.txt",
        {"QUESTION": question},
        world_mode=world_mode,
    )


def create_mechanics_promt(question: str, *, world_mode: str | None = None) -> str:
    return load_prompt(
        "mechanics_prompt.txt",
        {"QUESTION": question},
        world_mode=world_mode,
    )


def create_action_prompt(
    question: str,
    inventory: str = "",
    *,
    world_mode: str | None = None,
) -> str:
    return load_prompt(
        "action_prompt.txt",
        {"INVENTORY": inventory, "QUESTION": question},
        world_mode=world_mode,
    )


def create_question_prompt(goal: str, *, world_mode: str | None = None) -> str:
    return load_prompt(
        "question_prompt.txt",
        {"GOAL": goal},
        world_mode=world_mode,
    )


def create_path_prompt(
    question: str,
    agent_position: str = "not provided",
    target_location: str = "not provided",
    infer_goal_from_question: bool = False,
    *,
    world_mode: str | None = None,
) -> str:
    if infer_goal_from_question:
        return load_prompt(
            "path_prompt_infer_goal_from_question.txt",
            {
                "QUESTION": question,
                "AGENT_POSITION": agent_position,
            },
            world_mode=world_mode,
        )
    return load_prompt(
        "path_prompt.txt",
        {
            "QUESTION": question,
            "AGENT_POSITION": agent_position,
            "TARGET_LOCATION": target_location,
        },
        world_mode=world_mode,
    )


def create_path_expert_prompt(
    question: str,
    rendered_map: str,
    agent_position: str = "not provided",
    target_location: str = "not provided",
    colorful_waypoint: str = "",
    *,
    world_mode: str | None = None,
) -> str:
    return load_prompt(
        "path_expert_prompt.txt",
        {
            "QUESTION": question,
            "PATH_OBSERVATION": rendered_map,
            "AGENT_POSITION": agent_position,
            "TARGET_LOCATION": target_location,
            "COLORFUL_WAYPOINT": colorful_waypoint,
        },
        world_mode=world_mode,
    )


def create_goal_prompt(
    goal: str,
    question_1: str,
    answer_1: str,
    question_2: str,
    answer_2: str,
    action_answer: str = "",
    agent_position: str = "not provided",
    *,
    world_mode: str | None = None,
) -> str:
    """
    Create the goal aggregation prompt by loading the template and substituting
    the agent question, agent position, and sub-expert questions and answers.

    The ``goal`` parameter is the agent's question routed to the goal expert.
    Handles empty answers gracefully by providing fallback text.
    """
    # Handle empty answers
    if not answer_1.strip():
        answer_1 = "No specific information available."
    if not answer_2.strip():
        answer_2 = "No specific information available."
    if not action_answer.strip():
        action_answer = "No specific action guidance available."

    agent_question = goal.strip() or "N/A"

    return load_prompt(
        "goal_prompt.txt",
        {
            "QUESTION": agent_question,
            "AGENT_POSITION": agent_position,
            "QUESTION_1": question_1 if question_1.strip() else "N/A",
            "ANSWER_1": answer_1,
            "QUESTION_2": question_2 if question_2.strip() else "N/A",
            "ANSWER_2": answer_2,
            "ACTION_ANSWER": action_answer,
        },
        world_mode=world_mode,
    )
