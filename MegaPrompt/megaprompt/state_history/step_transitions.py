from __future__ import annotations

from typing import Any, Dict, List, Mapping

EnvState = Any

_EMPTY_MESSAGE = (
    "No step effects recorded yet (first tick after reset, after a new goal, "
    "after asking the operator, or no environment actions were applied)."
)


def _inventory_count_map(state: EnvState) -> Dict[str, int]:
    inv = state.inventory
    return {
        "wood": int(inv.wood),
        "stone": int(inv.stone),
        "coal": int(inv.coal),
        "iron": int(inv.iron),
        "diamond": int(inv.diamond),
        "sapling": int(inv.sapling),
        "wood_pickaxe": int(inv.wood_pickaxe),
        "stone_pickaxe": int(inv.stone_pickaxe),
        "iron_pickaxe": int(inv.iron_pickaxe),
        "wood_sword": int(inv.wood_sword),
        "stone_sword": int(inv.stone_sword),
        "iron_sword": int(inv.iron_sword),
    }


def _line_for_transition(before: EnvState, after: EnvState, action_token: str) -> str:
    """
    Compare two environment observations (map position + inventory) and the
    single primitive action that was applied between them.
    """
    act = action_token.strip().upper()
    prev_pos = [int(x) for x in before.player_position.tolist()]
    new_pos = [int(x) for x in after.player_position.tolist()]
    prev_inv = _inventory_count_map(before)
    new_inv = _inventory_count_map(after)

    inv_parts: List[str] = []
    for key in sorted(set(prev_inv) | set(new_inv)):
        delta = new_inv.get(key, 0) - prev_inv.get(key, 0)
        if delta == 0:
            continue
        label = key.upper()
        if delta > 0:
            inv_parts.append(f"{label} +{delta}")
        else:
            inv_parts.append(f"{label} {delta}")

    if inv_parts:
        return (
            f"In the previous step you took action {act} and it changed your inventory: "
            f"{', '.join(inv_parts)}."
        )

    if prev_pos != new_pos:
        return (
            f"In the previous step you took action {act} and your coordinates changed "
            f"from {prev_pos} to {new_pos}."
        )

    if act in {"LEFT", "RIGHT", "UP", "DOWN"}:
        return (
            f"In the previous step you took action {act} and nothing changed "
            f"(you stayed at {new_pos}). You may be blocked by an obstacle or the map edge."
        )

    if act == "DO":
        return (
            "In the previous step you took action DO and nothing changed in your position or inventory. "
            "You might not be facing the right object, you may be too far away, or you may lack the right "
            "tools for this interaction."
        )

    if act.startswith("PLACE_") or act.startswith("MAKE_"):
        return (
            f"In the previous step you took action {act} but nothing changed in your position or inventory "
            f"(you are still at {new_pos}). Check recipe requirements, materials, and whether you face the correct tile."
        )

    if act == "SLEEP":
        return (
            f"In the previous step you took action {act} and there was no visible change to position or inventory "
            f"(you remain at {new_pos})."
        )

    if act == "NOOP":
        return "In the previous step you took action NOOP; your position and inventory did not change."

    return (
        f"In the previous step you took action {act} and there was no visible change to your coordinates "
        f"or inventory (still at {new_pos})."
    )


def _normalize_item(item: Any) -> List[str]:
    if isinstance(item, str):
        s = item.strip()
        return [s] if s else []

    if not isinstance(item, Mapping):
        return []

    before = item.get("before")
    after = item.get("after")
    action = item.get("action", "")
    if before is None or after is None:
        return []
    act = str(action).strip().upper()
    if not act:
        return []
    return [_line_for_transition(before, after, act)]


def render(transitions: Any) -> str:
    """
    Build human-readable lines from step transitions.

    Expected input: zero or one mapping (or a one-element list) with:
      - ``before``: environment state before the action
      - ``after``: environment state after the action
      - ``action``: primitive action name (e.g. ``LEFT``, ``DO``)

    If several transition dicts appear in a list, only the **last** is used
    (analysis of the latest primitive step only).

    For backward compatibility, a list of non-empty strings is joined as-is.
    """
    if transitions is None:
        return _EMPTY_MESSAGE

    if isinstance(transitions, (str, bytes)):
        text = str(transitions).strip()
        return text if text else _EMPTY_MESSAGE

    if isinstance(transitions, Mapping):
        transitions = [transitions]

    if not isinstance(transitions, list):
        return _EMPTY_MESSAGE

    dict_entries = [
        t
        for t in transitions
        if isinstance(t, Mapping) and t.get("before") is not None and t.get("after") is not None
    ]
    if len(dict_entries) > 1:
        transitions = [dict_entries[-1]]

    lines: List[str] = []
    for item in transitions:
        lines.extend(_normalize_item(item))

    if not lines:
        return _EMPTY_MESSAGE
    return "\n".join(lines)
