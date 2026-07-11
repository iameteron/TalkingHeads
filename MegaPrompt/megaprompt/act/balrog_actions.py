from __future__ import annotations

from collections.abc import Iterable
from typing import Any


BALROG_ACTION_DESCRIPTIONS = {
    "Noop": "do nothing",
    "Move West": "move west on flat ground",
    "Move East": "move east on flat ground",
    "Move North": "move north on flat ground",
    "Move South": "move south on flat ground",
    "Do": "multiuse action to collect material, drink from lake and hit creature in front",
    "Sleep": "sleep when energy level is below maximum",
    "Place Stone": "place a stone in front",
    "Place Table": "place a table",
    "Place Furnace": "place a furnace",
    "Place Plant": "place a plant",
    "Make Wood Pickaxe": "craft a wood pickaxe with a nearby table and wood in inventory",
    "Make Stone Pickaxe": "craft a stone pickaxe with a nearby table, wood, and stone in inventory",
    "Make Iron Pickaxe": "craft an iron pickaxe with a nearby table and furnace, wood, coal, and iron in inventory",
    "Make Wood Sword": "craft a wood sword with a nearby table and wood in inventory",
    "Make Stone Sword": "craft a stone sword with a nearby table, wood, and stone in inventory",
    "Make Iron Sword": "craft an iron sword with a nearby table and furnace, wood, coal, and iron in inventory",
}

ACTION_TO_BALROG_TEXT = (
    "Noop",
    "Move West",
    "Move East",
    "Move North",
    "Move South",
    "Do",
    "Sleep",
    "Place Stone",
    "Place Table",
    "Place Furnace",
    "Place Plant",
    "Make Wood Pickaxe",
    "Make Stone Pickaxe",
    "Make Iron Pickaxe",
    "Make Wood Sword",
    "Make Stone Sword",
    "Make Iron Sword",
)

ACTION_ALIASES = {
    "NOOP": "Noop",
    "LEFT": "Move West",
    "RIGHT": "Move East",
    "UP": "Move North",
    "DOWN": "Move South",
    "DO": "Do",
    "SLEEP": "Sleep",
    "PLACE_STONE": "Place Stone",
    "PLACE_TABLE": "Place Table",
    "PLACE_FURNACE": "Place Furnace",
    "PLACE_PLANT": "Place Plant",
    "MAKE_WOOD_PICKAXE": "Make Wood Pickaxe",
    "MAKE_STONE_PICKAXE": "Make Stone Pickaxe",
    "MAKE_IRON_PICKAXE": "Make Iron Pickaxe",
    "MAKE_WOOD_SWORD": "Make Wood Sword",
    "MAKE_STONE_SWORD": "Make Stone Sword",
    "MAKE_IRON_SWORD": "Make Iron Sword",
}


def _to_list(actions: Any) -> list[str]:
    if actions is None:
        return []
    if isinstance(actions, str):
        return [actions]
    if isinstance(actions, Iterable):
        return [str(a) for a in actions]
    return [str(actions)]


def render(actions: Any) -> str:
    raw_actions = _to_list(actions)
    if not raw_actions:
        balrog_actions = ACTION_TO_BALROG_TEXT
    else:
        balrog_actions = []
        for action in raw_actions:
            mapped = ACTION_ALIASES.get(action.strip().upper(), action.strip())
            if mapped in BALROG_ACTION_DESCRIPTIONS:
                balrog_actions.append(mapped)

    return "\n".join(
        f"{action}: {BALROG_ACTION_DESCRIPTIONS[action]}"
        for action in balrog_actions
    )
