"""Map exo-planet agent actions to Craftax engine tokens."""

from __future__ import annotations

from actions import EXO_ENV_ACTIONS, EXO_META_ACTIONS

EXO_TO_CRAFTAX: dict[str, str] = {
    "EXTRACT": "DO",
    "ENGAGE_HOSTILE": "DO",
    "DRINK_BRINE": "DO",
    "DORMANCY": "SLEEP",
    "RECHARGE": "REST",
    "PLACE_REPLICATOR": "PLACE_TABLE",
    "PLACE_THERMAL_OVEN": "PLACE_FURNACE",
    "PLACE_BASALT_BEACON": "PLACE_STONE",
    "PLACE_BIO_SPROUT": "PLACE_PLANT",
    "MAKE_BONE_DRILL": "MAKE_WOOD_PICKAXE",
    "MAKE_ROCK_DRILL": "MAKE_STONE_PICKAXE",
    "MAKE_TITAN_DRILL": "MAKE_IRON_PICKAXE",
    "MAKE_BONE_DAGGER": "MAKE_WOOD_SWORD",
    "MAKE_ROCK_CUTTER": "MAKE_STONE_SWORD",
    "MAKE_TITAN_BLADE": "MAKE_IRON_SWORD",
}

# DO sub-intents preserved for env helpers that inspect full action strings.
EXO_TO_CRAFTAX_FULL: dict[str, str] = {
    **{k: v for k, v in EXO_TO_CRAFTAX.items() if v != "DO"},
    "EXTRACT": "DO (TO GATHER SOMETHING)",
    "ENGAGE_HOSTILE": "DO (TO FIGHT)",
    "DRINK_BRINE": "DO (DRINK WATER)",
}

CRAFTAX_TO_EXO: dict[str, str] = {v: k for k, v in EXO_TO_CRAFTAX_FULL.items()}

# After engine sanitization DO hints are stripped; map bare Craftax tokens back to exo.
CRAFTAX_TO_EXO_BARE: dict[str, str] = {
    "DO": "EXTRACT",
    "SLEEP": "DORMANCY",
    "REST": "RECHARGE",
    "PLACE_TABLE": "PLACE_REPLICATOR",
    "PLACE_FURNACE": "PLACE_THERMAL_OVEN",
    "PLACE_STONE": "PLACE_BASALT_BEACON",
    "PLACE_PLANT": "PLACE_BIO_SPROUT",
    "MAKE_WOOD_PICKAXE": "MAKE_BONE_DRILL",
    "MAKE_STONE_PICKAXE": "MAKE_ROCK_DRILL",
    "MAKE_IRON_PICKAXE": "MAKE_TITAN_DRILL",
    "MAKE_WOOD_SWORD": "MAKE_BONE_DAGGER",
    "MAKE_STONE_SWORD": "MAKE_ROCK_CUTTER",
    "MAKE_IRON_SWORD": "MAKE_TITAN_BLADE",
}

EXO_TOKENS = set(EXO_ENV_ACTIONS) | set(EXO_META_ACTIONS)


def to_engine_action(exo_action: str, *, full_do: bool = True) -> str:
    """Translate one exo action token to Craftax action string."""
    token = str(exo_action or "").strip().upper()
    mapping = EXO_TO_CRAFTAX_FULL if full_do else EXO_TO_CRAFTAX
    return mapping.get(token, token)


def to_display_action(raw_action: str, engine_action: str | None = None) -> str:
    """
    Pick the exo-planet action label for UI/history.

    Prefers the agent's raw exo token when present; otherwise reverse-maps the
    sanitized Craftax engine action.
    """
    raw = str(raw_action or "").strip()
    if not raw:
        raw = str(engine_action or "").strip()
    if not raw:
        return ""

    first = raw.upper().split()[0]
    if first in EXO_TOKENS:
        return first if len(raw.split()) == 1 else raw.strip().upper()

    engine = str(engine_action or raw).strip().upper()
    engine = " ".join(engine.split())
    if engine in CRAFTAX_TO_EXO:
        return CRAFTAX_TO_EXO[engine]
    bare = engine.split()[0] if engine else ""
    if bare in CRAFTAX_TO_EXO_BARE:
        return CRAFTAX_TO_EXO_BARE[bare]
    return raw.strip().upper()
