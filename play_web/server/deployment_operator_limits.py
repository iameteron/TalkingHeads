from __future__ import annotations

from oracle.prompts.prompt_generation import is_exo_world_mode, normalize_world_mode

DEPLOYMENT_CRAFTAX_MEGAPROMPT = "database_formulation_deployment"
DEPLOYMENT_EXO_MEGAPROMPT = "exo_database_formulation_deployment"

DEPLOYMENT_MEGAPROMPTS = frozenset({DEPLOYMENT_CRAFTAX_MEGAPROMPT, DEPLOYMENT_EXO_MEGAPROMPT})
# Legacy megaprompt ids from before the KARA/DUSA → exploration/deployment rename.
_LEGACY_DEPLOYMENT_MEGAPROMPTS = frozenset(
    {"database_formulation_dusa", "exo_database_formulation_dusa"}
)

# Before stone collection: wood, table, wood pickaxe.
_LIMIT_1_TASKS = frozenset({"collect_wood", "place_table", "make_wood_pickaxe"})
# Stone pickaxe / coal / iron tier.
_LIMIT_2_TASKS = frozenset(
    {"collect_stone", "make_stone_pickaxe", "collect_coal", "collect_iron"}
)
# Furnace, iron pickaxe, and diamond / core ore.
_LIMIT_4_TASKS = frozenset({"make_furnace", "make_iron_pickaxe", "collect_diamond"})

_TASK_LIMITS: dict[str, int] = {
    **{key: 1 for key in _LIMIT_1_TASKS},
    **{key: 2 for key in _LIMIT_2_TASKS},
    **{key: 4 for key in _LIMIT_4_TASKS},
}


def is_deployment_megaprompt(config_name: str) -> bool:
    name = str(config_name or "").strip()
    return name in DEPLOYMENT_MEGAPROMPTS or name in _LEGACY_DEPLOYMENT_MEGAPROMPTS


def deployment_megaprompt_config_name(world_mode: str | None) -> str:
    if is_exo_world_mode(world_mode):
        return DEPLOYMENT_EXO_MEGAPROMPT
    return DEPLOYMENT_CRAFTAX_MEGAPROMPT


def operator_call_limit_for_task(task_key: str) -> int:
    key = str(task_key or "").strip()
    return int(_TASK_LIMITS.get(key, 2))


def format_operator_call_budget_text(*, used: int, limit: int) -> str:
    safe_used = max(0, int(used))
    safe_limit = max(1, int(limit))
    if safe_used <= 0:
        return (
            f"You may call the Remote Operator at most **{safe_limit}** time(s) during this task. "
            f"You have not used any calls yet. "
            f"If you exceed **{safe_limit}** calls, your benchmark run will be penalized."
        )
    if safe_used >= safe_limit:
        return (
            f"You have used **{safe_used}** of **{safe_limit}** allowed Remote Operator call(s) "
            f"for this task. "
            f"If you exceed **{safe_limit}** calls, your benchmark run will be penalized."
        )
    return (
        f"You have used **{safe_used}** of **{safe_limit}** allowed Remote Operator call(s) "
        f"for this task. "
        f"If you exceed **{safe_limit}** calls, your benchmark run will be penalized."
    )


def operator_call_limit_violated(*, questions_count: int, limit: int) -> bool:
    return int(questions_count) > max(0, int(limit))
