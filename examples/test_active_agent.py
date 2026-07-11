"""
Example usage of ActiveAgent with the Craftax environment.

The script demonstrates three steps:
1) Build a textual observation from EnvState in the format:
       You in [32, 32]
       Stone in [30, 31]
       Tree in [29, 28]
       Tree in [27, 28]
       Path in [28, 28]

       The rest is grass. (You can walk only on grass or path)
2) Generate a full prompt for the ActiveAgent for the goal "Drink a water".
3) Call ActiveAgent and print its answer.
"""

from pathlib import Path
from typing import List, Tuple

import jax
import jax.numpy as jnp

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.craftax_state import EnvState
from craftax.craftax.constants import BlockType

from oracle.prompts.prompt_generation import generate_agent_prompt
from oracle.active_agent_base import ActiveAgent
from oracle.configs import GenConfig


def _tile_name_from_id(block_id: int) -> str:
    """Convert numeric block id to a human‑readable name like 'Stone', 'Tree', 'Path'."""
    try:
        bt = BlockType(block_id)
    except ValueError:
        return "Unknown"
    return bt.name.title().replace("_", " ")


def _iter_non_grass_tiles(state: EnvState) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Return a list of (tile_name, (x, y)) for all tiles that are NOT
    out-of-bounds / darkness / grass.
    """
    h, w = state.map.shape
    tiles: List[Tuple[str, Tuple[int, int]]] = []

    for x in range(h):
        for y in range(w):
            block_id = int(state.map[x, y])
            if block_id in (
                BlockType.OUT_OF_BOUNDS.value,
                BlockType.DARKNESS.value,
                BlockType.GRASS.value,
            ):
                continue
            name = _tile_name_from_id(block_id)
            tiles.append((name, (x, y)))

    return tiles


def format_inventory_from_state(state: EnvState) -> str:
    """Build text description of the agent's inventory (items with count > 0)."""
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
    return ", ".join(items) if items else "Empty"


def format_observation_from_state(state: EnvState, k: int = 4) -> str:
    """
    Build a short text observation like:

        You in [32, 32]
        Stone in [30, 31]
        Tree in [29, 28]
        Tree in [27, 28]
        Path in [28, 28]

        The rest is grass. (You can walk only on grass or path)
    """
    # Player position (world coordinates)
    px, py = map(int, state.player_position.tolist())

    # Collect nearby interesting tiles
    tiles = _iter_non_grass_tiles(state)

    # Sort by Manhattan distance to the player
    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    tiles_sorted = sorted(
        tiles,
        key=lambda t: manhattan((px, py), t[1]),
    )

    # Take k closest
    closest = tiles_sorted[:k]

    lines = [f"You in [{px}, {py}]"]
    for name, (x, y) in closest:
        lines.append(f"{name} in [{x}, {y}]")

    lines.append("")
    lines.append("The rest is grass. (You can walk only on grass or path)")

    return "\n".join(lines)


def build_active_agent_prompt_for_drink_water(state: EnvState) -> str:
    """Generate the full ActiveAgent prompt for the fixed goal 'Drink a water'."""
    inventory = format_inventory_from_state(state)
    goal = "Drink a water"
    msg_from_operator = ""
    return generate_agent_prompt(
        goal=goal,
        observation=state,
        message_from_operator=msg_from_operator,
        inventory=inventory,
    )


def example_predicted_answer() -> str:
    """
    Example of a *plausible* ActiveAgent answer for goal 'Drink a water'
    when water is two tiles below the agent.
    """
    return "\n".join(
        [
            "--- Act ---",
            "DOWN DOWN DO (TO GATHER SOMETHING)",
            "--- Act ---",
        ]
    )


def main():
    # 1) Create environment and reset to get an EnvState
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", False)
    rngs = jax.random.PRNGKey(0)
    rngs, reset_key = jax.random.split(rngs)
    obs, state = env.reset(reset_key)

    # 2) Format observation
    observation_text = format_observation_from_state(state)
    print("=== Observation ===")
    print(observation_text)
    print()

    # 3) Build prompt for goal "Drink a water"
    prompt = build_active_agent_prompt_for_drink_water(state)
    print("=== Prompt to ActiveAgent ===")
    print(prompt)
    print()

    # 4) Create ActiveAgent (model name is an example, adjust to your setup)
    #    Requires OPENROUTER_API_KEY env variable or explicit api_key.
    agent = ActiveAgent(
        model_name="openai/gpt-5.2",
        reasoning=False,
    )

    gen_cfg = GenConfig(max_new_tokens=64, do_sample=False)

    # 5) Call the agent
    answer = agent.chat(
        user_message=prompt,
        system_message="You are the Craftax agent described in the prompt.",
        gen=gen_cfg,
    )

    print("=== ActiveAgent answer ===")
    print(answer)
    print()


if __name__ == "__main__":
    main()

