"""
Utility functions for benchmarking agent performance.

These functions check if specific instructions have been completed based on the game state.
Also includes helper functions for formatting observations and parsing agent answers.
"""

import re
from typing import Any, Dict, List, Tuple
import numpy as np

from craftax.craftax.craftax_state import EnvState
from craftax.craftax.constants import BlockType, Action, DIRECTIONS


def check_collect_wood(initial_state: EnvState, current_state: EnvState) -> bool:
    """
    Check if "Collect wood" instruction is completed.
    
    Returns True if the agent has at least 1 wood in inventory.
    """
    return current_state.inventory.wood > 0


def check_place_table(initial_state: EnvState, current_state: EnvState) -> bool:
    """
    Check if "Place table" instruction is completed.
    
    Returns True if a crafting table exists on the map.
    """
    h, w = current_state.map.shape
    for x in range(h):
        for y in range(w):
            block_id = int(current_state.map[x, y])
            if block_id == BlockType.CRAFTING_TABLE.value:
                return True
    return False


def check_make_wooden_pickaxe(initial_state: EnvState, current_state: EnvState) -> bool:
    """
    Check if "Make wooden pickaxe" instruction is completed.
    
    Returns True if the agent has at least 1 wooden pickaxe in inventory.
    """
    return current_state.inventory.wood_pickaxe > 0
 

def check_dig_stone_from_rock(initial_state: EnvState, current_state: EnvState) -> bool:
    """
    Check if "Dig a stone from rock" instruction is completed.
    
    Returns True if the agent has at least 1 stone in inventory.
    """
    return current_state.inventory.stone > 0


def check_place_stone(initial_state: EnvState, current_state: EnvState) -> bool:
    """
    Check if "Place stone" instruction is completed.
    
    Returns True if a stone block exists on the map (that wasn't there initially).
    Note: This checks if stone exists on map, assuming initial state had no placed stones.
    """
    # Count stones in current state
    h, w = current_state.map.shape
    current_stones = 0
    for x in range(h):
        for y in range(w):
            block_id = int(current_state.map[x, y])
            if block_id == BlockType.STONE.value:
                current_stones += 1
    
    # Count stones in initial state
    h_init, w_init = initial_state.map.shape
    initial_stones = 0
    for x in range(h_init):
        for y in range(w_init):
            block_id = int(initial_state.map[x, y])
            if block_id == BlockType.STONE.value:
                initial_stones += 1
    
    # If there are more stones now than initially, stone was placed
    return current_stones > initial_stones


# Mapping from instruction text to check function
INSTRUCTION_CHECKERS = {
    "Collect wood": check_collect_wood,
    "Place table": check_place_table,
    "Make wooden pickaxe": check_make_wooden_pickaxe,
    "Dig a stone from rock": check_dig_stone_from_rock,
    "Place stone": check_place_stone,
}


def check_instruction_completed(
    instruction: str, initial_state: EnvState, current_state: EnvState
) -> bool:
    """
    Generic function to check if an instruction is completed.
    
    Args:
        instruction: The instruction text
        initial_state: The initial game state before the instruction
        current_state: The current game state
        
    Returns:
        True if the instruction is completed, False otherwise
    """
    checker = INSTRUCTION_CHECKERS.get(instruction)
    if checker is None:
        raise ValueError(f"Unknown instruction: {instruction}")
    return checker(initial_state, current_state)


# ============================================================================
# Agent answer parsing and observation formatting functions
# (adapted from play_web/active_agent_helpers.py)
# ============================================================================

def _tile_name_from_id(block_id: int) -> str:
    """Convert numeric block id to human-readable name."""
    try:
        bt = BlockType(block_id)
    except ValueError:
        return "Unknown"
    return bt.name.title().replace("_", " ")


def _iter_non_grass_tiles(state: EnvState) -> List[Tuple[str, Tuple[int, int]]]:
    """Return (tile_name, (x, y)) for all tiles that are not out-of-bounds/darkness/grass."""
    # Convert map to numpy array once to avoid repeated JAX operations
    map_array = np.array(state.map)
    h, w = map_array.shape
    tiles: List[Tuple[str, Tuple[int, int]]] = []

    for x in range(h):
        for y in range(w):
            block_id = int(map_array[x, y])
            if block_id in (
                BlockType.OUT_OF_BOUNDS.value,
                BlockType.DARKNESS.value,
                BlockType.GRASS.value,
            ):
                continue
            if block_id < 0:
                continue
            name = _tile_name_from_id(block_id)
            tiles.append((name, (x, y)))

    return tiles


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


def format_observation_from_state(state: EnvState, k: int = 5) -> str:
    """
    Build text observation like:
        You in [32, 32]
        Facing: up
        Block you're turned to: Stone at [31, 32]
        Stone in [30, 31]
        ...
        The rest is grass. (You can walk only on grass or path)

    Shows up to k unique nearest object types (nearest instance of each type).
    """
    # Use direct indexing instead of tolist() to avoid memory issues with JAX
    px = int(state.player_position[0])
    py = int(state.player_position[1])
    dir_idx = int(state.player_direction)

    # Which side is turned on (direction the agent is facing)
    direction_name = (
        Action(dir_idx).name.lower()
        if 1 <= dir_idx <= 4
        else "unknown"
    )
    lines = [f"You in [{px}, {py}]", f"Facing: {direction_name}"]

    # Block directly in front (the one the agent is looking at)
    if 1 <= dir_idx <= 4:
        dir_vec = DIRECTIONS[dir_idx]
        front_x = px + int(dir_vec[0])
        front_y = py + int(dir_vec[1])
        map_array = np.array(state.map)
        h, w = map_array.shape
        if 0 <= front_x < h and 0 <= front_y < w:
            block_id = int(map_array[front_x, front_y])
            block_name = _tile_name_from_id(block_id)
            lines.append(f"Block you're turned to: {block_name} at [{front_x}, {front_y}]")
        else:
            lines.append("Block you're turned to: out of bounds")
    lines.append("")

    tiles = _iter_non_grass_tiles(state)

    def manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    tiles_sorted = sorted(tiles, key=lambda t: manhattan((px, py), t[1]))
    # Take up to k unique object types: for each type, show the nearest instance
    seen_names = set()
    closest = []
    for name, pos in tiles_sorted:
        if name not in seen_names:
            seen_names.add(name)
            closest.append((name, pos))
            if len(closest) >= k:
                break

    for name, (x, y) in closest:
        lines.append(f"{name} in [{x}, {y}]")
    lines.append("")
    lines.append("The rest is grass. (You can walk only on grass or path)")

    return "\n".join(lines)


def parse_agent_answer(raw_answer: str) -> Dict[str, str]:
    """
    Parse ActiveAgent answer. Returns one of:
    - {"action": "<ACTION STRING>"}
    - {"question": "<QUESTION STRING>"}
    - {}
    """
    if not isinstance(raw_answer, str):
        raw_answer = str(raw_answer)
    text = raw_answer.strip()

    if "--- Act ---" in text:
        parts = text.split("--- Act ---", maxsplit=1)
        if len(parts) >= 2:
            content = parts[1].strip()
            return {"action": content} if content else {}

    if "--- Q ---" in text:
        parts = text.split("--- Q ---", maxsplit=1)
        if len(parts) >= 2:
            content = parts[1].strip()
            return {"question": content} if content else {}

    action_match = re.search(r"<(?:action|act)>\s*(.*?)\s*</(?:action|act)>", text, flags=re.IGNORECASE | re.DOTALL)
    if action_match and action_match.group(1).strip():
        return {"action": action_match.group(1).strip()}

    question_match = re.search(r"<(?:question|ask|q)>\s*(.*?)\s*</(?:question|ask|q)>", text, flags=re.IGNORECASE | re.DOTALL)
    if question_match and question_match.group(1).strip():
        return {"question": question_match.group(1).strip()}

    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("ACTION:"):
            content = s.split(":", 1)[1].strip()
            if content:
                return {"action": content}
        if s.upper().startswith("QUESTION:"):
            content = s.split(":", 1)[1].strip()
            if content:
                return {"question": content}

    return {}
