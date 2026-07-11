"""
Example usage of `ActiveAgent` with the Craftax environment.

The script demonstrates three steps:
1) Build a textual observation from `EnvState` in the format:
       You in [32, 32]
       Stone in [30, 31]
       Tree in [29, 28]
       Tree in [27, 28]
       Path in [28, 28]

       The rest is grass. (You can walk only on grass or path)
2) Generate a full prompt for the `ActiveAgent` for the goal "Drink a water".
3) Call `ActiveAgent` and apply its answer to the environment,
   or route questions to the `Oracle`.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

import jax

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.craftax_state import EnvState
from craftax.craftax.constants import BlockType, Action

from oracle.prompts.prompt_generation import generate_agent_prompt
from oracle.active_agent_base import ActiveAgent
from oracle.configs import GenConfig

from oracle.config_loader import load_config
from oracle.oracle import Oracle


def test_oracle_hub(
    question: str,
    module_name: str,
    run_code: bool,
    oracle: Oracle,
    env_state: None = None,
) -> str:
    """Ask the `Oracle` a question on behalf of the agent."""
    return oracle.answer(
        question,
        module_name=module_name,
        run_code=run_code,
        env_state=env_state,
    )

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
    def manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
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


def build_active_agent_prompt_for_drink_water(
    state: EnvState,
    message_from_operator: str,
) -> str:
    """Generate the full ActiveAgent prompt for the fixed goal 'Drink a water'."""
    inventory = format_inventory_from_state(state)
    goal = "Drink a water"
    return generate_agent_prompt(
        goal=goal,
        observation=state,
        message_from_operator=message_from_operator,
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


def parse_agent_answer(raw_answer: str) -> Dict[str, str]:
    """
    Parse the string answer produced by the `ActiveAgent`.

    Returns a dict with exactly one of the keys:
    - ``{"action": "<ACTION STRING>"}``
    - ``{"question": "<QUESTION STRING>"}``

    If the format is not recognised, an empty dict is returned.
    """
    text = str(raw_answer or "").strip()

    if "--- Act ---" in text:
        content = text.split("--- Act ---", maxsplit=1)[1].strip()
        return {"action": content} if content else {}

    if "--- Q ---" in text:
        content = text.split("--- Q ---", maxsplit=1)[1].strip()
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


def do_actions(
    actions: List[str],
    env: Any,
    state: EnvState,
    rngs: jax.Array,
) -> Tuple[Any, EnvState, float, bool, Dict[str, Any], jax.Array]:
    """
    Convert string actions to Craftax `Action` values and apply them.

    The input strings are expected to look like "UP", "DOWN", "LEFT", "RIGHT",
    "DO", etc. Any tokens that do not correspond to a valid `Action` member
    (e.g. commentary words like "(TO", "GATHER", "SOMETHING)") are ignored.
    """
    last_transition: Tuple[Any, EnvState, float, bool, Dict[str, Any]] | None = None

    for action_str in actions:
        token = action_str.strip().upper()

        # Skip empty / non‑action tokens
        if not token:
            continue

        # Only keep tokens that map to a valid Action enum member
        if token not in Action.__members__:
            continue

        action_value = Action[token].value

        # Step the environment with a fresh RNG key and current state
        rngs, step_key = jax.random.split(rngs)
        obs, state, reward, done, info = env.step(step_key, state, action_value)
        last_transition = (obs, state, reward, done, info)

    if last_transition is None:
        raise ValueError("No valid Craftax actions were provided.")

    # Return the last transition together with the updated RNG key
    obs, state, reward, done, info = last_transition
    return obs, state, reward, done, info, rngs


def run_interaction_loop(
    env: Any,
    agent: ActiveAgent,
    oracle: Oracle,
    num_steps: int = 10,
) -> None:
    """Run a simple loop of agent–environment–oracle interaction."""
    rngs = jax.random.PRNGKey(0)
    rngs, reset_key = jax.random.split(rngs)
    obs, state = env.reset(reset_key)

    message_from_operator = "Hello! Can I help you?"

    for step in range(num_steps):
        prompt = build_active_agent_prompt_for_drink_water(state, message_from_operator)
        print("=== Prompt to ActiveAgent ===")
        print(prompt)
        print()

        gen_cfg = GenConfig(max_new_tokens=64, do_sample=False)

        raw_answer = agent.chat(
            user_message=prompt,
            system_message="You are the Craftax agent described in the prompt.",
            gen=gen_cfg,
        )

        agent_answer = parse_agent_answer(raw_answer)

        print("=== ActiveAgent answer ===")
        print(agent_answer or raw_answer)
        print()

        if "action" in agent_answer:
            print("=== Action ===")
            print(agent_answer["action"])
            craftax_actions = agent_answer["action"].split()
            obs, state, reward, done, info, rngs = do_actions(
                craftax_actions, env, state, rngs
            )
            print()

            if done:
                print(f"Episode finished after step {step + 1} with reward {reward}.")
                break

        elif "question" in agent_answer:
            print("=== Question ===")
            print(agent_answer["question"])
            message_from_operator = test_oracle_hub(
                agent_answer["question"],
                "answer_code_oracle",
                True,
                oracle,
                state,
            )
            # # save message to file
            # with open("message_from_operator.txt", "w") as f:
            #     f.write(message_from_operator)
            
            # break
            print(message_from_operator)
            print()
        else:
            print("=== Raw answer ===")
            print(raw_answer)
            print()
            break


def main() -> None:
    """Entry point for running the example interaction loop."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "oracle_config.yaml"
    config = load_config(path=str(config_path))
    oracle = Oracle(config)

    # 1) Create environment
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", False)

    # 2) Create agent
    agent = ActiveAgent(
        model_name="openai/gpt-5.1",
        reasoning=False,
    )

    # 3) Run interaction loop
    run_interaction_loop(env=env, agent=agent, oracle=oracle, num_steps=10)


if __name__ == "__main__":
    main()
