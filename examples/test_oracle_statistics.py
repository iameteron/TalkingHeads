"""
Example: test Oracle expert statistics.

This script:
 1. Creates an Oracle from the standard config.
 2. Asks a few questions that should route to different experts.
 3. Prints aggregated statistics from `Oracle.return_statistics()`.
"""

from pathlib import Path

import jax
from craftax.craftax_env import make_craftax_env_from_name

from oracle.config_loader import load_config
from oracle.oracle import Oracle


def main():
    # Load the same config as other examples.
    config_path = Path(__file__).resolve().parent.parent / "config" / "oracle_config.yaml"
    config = load_config(path=str(config_path))
    oracle = Oracle(config)

    # Prepare environment state for map/goal/action experts.
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", False)
    rngs = jax.random.PRNGKey(42)
    rngs, reset_key = jax.random.split(rngs)
    obs, state = env.reset(reset_key)

    questions = [
        "What is nearby on the map?",          # likely MAP or QUESTION
        "How do I collect coal?",              # likely GOAL
        "What are the game mechanics of fire?",# MECHANICS / QUESTION
        "What should I do next?",              # ACTION / QUESTION
    ]

    for q in questions:
        print(f"\n=== Question: {q!r} ===")
        try:
            answer = oracle.answer(q, env_state=state, module_name="stats_test")
            print("Answer:", answer[:300], "..." if len(answer) > 300 else "")
        except Exception as e:
            print("Error during answer:", e)

    print("\n=== Oracle statistics ===")
    stats = oracle.return_statistics()
    for expert_name, expert_stats in stats.items():
        print(f"{expert_name}: {expert_stats}")


if __name__ == "__main__":
    main()

