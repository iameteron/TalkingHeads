"""
Example: test the goal expert pipeline.

The goal expert:
1. Asks the question expert what to ask (map + mechanics questions for the goal)
2. Asks the map and mechanics experts those questions
3. Aggregates their answers into a short "how to reach the goal" response

This example forces intent to GOAL_EXPERT so the pipeline runs regardless of
whether the intent model was retrained with goal_expert examples.
"""
from pathlib import Path
from unittest.mock import patch

import jax
from craftax.craftax_env import make_craftax_env_from_name

from oracle.config_loader import load_config
from oracle.oracle import Oracle
from oracle.intent import Intent


def main():
    config_path = Path(__file__).resolve().parent.parent / "config" / "oracle_config.yaml"
    config = load_config(path=str(config_path))
    oracle = Oracle(config)

    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", False)
    rngs = jax.random.PRNGKey(42)
    rngs, reset_key = jax.random.split(rngs)
    obs, state = env.reset(reset_key)

    goal_question = "How do I collect coal?"
    print("Goal question:", goal_question)
    print("(Intent forced to GOAL_EXPERT for this test)\n")

    with patch.object(oracle, "_decide_intent", return_value=Intent.GOAL_EXPERT):
        answer = oracle.answer(goal_question, run_code=False, env_state=state)

    print("\n--- Aggregated answer ---")
    print(answer)


if __name__ == "__main__":
    main()
