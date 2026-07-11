"""
Statistical tests for the MAP expert.

We measure three things (all using the *same* fixed environment state):

1) **No‑answer rate for the profile question**  
   Question: "Where is closest to agent coal?"  
   We ask this 20 times, run the MAP expert code, and count how many
   times we fail to get a usable answer (exception or empty/None).

2) **Error score for a non‑profile question being routed to the MAP expert**  
   Question: "How to make a stone sword"  
   We query the intent classifier 20 times and count how many times it
   (incorrectly) predicts `map_expert` for this mechanics question.

3) **Wrong‑answer rate for the profile question**  
   Same profile question as (1). For each successful run we check
   whether the answer correctly identifies the closest coal to the agent.
   This uses a ground‑truth checker implemented here.
"""

from pathlib import Path

import jax
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name

from oracle.config_loader import load_config
from oracle.intent import Intent
from oracle.oracle import Oracle


N_TRIALS = 20
PROFILE_QUESTION = "Where is closest to agent coal?"
NON_PROFILE_QUESTION = "How to make a stone sword"


def make_fixed_state():
    """
    Create a deterministic Craftax environment state.

    We always reset with the same RNG key so that the map and agent
    position are identical between runs. Within this script we also
    reuse the same `state` object for all questions.
    """
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", False)
    rngs = jax.random.PRNGKey(123)  # fixed seed for reproducibility
    rngs, reset_key = jax.random.split(rngs)
    obs, state = env.reset(reset_key)
    return env, state


def compute_nearest_coal(state):
    """
    Ground‑truth: find the coal block closest to the agent.

    Returns:
        (row, col) of the closest coal, or None if there is no coal.
    """
    game_map = np.array(state.map)
    # `player_position` is expected to be (row, col, ...), keep first 2 coords.
    agent_pos = np.array(state.player_position)[:2]

    coal_positions = np.argwhere(game_map == 8)  # 8 == COAL in the mapping
    if coal_positions.size == 0:
        return None

    # Euclidean distance from agent to each coal cell
    diffs = coal_positions - agent_pos
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    row, col = coal_positions[idx]
    return int(row), int(col)


def is_profile_answer_correct(answer: str, state) -> bool:
    """
    Correctness checker for the profile question.

    We consider an answer correct if:
      * there is no coal in the map AND the answer says so, or
      * the textual answer contains the coordinates of the closest coal.
    """
    if answer is None:
        return False

    answer_lower = str(answer).lower()
    nearest = compute_nearest_coal(state)

    # Case 1: no coal exists in this map
    if nearest is None:
        # Accept if the model clearly states that there is no coal.
        return "no coal" in answer_lower

    row, col = nearest
    coord_patterns = [
        f"({row}, {col})",
        f"({row},{col})",
        f"{row}, {col}",
        f"{row},{col}",
    ]
    for pat in coord_patterns:
        if pat in answer:
            return True
    return False


def run_profile_question_stats(oracle: Oracle, state) -> None:
    """
    Test 1 and 3:
      * how often the MAP expert fails to return an answer
      * how often it returns an incorrect answer
    """
    no_answer = 0
    incorrect = 0

    for i in range(N_TRIALS):
        try:
            result = oracle.answer(
                PROFILE_QUESTION,
                run_code=True,
                module_name="answer_code_map_stats",
                env_state=state,
            )
        except Exception:
            no_answer += 1
            continue

        if result is None or (isinstance(result, str) and not result.strip()):
            no_answer += 1
            continue

        if not is_profile_answer_correct(result, state):
            incorrect += 1

    print(f"Profile question: {PROFILE_QUESTION!r}")
    print(f"  trials               : {N_TRIALS}")
    print(f"  no‑answer count      : {no_answer}")
    print(f"  wrong‑answer count   : {incorrect}")
    success = N_TRIALS - no_answer - incorrect
    print(f"  success count        : {success}")
    print(f"  success rate         : {success / N_TRIALS:.2%}")
    print()


def run_non_profile_intent_error_stats(oracle: Oracle) -> None:
    """
    Test 2:
      Error score into non‑profile question:
      how often the intent classifier routes a mechanics question
      to the MAP expert (which would be an error).

    Note: the intent model is deterministic, but we still run 20
    "trials" for symmetry with the other tests.
    """
    wrong_to_map = 0

    for i in range(N_TRIALS):
        label = oracle.intent.predict(NON_PROFILE_QUESTION)[0]
        if label == Intent.MAP_EXPERT:
            wrong_to_map += 1

    print(f"Non‑profile question: {NON_PROFILE_QUESTION!r}")
    print(f"  trials                         : {N_TRIALS}")
    print(f"  routed to MAP expert (errors) : {wrong_to_map}")
    print(f"  error rate                    : {wrong_to_map / N_TRIALS:.2%}")
    print()


def main():
    # Load oracle config and create Oracle instance
    config_path = Path(__file__).resolve().parent.parent / "config" / "oracle_config.yaml"
    config = load_config(path=str(config_path))
    oracle = Oracle(config)

    # Create one fixed environment state and reuse it for all tests.
    env, state = make_fixed_state()

    # 1) & 3) Profile question availability + correctness stats
    run_profile_question_stats(oracle, state)

    # 2) Non‑profile question intent‑routing error stats
    run_non_profile_intent_error_stats(oracle)


if __name__ == "__main__":
    main()

