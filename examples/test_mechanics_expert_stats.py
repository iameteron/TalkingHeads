"""
Statistical tests for the MECHANICS expert.

We measure three things (same structure as map expert tests):

1) **No‑answer rate for the profile question**
   Question: "How to make a stone sword?"
   We ask this 20 times, run the MECHANICS expert code, and count how many
   times we fail to get a usable answer (exception or empty/None).

2) **Error score for a non‑profile question being routed to the MECHANICS expert**
   Question: "Where is closest to agent coal?"
   We query the intent classifier 20 times and count how many times it
   (incorrectly) predicts `mechanics_expert` for this map question.

3) **Wrong‑answer rate for the profile question**
   Same profile question as (1). For each successful run we check
   whether the answer correctly describes how to make a stone sword
   (required: stone, sword/crafting, crafting table or equivalent).
   This uses a ground‑truth checker implemented here.
"""

from pathlib import Path

from oracle.config_loader import load_config
from oracle.intent import Intent
from oracle.oracle import Oracle


N_TRIALS = 20
PROFILE_QUESTION = "How to make a stone sword?"
NON_PROFILE_QUESTION = "Where is closest to agent coal?"


def is_mechanics_profile_answer_correct(answer: str) -> bool:
    """
    Correctness checker for the mechanics profile question
    "How to make a stone sword?".

    Ground truth (from game mechanics): stone sword requires
    collect_stone + place_table (crafting table). We consider the answer
    correct if it mentions:
      - stone (material)
      - sword (or make_stone_sword)
      - crafting table / table / place_table (where to craft)
    """
    if answer is None:
        return False

    text = str(answer).lower()

    has_stone = "stone" in text
    has_sword = "sword" in text or "make_stone_sword" in text
    has_crafting = (
        "crafting" in text
        or "table" in text
        or "place_table" in text
        or "workbench" in text
    )

    return has_stone and has_sword and has_crafting


def run_profile_question_stats(oracle: Oracle) -> None:
    """
    Test 1 and 3:
      * how often the MECHANICS expert fails to return an answer
      * how often it returns an incorrect answer
    """
    no_answer = 0
    incorrect = 0

    for i in range(N_TRIALS):
        try:
            result = oracle.answer(
                PROFILE_QUESTION,
                run_code=True,
                module_name="answer_code_mechanics_stats",
            )
        except Exception:
            no_answer += 1
            continue

        if result is None or (isinstance(result, str) and not result.strip()):
            no_answer += 1
            continue

        if not is_mechanics_profile_answer_correct(result):
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
      Error score for non‑profile question:
      how often the intent classifier routes a map question
      to the MECHANICS expert (which would be an error).
    """
    wrong_to_mechanics = 0

    for i in range(N_TRIALS):
        label = oracle.intent.predict(NON_PROFILE_QUESTION)[0]
        if label == Intent.MECHANICS_EXPERT:
            wrong_to_mechanics += 1

    print(f"Non‑profile question: {NON_PROFILE_QUESTION!r}")
    print(f"  trials                              : {N_TRIALS}")
    print(f"  routed to MECHANICS expert (errors) : {wrong_to_mechanics}")
    print(f"  error rate                         : {wrong_to_mechanics / N_TRIALS:.2%}")
    print()


def main():
    config_path = Path(__file__).resolve().parent.parent / "config" / "oracle_config.yaml"
    config = load_config(path=str(config_path))
    oracle = Oracle(config)

    # 1) & 3) Profile question availability + correctness stats
    run_profile_question_stats(oracle)

    # 2) Non‑profile question intent‑routing error stats
    run_non_profile_intent_error_stats(oracle)


if __name__ == "__main__":
    main()
