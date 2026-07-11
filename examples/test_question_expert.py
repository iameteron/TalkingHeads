"""
Example: call the question expert to generate map_expert_question and mechanics_expert_question for a goal.
"""
from pathlib import Path

from oracle.config_loader import load_config
from oracle.oracle import Oracle


def main():
    config_path = Path(__file__).resolve().parent.parent / "config" / "oracle_config.yaml"
    config = load_config(path=str(config_path))
    oracle = Oracle(config)

    goal = "Collect coal to fuel a furnace"
    result = oracle.generate_expert_questions(goal)

    print("Goal:", goal)
    print("Generated questions:")
    print("  map_expert_question:", result["map_expert_question"])
    print("  mechanics_expert_question:", result["mechanics_expert_question"])
    print("Full dict:", result)


if __name__ == "__main__":
    main()
