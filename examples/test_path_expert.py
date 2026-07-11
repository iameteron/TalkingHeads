"""
Example: call path expert and print real model outputs.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import jax
from craftax.craftax_env import make_craftax_env_from_name


def main():
    project_root = Path(__file__).resolve().parent.parent
    # Force imports from this repository checkout.
    sys.path.insert(0, str(project_root))

    from oracle.config_loader import load_config
    from oracle.oracle import Oracle

    config_path = project_root / "config" / "oracle_config.yaml"
    config = load_config(path=str(config_path))
    with patch("oracle.oracle.Intent") as mocked_intent_cls:
        mocked_intent = mocked_intent_cls.return_value
        mocked_intent.predict.return_value = ["path_expert"]
        oracle = Oracle(config)

    print("Imported oracle module from:", Path(sys.modules["oracle.oracle"].__file__).resolve())

    # Use real env_state with map so render_textual_observation_with_path_from_env_state can work.
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", False)
    rng = jax.random.PRNGKey(42)
    _, base_state = env.reset(rng)
    state = SimpleNamespace(
        map=base_state.map,
        player_position=base_state.player_position,
        target_location=[20, 24],
    )

    questions = [
        "What is near the target point?",
        "In which direction should I go now?",
    ]

    for question in questions:
        print("\n" + "- " * 30)
        print("Question:", question)
        answer = oracle.answer_gm_only(question, env_state=state, run_code=False)
        print("Model output:")
        print(answer)


if __name__ == "__main__":
    main()

