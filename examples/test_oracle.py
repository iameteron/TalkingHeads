from pathlib import Path

from craftax.craftax_env import make_craftax_env_from_name
import jax
import time

from oracle.config_loader import load_config
from oracle.oracle import Oracle


def test_oracle_hub(question: str, module_name: str, run_code: bool, oracle: Oracle, env_state=None):
    print(question)
    t0 = time.perf_counter()
    answer = oracle.answer(
        question,
        module_name=module_name,
        run_code=run_code,
        env_state=env_state,
    )
    elapsed = time.perf_counter() - t0
    print(f"query took {elapsed:.2f} s")
    print(answer)
    return answer


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent / "config" / "oracle_config.yaml"
    config = load_config(path=str(config_path))
    oracle = Oracle(config)

    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", False)
    rngs = jax.random.PRNGKey(42)
    rngs, reset_key = jax.random.split(rngs)
    obs, state = env.reset(reset_key)

    print("- - - - -" * 10)
    test_oracle_hub(
        question="Where is the nearest coal?",
        module_name="answer_code_coal_oracle",
        run_code=True,
        oracle=oracle,
        env_state=state,
    )
    print("- - - - -" * 10)

    print("- - - - -" * 10)
    test_oracle_hub(
        question="What block is nearest to the agent?",
        module_name="nearest_block_oracle",
        run_code=True,
        oracle=oracle,
        env_state=state,
    )
    print("- - - - -" * 10)

    print("- - - - -" * 10)
    test_oracle_hub(
        question="What achievements are required before collecting coal?",
        module_name="mechanics_coal_oracle",
        run_code=True,
        oracle=oracle,
        env_state=state,
    )
    print("- - - - -" * 10)

    print("- - - - -" * 10)
    test_oracle_hub(
        question="What should I do before making an iron sword?",
        module_name="mechanics_iron_sword_oracle",
        run_code=True,
        oracle=oracle,
        env_state=state,
    )
    print("- - - - -" * 10)
