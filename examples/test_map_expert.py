from oracle.prompts.prompt_generation import create_map_prompt

from oracle.code_based_expert import CodeBasedExpertWrapper, Expert
from oracle.configs import ExpertConfig

from oracle.utils import run_llm_code

from craftax.craftax_env import make_craftax_env_from_name
import jax
import time


def test_map_expert_local(question: str, module_name: str):
    model_name = "Qwen/Qwen2.5-Coder-7B"
    expert_cfg = ExpertConfig(mode="local", model_path=model_name)
    expert = Expert(expert_cfg, prompt_function=create_map_prompt)
    map_expert = CodeBasedExpertWrapper(expert)

    t0 = time.perf_counter()
    print(question)

    # Fast default: greedy, shorter output
    answer = map_expert.chat(question)
    code = map_expert.get_code_block(answer)

    env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', False)
    rngs = jax.random.PRNGKey(42)
    rngs, reset_key = jax.random.split(rngs)
    obs, state = env.reset(reset_key)

    run_llm_code(code, state, module_name=module_name)
    elapsed = time.perf_counter() - t0
    print(f"query took {elapsed:.2f} s")
    return answer

def test_map_expert_hub(question: str, module_name: str):
    model_name = "Qwen/Qwen2.5-Coder-7B"
    expert_cfg = ExpertConfig(mode="hub", model_path=model_name)
    expert = Expert(expert_cfg, prompt_function=create_map_prompt)
    map_expert = CodeBasedExpertWrapper(expert)
    print(question)
    t0 = time.perf_counter()
    answer = map_expert.chat(question)
    code = map_expert.get_code_block(answer)
    env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', False)
    rngs = jax.random.PRNGKey(42)
    rngs, reset_key = jax.random.split(rngs)
    obs, state = env.reset(reset_key)
    run_llm_code(code, state, module_name=module_name)
    elapsed = time.perf_counter() - t0
    print(f"query took {elapsed:.2f} s")
    return answer

if __name__ == "__main__":
    # print("- - - - -"*10)
    # test_map_expert_local()
    print("- - - - -"*10)
    test_map_expert_hub(question="Where is the nearest coal?", module_name="answer_code_coal_hub")
    print("- - - - -"*10)
    
    print("- - - - -"*10)
    test_map_expert_hub(question="What block is nearest to the agent?", module_name="nearest_block_hub")
    print("- - - - -"*10)
    
    print("- - - - -"*10)
    test_map_expert_hub(question="What block is nearest to the agent except grass?", module_name="nearest_block_hub_except_grass_hub")
    print("- - - - -"*10)
    
    print("- - - - -"*10)
    test_map_expert_hub(question="What blocks are around agent?", module_name="blocks_around_agent_hub")
    print("- - - - -"*10)
