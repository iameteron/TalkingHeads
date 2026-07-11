from oracle.prompts.prompt_generation import create_mechanics_promt

from oracle.code_based_expert import CodeBasedExpertWrapper, Expert
from oracle.configs import ExpertConfig

from oracle.utils import run_llm_mechanics_code

import time


def test_mechanics_expert_local(question: str, module_name: str):
    model_name = "Qwen/Qwen2.5-Coder-7B"
    expert_cfg = ExpertConfig(mode="local", model_path=model_name)
    expert = Expert(expert_cfg, prompt_function=create_mechanics_promt)
    mechanics_expert = CodeBasedExpertWrapper(expert)

    t0 = time.perf_counter()
    print(question)

    answer = mechanics_expert.chat(question)
    code = mechanics_expert.get_code_block(answer)
    run_llm_mechanics_code(code, module_name=module_name)
    
    elapsed = time.perf_counter() - t0

    print(f"query took {elapsed:.2f} s")
    return answer

def test_mechanics_expert_hub(question: str, module_name: str):
    model_name = "Qwen/Qwen2.5-Coder-7B"
    expert_cfg = ExpertConfig(mode="hub", model_path=model_name)
    expert = Expert(expert_cfg, prompt_function=create_mechanics_promt)
    mechanics_expert = CodeBasedExpertWrapper(expert)

    t0 = time.perf_counter()
    print(question)

    answer = mechanics_expert.chat(question)
    
    code = mechanics_expert.get_code_block(answer)
    run_llm_mechanics_code(code, module_name=module_name)
    elapsed = time.perf_counter() - t0

    print(f"query took {elapsed:.2f} s")
    return answer


if __name__ == "__main__":
    print("- " * 10)
    test_mechanics_expert_hub(
        question="What achievements are required before collecting coal?",
        module_name="mechanics_coal_dependencies",
    )
    print("- " * 10)

    print("- " * 10)
    test_mechanics_expert_hub(
        question="What should I do before making an iron sword?",
        module_name="mechanics_iron_sword_dependencies",
    )
    print("- " * 10)

