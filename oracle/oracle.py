import logging
from typing import Dict, Optional, Sequence

from craftax.craftax_env import make_craftax_env_from_name
import jax
from oracle.intent import Intent
from oracle.prompts.prompt_generation import (
    create_map_prompt,
    create_mechanics_promt,
    create_action_prompt,
    create_question_prompt,
    create_goal_prompt,
    create_path_prompt,
    create_path_expert_prompt,
    normalize_world_mode,
)

from oracle.statistics_wrapper import ExpertStatistics
from oracle.utils import (
    run_llm_code,
    run_llm_mechanics_code,
    format_inventory_from_env_state,
    PathExpertPipeline,
    stringify_agent_position,
)

from oracle.expert_base import (
    CodeBasedExpertWrapper,
    Expert,
    GoalExpert,
)

from oracle.configs import (
    OracleConfig,
    ExpertConfig, 
    oracle_config_to_expert_config
)

logger = logging.getLogger(__name__)

TRIES_FOR_ANSWER = 1


class Oracle:
    """
    High‑level orchestrator:
    1. Takes a user question.
    2. Classifies it with `Intent` into a target expert.
    3. Builds the expert prompt.
    4. Calls the expert (local transformers or HF Hub, depending on config).
    5. Optionally executes the returned code.

    Logging (INFO level):
      - Intent decision
      - Query / prompt creation
      - Expert answer
    """

    def __init__(self, config: OracleConfig, *, world_mode: str | None = None):
        self.config = config
        self.world_mode = normalize_world_mode(world_mode)

        experts_configs_dict = oracle_config_to_expert_config(config)
        expert_factories = {
            "map_expert": ("map", experts_configs_dict["map"]),
            "mechanics_expert": ("mechanics", experts_configs_dict["mechanics"]),
            "action_expert": ("action", experts_configs_dict["action"]),
            "question_expert": ("question", experts_configs_dict["question"]),
            "path_expert_helper": ("path_expert_helper", experts_configs_dict["path_expert_helper"]),
            "path_expert": ("path_expert", experts_configs_dict["path_expert"]),
        }
        self.experts = {
            expert_name: self.make_expert(expert_type, expert_config)
            for expert_name, (expert_type, expert_config) in expert_factories.items()
        }
        self.map_expert = self.experts["map_expert"]
        self.mechanics_expert = self.experts["mechanics_expert"]
        self.action_expert = self.experts["action_expert"]
        self.question_expert = self.experts["question_expert"]
        self.path_expert_helper = self.experts["path_expert_helper"]
        self.path_expert = self.experts["path_expert"]
        
        # Goal expert orchestrates question, map, and mechanics experts.
        self.goal_expert = GoalExpert(
            experts_configs_dict["goal"],
            map_expert=self.map_expert,
            mechanics_expert=self.mechanics_expert,
            question_expert=self.question_expert,
            action_expert=self.action_expert,
            gen_config=config.gen_config,
            world_mode=self.world_mode,
        )
        self.path_pipeline = PathExpertPipeline(
            helper_expert=self.path_expert_helper,
            path_expert=self.path_expert,
            operator_max_answer_chars=config.operator_max_answer_chars,
        )
        self.intent = None

    def make_expert(self, 
                    expert_type: str, 
                    expert_config: ExpertConfig):
        
        expert_to_prompt_function = {
            "map": create_map_prompt,
            "mechanics": create_mechanics_promt,
            "action": create_action_prompt,
            "question": create_question_prompt,
            "goal": create_goal_prompt,
            "path_expert_helper": create_path_prompt,
            "path_expert": create_path_expert_prompt,
        }
        expert_to_run_function = {
            "map": run_llm_code,
            "mechanics": run_llm_mechanics_code,
        }
        prompt_fn = expert_to_prompt_function.get(expert_type)
        if prompt_fn is not None:
            world_mode = self.world_mode
            base_prompt_fn = prompt_fn

            def prompt_fn_wrapped(*args, _base_prompt_fn=base_prompt_fn, **kwargs):
                kwargs.setdefault("world_mode", world_mode)
                return _base_prompt_fn(*args, **kwargs)

            prompt_fn = prompt_fn_wrapped

        # The Expert expects a full ExpertConfig, not individual fields.
        expert = Expert(
            expert_config=expert_config,
            gen_config=self.config.gen_config,
            tries=TRIES_FOR_ANSWER,
            prompt_function=prompt_fn,
        )
        if expert_type in ("map", "mechanics"):
            expert =  CodeBasedExpertWrapper(expert, 
                                          run_function=expert_to_run_function[expert_type],
                                          kwarfs_for_run_function={})
        
        expert = ExpertStatistics(expert)
        return expert


    def _decide_intent(
        self,
        question: str,
        allowed_experts: Optional[Sequence[str]] = None,
    ) -> str:
        """Run the intent classifier and return the target expert label."""
        if self.intent is None:
            self.intent = Intent()
        return self.intent.predict(question, allowed_experts=allowed_experts)[0]

    def _call_goal_expert(self, question: str, env_state) -> str:
        logger.info("Expert: GOAL | Question: %s", question)
        return self.goal_expert.chat(
            question,
            agent_position=stringify_agent_position(env_state),
            env_state=env_state,
        )

    def _call_path_expert(
        self,
        question: str,
        env_state,
        goal_hint: Optional[str] = None,
        operator_max_answer_chars_override: Optional[int] = None,
        target_location: Optional[Sequence[int]] = None,
    ) -> str:
        logger.info("Expert: PATH | Question: %s", question)
        return self.path_pipeline.answer(
            question,
            env_state,
            goal_hint=goal_hint,
            operator_max_answer_chars_override=operator_max_answer_chars_override,
            target_location=target_location,
        )

    def _call_question_expert(self, question: str, _env_state) -> str:
        logger.info("Expert: QUESTION | Question: %s", question)
        return self.question_expert.chat_with_retry(question)

    def _call_action_expert(self, question: str, env_state) -> str:
        logger.info("Expert: ACTION | Question: %s", question)
        inventory_str = format_inventory_from_env_state(env_state, world_mode=self.world_mode)
        return self.action_expert.chat_with_retry(question, inventory=inventory_str)

    def _call_map_expert(self, question: str, env_state, module_name: Optional[str]) -> str:
        logger.info("Expert: MAP | Question: %s", question)
        self.map_expert.set_kwargs_for_run_function({
            "state": env_state,
            "module_name": module_name,
        })
        return self.map_expert.chat_with_retry(question)

    def _call_mechanics_expert(self, question: str, module_name: Optional[str]) -> str:
        logger.info("Expert: MECHANICS | Question: %s", question)
        self.mechanics_expert.set_kwargs_for_run_function({
            "module_name": module_name,
        })
        return self.mechanics_expert.chat_with_retry(question)

    def answer_gm_only(
        self,
        question: str,
        *,
        module_name: Optional[str] = None,
        env_state=None,
        run_code: bool = False,
        allowed_experts: Optional[Sequence[str]] = None,
    ) -> str:
        _ = (module_name, run_code, allowed_experts)
        return self._call_goal_expert(question, env_state)
            
        
    def answer(
        self,
        question: str,
        *,
        module_name: Optional[str] = None,
        env_state=None,
        run_code: bool = False,
        allowed_experts: Optional[Sequence[str]] = None,
        forced_expert: Optional[str] = None,
        goal_hint: Optional[str] = None,
        target_location: Optional[Sequence[int]] = None,
    ) -> str:
        if forced_expert:
            intent_label = Intent.normalize_expert_name(forced_expert)
        else:
            intent_label = self._decide_intent(question, allowed_experts=allowed_experts)

        handlers = {
            Intent.PATH_EXPERT: lambda: self._call_path_expert(
                question,
                env_state,
                goal_hint=goal_hint,
                target_location=target_location,
            ),
            Intent.QUESTION_EXPERT: lambda: self._call_question_expert(question, env_state),
            Intent.ACTION_EXPERT: lambda: self._call_action_expert(question, env_state),
            Intent.GOAL_EXPERT: lambda: self._call_goal_expert(question, env_state),
            Intent.MAP_EXPERT: lambda: self._call_map_expert(question, env_state, module_name),
            Intent.MECHANICS_EXPERT: lambda: self._call_mechanics_expert(question, module_name),
        }
        if intent_label not in handlers:
            raise ValueError(f"Unsupported intent label: {intent_label}")
        return handlers[intent_label]()

    def return_statistics(self) -> Dict[str, Dict[str, int]]:
        """
        Aggregate simple statistics for all wrapped experts.

        For each expert we expose:
          - messages: how many times `chat` / `chat_with_retry` was called
          - failures: how many calls raised an exception
        Additionally, a `total` entry sums these counts over all experts.
        """

        def _stats_for(expert_wrapper) -> Dict[str, int]:
            return {
                "messages": getattr(expert_wrapper, "messages", 0),
                "failures": getattr(expert_wrapper, "failures", 0),
            }

        map_stats = _stats_for(self.experts["map_expert"])
        mechanics_stats = _stats_for(self.experts["mechanics_expert"])
        question_stats = _stats_for(self.experts["question_expert"])
        action_stats = _stats_for(self.experts["action_expert"])
        path_stats = _stats_for(self.experts["path_expert"])
        path_helper_stats = _stats_for(self.experts["path_expert_helper"])

        total_messages = (
            map_stats["messages"]
            + mechanics_stats["messages"]
            + question_stats["messages"]
            + action_stats["messages"]
            + path_helper_stats["messages"]
            + path_stats["messages"]
        )
        total_failures = (
            map_stats["failures"]
            + mechanics_stats["failures"]
            + question_stats["failures"]
            + action_stats["failures"]
            + path_helper_stats["failures"]
            + path_stats["failures"]
        )

        return {
            "total": {
                "messages": total_messages,
                "failures": total_failures,
            },
            "map_expert": map_stats,
            "mechanics_expert": mechanics_stats,
            "question_expert": question_stats,
            "action_expert": action_stats,
            "path_expert_helper": path_helper_stats,
            "path_expert": path_stats,
        }


def configure_logging(level: int = logging.INFO):
    """
    Helper to configure basic logging for quick scripts.

    Logs to stdout with timestamps and the module name.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


if __name__ == "__main__":
    # Example CLI usage:
    #   python -m oracle.oracle "Where is the nearest coal?"
    # Optional: CONFIG_PATH=/path/to/config.yaml python -m oracle.oracle "question"
    import os
    import sys

    from oracle.config_loader import load_config

    configure_logging()
    if len(sys.argv) < 2:
        print("Usage: python -m oracle.oracle 'Your question here'")
        sys.exit(1)

    question = sys.argv[1]
    config_path = os.environ.get("CONFIG_PATH")
    config = load_config(path=config_path)
    oracle = Oracle(config)
    
    env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', False)
    rngs = jax.random.PRNGKey(42)
    rngs, reset_key = jax.random.split(rngs)
    obs, state = env.reset(reset_key)
    
    # For simple CLI usage we do not execute the code, we just print the answer.
    answer = False
    while not answer:
        try:
            result = oracle.answer(question,
                                   run_code=True, 
                                   module_name="answer_test",
                                   env_state=state)
            answer = True
        except Exception as e:
            print(f"Error: {e}")

        