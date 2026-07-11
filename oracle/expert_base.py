"""
Expert abstractions used by `Oracle`.

There are two layers:
- `Expert` – wraps an underlying LLM agent (local or HF Hub) and optionally
  applies a prompt-construction function to a user question.
- `CodeBasedExpertWrapper` – thin helper around an `Expert` that knows how to
  extract a Python code block from the model's answer. Execution of that code
  is done by the caller (e.g. `Oracle`, tests).
"""
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)
from oracle.prompts.prompt_generation import create_goal_prompt
from oracle.agent_base import (
    HFLLMAgentBase,
    HFHubLLMAgentBase,
    OpenRouterLLMAgentBase,
    load_qwen,
)
from oracle.configs import ExpertConfig, GenConfig
from oracle.model_ids import adapt_model_id_for_mode


class Expert:
    """
    High-level wrapper over an LLM agent.

    - Chooses between local transformers vs HF Hub inference.
    - Optionally applies a `prompt_function(question) -> str` to build the
      actual model prompt from a high-level question.
    """

    def __init__(
        self,
        expert_config: ExpertConfig,
        *,
        gen_config: Optional[GenConfig] = None,
        tries: int = 3,
        prompt_function=None,
    ):
        """
        Args:
            expert_config: Low-level configuration describing how to talk to the
                underlying model (mode, model_path, api_key, base_url).
            gen_config: Optional text-generation configuration shared across experts.
            tries: How many times to retry `chat` on transient errors.
            prompt_function: Optional callable `(question, **kwargs) -> str` that
                converts a high-level question into the actual LLM prompt.
        """
        self.tries = tries
        self.expert_config = expert_config
        self.gen_config = gen_config
        self.prompt_function = prompt_function
        self.agent = self._make_agent()

    def _make_agent(self):
        mode = self.expert_config.mode
        model_path = adapt_model_id_for_mode(self.expert_config.model_path, mode)
        if mode == "local":
            tokenizer, model = load_qwen(model_path)
            return HFLLMAgentBase(tokenizer, model)
        if mode == "hub":
            return HFHubLLMAgentBase(
                model_path,
                api_key=self.expert_config.api_key,
                base_url=self.expert_config.base_url,
                provider=self.expert_config.provider,
            )
        if mode == "openrouter":
            return OpenRouterLLMAgentBase(
                model_path,
                api_key=self.expert_config.api_key,
                base_url=self.expert_config.base_url or "https://openrouter.ai/api/v1",
            )
        raise ValueError(f"Unknown expert mode: {mode!r}")

    def build_prompt(self, question: str, **kwargs) -> str:
        if self.prompt_function is None:
            return question
        return self.prompt_function(question, **kwargs)

    def chat(self, question: str, **kwargs) -> str:
        """
        Main entry point: takes a high-level question string.
        """
        prompt = self.build_prompt(question, **kwargs)
        return self.agent.chat(prompt, gen=self.gen_config)

    def chat_with_retry(self, question: str, **kwargs) -> str:
        """
        Same as `chat`, but retries a few times on transient errors.
        """
        last_error: Optional[Exception] = None
        for _ in range(self.tries):
            try:
                return self.chat(question, **kwargs)
            except Exception as e:  # pragma: no cover - primarily defensive
                last_error = e
                logger.exception("Error when calling expert")
        return "Failed to connect to code-based expert."


class CodeBasedExpertWrapper:
    """
    Helper for code-based experts.

    It delegates to an underlying `Expert` to obtain the raw answer string,
    and exposes a utility to pull out the Python code block.
    """

    def __init__(
        self,
        expert: Expert,
        run_function=None,
        kwarfs_for_run_function=None,
    ):
        """
        Args:
            expert: Underlying `Expert` instance to obtain raw answers from.
            run_function: Callable that will be used to execute the extracted
                Python code block, e.g. `run_llm_code` or `run_llm_mechanics_code`.
            kwarfs_for_run_function: Initial kwargs dict passed to `run_function`.
        """
        self.expert = expert
        self.run_function = run_function
        # Keep the public attribute name for backwards compatibility, but make
        # sure we always store a proper dict internally.
        self.kwargs_for_run_function = kwarfs_for_run_function or {}
    
    def set_kwargs_for_run_function(self, kwargs):
        self.kwargs_for_run_function = kwargs or {}
        
    def chat(self, question: str) -> str:
        """Return the raw LLM answer for the given high-level question."""
        raw_answer = self.expert.chat(question)
        code_block = self.get_code_block(raw_answer)
        parsed_answer = self.run_function(
            code_block,
            **self.kwargs_for_run_function,
        )
        return parsed_answer

    def chat_with_retry(self, question: str) -> str:
        """
        Same interface as `Expert.chat_with_retry`, but also runs the code.

        We mirror the retry behaviour of `Expert.chat_with_retry`, but call our
        own `chat` method so that the returned value is the parsed / executed
        answer instead of the raw LLM text.
        """
        last_error: Optional[Exception] = None
        tries = getattr(self.expert, "tries", 3)
        for _ in range(tries):
            try:
                return self.chat(question)
            except Exception as e:  # pragma: no cover - defensive
                last_error = e
                logger.exception("Error when calling code-based expert")
        return "Failed to connect to code-based expert."

    def get_code_block(self, answer: str) -> str:
        """
        Extract the first ```python ... ``` block from the LLM answer.
        """
        return answer.split("```python", 1)[1].split("```", 1)[0]

from oracle.utils import (
    run_llm_code,
    run_llm_mechanics_code,
    format_inventory_from_env_state,
)


class GoalExpert(Expert):
    def __init__(
        self,
        expert_config: ExpertConfig,
        *,
        map_expert,
        mechanics_expert,
        question_expert,
        action_expert=None,
        gen_config: Optional[GenConfig] = None,
        tries: int = 3,
        world_mode: str | None = None,
    ):
        # GoalExpert does not use a simple prompt_function; it builds its own
        # aggregation prompt from other experts' answers, so we pass None here.
        super().__init__(
            expert_config,
            gen_config=gen_config,
            tries=tries,
            prompt_function=None,
        )
        self.map_expert = map_expert
        self.mechanics_expert = mechanics_expert
        self.question_expert = question_expert
        # Optional plain-text expert that explains concrete actions based on mechanics.
        self.action_expert = action_expert
        from oracle.prompts.prompt_generation import normalize_world_mode

        self.world_mode = normalize_world_mode(world_mode)

    def _generate_expert_questions(self, goal: str) -> Dict[str, str]:
        """
        Use the question expert to generate questions for the map, mechanics,
        and (optionally) action experts, given a player goal.

        Returns a dict with keys: map_expert_question, mechanics_expert_question,
        and action_expert_question.
        """
        # Local imports to avoid any potential circular-import problems.
        from oracle.prompts.prompt_generation import create_question_prompt
        from oracle.utils import parse_question_expert_response, ensure_expert_questions

        built_prompt = create_question_prompt(goal, world_mode=self.world_mode)
        logger.info(
            "Expert: QUESTION (sub-call from Goal) | Goal: %s | Prompt: %s",
            goal,
            built_prompt[:200] + ("..." if len(built_prompt) > 200 else ""),
        )

        # Pass the raw goal; question_expert applies create_question_prompt itself.
        if hasattr(self.question_expert, "chat_with_retry"):
            response = self.question_expert.chat_with_retry(goal)
        else:
            response = self.question_expert.chat(goal)

        return ensure_expert_questions(parse_question_expert_response(response), goal)

    def chat(self, goal: str,
             agent_position: Optional[str] = "not provided", 
             env_state=None) -> str:
        """
        Main entry point for the goal expert.

        Parameters
        ----------
        goal: str
            The agent's question for the goal expert, e.g. "How do I collect coal?"
        agent_position: Optional[str]
            Optional textual description of the agent position to inject into the
            aggregation prompt. If not provided, "not provided" is used.

        Returns
        -------
        str
            A short answer (3–4 sentences) to the agent question, synthesized from
            map, mechanics, and action expert answers. See `goal_prompt.txt`.
        """

        # 1) Ask the question expert what to ask the other experts.
        questions = self._generate_expert_questions(goal)
        map_q = questions.get("map_expert_question", "").strip()
        mech_q = questions.get("mechanics_expert_question", "").strip()
        action_q = questions.get("action_expert_question", "").strip()

        # Build a human-readable inventory string from env_state (if available),
        # to pass into the action expert so it can reason about prerequisites.
        inventory_str = format_inventory_from_env_state(env_state, world_mode=self.world_mode)

        # 2) Ask the map and mechanics experts those questions.
        map_answer = ""
        mech_answer = ""
        action_answer = ""

        if map_q:
            logger.info("Expert: MAP (sub-call from Goal) | Question: %s", map_q)
            self.map_expert.set_kwargs_for_run_function({"state": env_state,
                                                            "module_name": "map_expert_for_goal_expert"})
            try:
                if hasattr(self.map_expert, "chat_with_retry"):
                    map_answer = self.map_expert.chat_with_retry(map_q)
                else:
                    map_answer = self.map_expert.chat(map_q)
                logger.info("Expert: MAP (sub-call from Goal) | Answer received: %s", map_answer[:200] + ("..." if len(map_answer) > 200 else ""))
            except Exception as e:
                logger.error("Expert: MAP (sub-call from Goal) | Error: %s", str(e))
                map_answer = f"Error retrieving map information: {str(e)}"
        else:
            logger.warning("Expert: MAP (sub-call from Goal) | No question generated")

        if mech_q:
            logger.info("Expert: MECHANICS (sub-call from Goal) | Question: %s", mech_q)
            self.mechanics_expert.set_kwargs_for_run_function({
                                                            "module_name": "mechanics_expert_for_goal_expert"})
            try:
                if hasattr(self.mechanics_expert, "chat_with_retry"):
                    mech_answer = self.mechanics_expert.chat_with_retry(mech_q)
                else:
                    mech_answer = self.mechanics_expert.chat(mech_q)
                logger.info("Expert: MECHANICS (sub-call from Goal) | Answer received: %s", mech_answer[:200] + ("..." if len(mech_answer) > 200 else ""))
            except Exception as e:
                logger.error("Expert: MECHANICS (sub-call from Goal) | Error: %s", str(e))
                mech_answer = f"Error retrieving mechanics information: {str(e)}"
        else:
            logger.warning("Expert: MECHANICS (sub-call from Goal) | No question generated")

        # Optionally, ask the action expert for concrete action guidance.
        if self.action_expert is not None and action_q:
            try:
                if hasattr(self.action_expert, "chat_with_retry"):
                    action_answer = self.action_expert.chat_with_retry(
                        action_q, inventory=inventory_str
                    )
                else:
                    action_answer = self.action_expert.chat(
                        action_q, inventory=inventory_str
                    )
                logger.info(
                    "Expert: ACTION (sub-call from Goal) | Answer received: %s",
                    action_answer[:200] + ("..." if len(action_answer) > 200 else ""),
                )
            except Exception as e:
                logger.error("Expert: ACTION (sub-call from Goal) | Error: %s", str(e))
                action_answer = f"Error retrieving action information: {str(e)}"

        # Validate that we have at least some answers
        if (
            not map_answer.strip()
            and not mech_answer.strip()
            and not action_answer.strip()
        ):
            error_msg = "Failed to retrieve answers from map, mechanics, and action experts."
            logger.error("Expert: GOAL | %s", error_msg)
            return error_msg

        # 3) Build the aggregation prompt using the goal expert template.
        goal_prompt = create_goal_prompt(
            goal=goal,
            question_1=map_q,
            answer_1=map_answer,
            question_2=mech_q,
            answer_2=mech_answer,
            action_answer=action_answer,
            agent_position=agent_position,
            world_mode=self.world_mode,
        )
        
        logger.info("Expert: GOAL | Prompt constructed with map_answer length: %d, mech_answer length: %d", 
                   len(map_answer), len(mech_answer))

        # 4) Call the underlying LLM agent with the constructed prompt (with retry).
        last_error: Optional[Exception] = None
        for attempt in range(self.tries):
            try:
                result = self.agent.chat(goal_prompt, gen=self.gen_config)
                logger.info("Expert: GOAL | Final answer generated: %s", result[:200] + ("..." if len(result) > 200 else ""))
                return result
            except Exception as e:
                last_error = e
                logger.warning("Expert: GOAL | Attempt %d/%d failed: %s", attempt + 1, self.tries, str(e))
        
        # If all retries failed
        err_text = str(last_error or "unknown error")
        if "model_not_supported" in err_text or "not supported by any provider" in err_text:
            error_msg = (
                "Goal expert model is not available on your HF/OpenRouter account. "
                "In Settings, set goal/question/action expert models to one you have enabled "
                "(for HF hub use a provider suffix, e.g. Qwen/Qwen3-4B-Instruct-2507:nscale), "
                "then reset session config or restart the server."
            )
        else:
            error_msg = f"Failed to generate goal answer after {self.tries} attempts. Last error: {err_text}"
        logger.error("Expert: GOAL | %s", error_msg)
        return error_msg
    

