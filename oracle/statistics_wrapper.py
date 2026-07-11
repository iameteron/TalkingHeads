class ExpertStatistics:
    """
    Simple statistics wrapper around an expert‑like object.

    Counts:
      - messages: how many times `chat` was called
      - failures: how many calls raised an exception
    """

    def __init__(self, expert):
        self.expert = expert
        self.messages = 0
        self.failures = 0

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped expert.

        This allows calling methods like `set_kwargs_for_run_function` on
        the wrapper transparently.
        """
        return getattr(self.expert, name)

    def chat(self, question, **kwargs):
        self.messages += 1
        try:
            # Delegate to the underlying expert and return its result.
            result = self.expert.chat(question, **kwargs)
        except Exception:
            # Count and re‑raise so existing error handling still works.
            self.failures += 1
            raise

        # Also treat the standard "failed to connect" sentinel as a failure,
        # even if no exception was propagated.
        if (
            isinstance(result, str)
            and result.strip().startswith("Failed to connect to code-based expert.")
        ):
            self.failures += 1

        return result

    def chat_with_retry(self, question, **kwargs):
        """
        Proxy for expert.chat_with_retry with the same statistics accounting.

        If the wrapped expert does not implement chat_with_retry, this falls
        back to a single chat call.
        """
        # Count this as a question as well.
        self.messages += 1

        # Prefer expert.chat_with_retry if it exists.
        target = getattr(self.expert, "chat_with_retry", None)
        if target is None:
            target = self.expert.chat

        try:
            result = target(question, **kwargs)
        except Exception:
            self.failures += 1
            raise

        # Same logic as in `chat`: count sentinel string as a failure.
        if (
            isinstance(result, str)
            and result.strip().startswith("Failed to connect to code-based expert.")
        ):
            self.failures += 1

        return result


class ActiveAgentStatistics:
    """
    Statistics wrapper around an ActiveAgent instance.

    Tracks:
      - total_calls: how many times `chat` or `chat_with_reasoning_history` was called
      - questions: how many responses contained "--- Q ---" (agent asked a question)
      - actions: how many responses contained "--- Act ---" (agent predicted an action)
      - failures: how many calls raised an exception
    """

    def __init__(self, agent):
        self.agent = agent
        self.total_calls = 0
        self.questions = 0
        self.actions = 0
        self.failures = 0

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped agent.
        """
        return getattr(self.agent, name)

    def _parse_and_count(self, result: str) -> None:
        """
        Parse the agent response and update question/action counters.
        """
        if not isinstance(result, str):
            return

        lower = result.lower()
        if (
            "--- act ---" in lower
            or "<action>" in lower
            or "<act>" in lower
            or "action:" in lower
        ):
            self.actions += 1
        elif (
            "--- q ---" in lower
            or "<question>" in lower
            or "<ask>" in lower
            or "<q>" in lower
            or "question:" in lower
        ):
            self.questions += 1

    def chat(
        self,
        user_message: str,
        system_message=None,
        gen=None,
        history=None,
    ) -> str:
        """
        Proxy for agent.chat with statistics tracking.
        """
        self.total_calls += 1
        try:
            result = self.agent.chat(
                user_message=user_message,
                system_message=system_message,
                gen=gen,
                history=history,
            )
            self._parse_and_count(result)
            return result
        except Exception:
            self.failures += 1
            raise

    def chat_with_reasoning_history(
        self,
        user_message: str,
        history=None,
        system_message=None,
        gen=None,
    ) -> tuple[str, list]:
        """
        Proxy for agent.chat_with_reasoning_history with statistics tracking.
        """
        self.total_calls += 1
        try:
            result_text, updated_history = self.agent.chat_with_reasoning_history(
                user_message=user_message,
                history=history,
                system_message=system_message,
                gen=gen,
            )
            self._parse_and_count(result_text)
            return result_text, updated_history
        except Exception:
            self.failures += 1
            raise
