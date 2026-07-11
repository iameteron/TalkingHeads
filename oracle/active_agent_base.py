import os
import re
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient

from oracle.configs import GenConfig
from oracle.model_ids import adapt_model_id_for_mode
from oracle.openrouter_client import make_openai_client


_IMAGE_MARKER_RE = re.compile(
    r"\[\[image:(data:image/(?:png|jpeg|jpg|webp);base64,[A-Za-z0-9+/=\r\n]+)\]\]",
    re.IGNORECASE,
)


def _content_from_image_markers(user_message: str, *, multimodal: bool) -> Any:
    """
    Convert ARC image markers into OpenAI-compatible multimodal content.

    Text-only providers receive the same prompt with a short placeholder, so
    prompts remain readable instead of sending a huge base64 blob as text.
    """
    text = str(user_message or "")
    image_urls: list[str] = []

    def replace_marker(match: re.Match[str]) -> str:
        image_urls.append(match.group(1).replace("\n", "").replace("\r", ""))
        return "[Attached ARC frame image]"

    text_without_markers = _IMAGE_MARKER_RE.sub(replace_marker, text)
    if not multimodal or not image_urls:
        return text_without_markers

    content: list[dict[str, Any]] = [{"type": "text", "text": text_without_markers}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in image_urls
    )
    return content


def _log_safe_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    safe_messages: list[dict[str, Any]] = []
    for msg in messages:
        copied = dict(msg)
        content = copied.get("content")
        if isinstance(content, list):
            safe_parts: list[dict[str, Any]] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    safe_parts.append({"type": "image_url", "image_url": "[omitted]"})
                else:
                    safe_parts.append(part)
            copied["content"] = safe_parts
        safe_messages.append(copied)
    return safe_messages


class ActiveAgent:
    """
    Agent that can use Hub or OpenRouter providers.
    Supports reasoning models on OpenRouter via extra_body when reasoning=True.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_system_message: str = "You are a helpful assistant.",
        reasoning: bool = False,
        mode: str = "openrouter",
    ):
        mode_norm = str(mode or "").strip().lower()
        if mode_norm not in {"hub", "openrouter"}:
            raise ValueError("ActiveAgent mode must be 'hub' or 'openrouter'.")

        if mode_norm == "hub":
            api_key = api_key or os.environ.get("HF_TOKEN")
            if not api_key:
                raise ValueError("HF_TOKEN is not set and api_key was not provided.")
            self.client = InferenceClient(api_key=api_key, base_url=base_url or None)
        else:
            api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY is not set and api_key was not provided.")
            self.client = make_openai_client(
                api_key=api_key,
                base_url=base_url or "https://openrouter.ai/api/v1",
            )

        self.mode = mode_norm
        self.model_name = adapt_model_id_for_mode(model_name, mode_norm)
        self.default_system_message = default_system_message
        self.reasoning = reasoning
        self.actions_history: List[str] = []
        self.consecutive_questions_count: int = 0

    def record_action(self, action_str: str) -> None:
        """Append an action string (as executed in one tick) to actions_history; resets consecutive questions."""
        self.actions_history.append(action_str)
        self.consecutive_questions_count = 0

    def record_question(self) -> None:
        """Record that the agent asked a question (increments consecutive questions count)."""
        self.consecutive_questions_count += 1

    def clear_actions_history(self) -> None:
        """Clear the list of recorded actions (e.g. when agent asks a question to remove repeated-actions warning)."""
        self.actions_history.clear()

    def reset_for_new_episode(self) -> None:
        """Clear actions history and consecutive questions count (call at start of a new run/episode)."""
        self.actions_history.clear()
        self.consecutive_questions_count = 0

    def chat(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        gen: Optional[GenConfig] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Send a chat message and return the assistant response.
        history: optional list of {"role": "...", "content": "...", ...} (may include
                 reasoning_details for multi-turn reasoning).
        """
        gen = gen or GenConfig()
        system_message = system_message or self.default_system_message

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_message}]
        if history:
            messages.extend(history)
        user_content = _content_from_image_markers(
            user_message,
            multimodal=self.mode == "openrouter",
        )
        messages.append({"role": "user", "content": user_content})
        if os.environ.get("ACTIVE_AGENT_DEBUG_MESSAGES", "").strip().lower() in {"1", "true", "yes", "on"}:
            with open("messages.txt", "w", encoding="utf-8") as f:
                f.write(f"Messages: {_log_safe_messages(messages)}")
        create_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": gen.max_new_tokens,
        }
        if gen.do_sample:
            create_kwargs["temperature"] = gen.temperature
            create_kwargs["top_p"] = gen.top_p

        if self.reasoning and self.mode == "openrouter":
            create_kwargs["extra_body"] = {"reasoning": {"enabled": True}}

        completion = self.client.chat.completions.create(**create_kwargs)
        msg = completion.choices[0].message

        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        
        if os.environ.get("ACTIVE_AGENT_DEBUG_MESSAGES", "").strip().lower() in {"1", "true", "yes", "on"}:
            print(f"Content: {content}")
        return (content or "").strip()

    def chat_with_reasoning_history(
        self,
        user_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        system_message: Optional[str] = None,
        gen: Optional[GenConfig] = None,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Chat and return (response_text, updated_history).
        The returned history includes the assistant reply with reasoning_details,
        so the model can continue reasoning in the next turn.
        """
        gen = gen or GenConfig()
        system_message = system_message or self.default_system_message

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_message}]
        if history:
            messages.extend(history)
        user_content = _content_from_image_markers(
            user_message,
            multimodal=self.mode == "openrouter",
        )
        messages.append({"role": "user", "content": user_content})

        create_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": gen.max_new_tokens,
        }
        if self.mode == "openrouter":
            create_kwargs["extra_body"] = {"reasoning": {"enabled": True}}
        if gen.do_sample:
            create_kwargs["temperature"] = gen.temperature
            create_kwargs["top_p"] = gen.top_p

        completion = self.client.chat.completions.create(**create_kwargs)
        msg = completion.choices[0].message

        content = getattr(msg, "content", None) or ""
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if hasattr(msg, "reasoning_details") and msg.reasoning_details is not None:
            assistant_msg["reasoning_details"] = msg.reasoning_details

        new_history = (history or []) + [{"role": "user", "content": user_content}, assistant_msg]
        return content.strip(), new_history
