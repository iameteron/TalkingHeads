from typing import Any, Optional, Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from oracle.configs import GenConfig
from oracle.openrouter_client import make_openai_client



def load_qwen(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Prefer GPU, half precision
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        # CPU fallback: still try bf16 if supported, otherwise fp32
        dtype = torch.bfloat16 if torch.backends.cpu.has_bf16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    model.eval()
    return tokenizer, model

class HFLLMAgentBase:
    def __init__(self, tokenizer: Any, model: Any):
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device

    def chat(
        self,
        user_message: str,
        system_message: str = "You are a helpful assistant.",
        gen: Optional[GenConfig] = None,
    ) -> str:
        gen = gen or GenConfig()

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Build generation kwargs cleanly
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=gen.max_new_tokens,
            do_sample=gen.do_sample,
            use_cache=gen.use_cache,
        )
        if gen.do_sample:
            gen_kwargs["temperature"] = gen.temperature
            gen_kwargs["top_p"] = gen.top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

from huggingface_hub import InferenceClient
import os

class HFHubLLMAgentBase:
    """
    Agent that calls HF Hub Inference (chat.completions) instead of local transformers.
    """
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        default_system_message: str = "You are a helpful assistant.",
    ):
        # api_key: pass explicitly or via HF_TOKEN env
        api_key = api_key or os.environ.get("HF_TOKEN")
        if not api_key:
            raise ValueError("HF_TOKEN is not set and api_key was not provided.")

        parsed_model_name = model_name
        parsed_provider = provider
        if parsed_provider is None and ":" in model_name:
            maybe_model, maybe_provider = model_name.rsplit(":", 1)
            if maybe_model and maybe_provider:
                parsed_model_name = maybe_model
                parsed_provider = maybe_provider

        self.model_name = parsed_model_name
        self.default_system_message = default_system_message
        self.client = InferenceClient(
            api_key=api_key,
            base_url=base_url,
            provider=parsed_provider,
        )

    def chat(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        gen: Optional[GenConfig] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        history: optional list of {"role": "...", "content": "..."} that will go before the new user_message.
        """
        gen = gen or GenConfig()
        system_message = system_message or self.default_system_message

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_message}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        # Save messages to file
        with open("messages.txt", "a") as f:
            f.write(f"{messages}\n\n")

        # Map your GenConfig to HF chat.completions params
        create_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": gen.max_new_tokens,  # HF uses max_tokens for the completion part
        }

        # If do_sample=False -> keep deterministic (don’t send sampling params)
        if gen.do_sample:
            create_kwargs["temperature"] = gen.temperature
            create_kwargs["top_p"] = gen.top_p
       
        completion = self.client.chat.completions.create(**create_kwargs)

        # huggingface_hub returns message object; extract the content string
        msg = completion.choices[0].message
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")

        return (content or "").strip()


class OpenRouterLLMAgentBase:
    """
    Agent that calls OpenRouter (OpenAI-compatible chat.completions API).
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        default_system_message: str = "You are a helpful assistant.",
    ):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set and api_key was not provided.")

        self.model_name = model_name
        self.default_system_message = default_system_message
        self.client = make_openai_client(api_key=api_key, base_url=base_url)

    def chat(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        gen: Optional[GenConfig] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        gen = gen or GenConfig()
        system_message = system_message or self.default_system_message

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_message}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        create_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": gen.max_new_tokens,
        }
        if gen.do_sample:
            create_kwargs["temperature"] = gen.temperature
            create_kwargs["top_p"] = gen.top_p

        completion = self.client.chat.completions.create(**create_kwargs)
        msg = completion.choices[0].message
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")

        return (content or "").strip()
    
