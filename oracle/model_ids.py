from __future__ import annotations

DEFAULT_HF_INFERENCE_PROVIDER = "nscale"

# Suffixes after ":" used by Hugging Face Inference Providers (not OpenRouter tiers).
HF_INFERENCE_PROVIDER_SUFFIXES = frozenset(
    {
        "nscale",
        "novita",
        "featherless-ai",
        "featherless",
        "together",
        "fireworks-ai",
        "hyperbolic",
        "groq",
        "cerebras",
        "sambanova",
        "cohere",
        "replicate",
    }
)


def _split_provider_suffix(model_id: str) -> tuple[str, str | None]:
    if ":" not in model_id:
        return model_id, None
    base, suffix = model_id.rsplit(":", 1)
    if not base:
        return model_id, None
    return base, suffix


def is_hf_inference_provider_suffix(suffix: str | None) -> bool:
    if not suffix:
        return False
    return suffix.strip().lower() in HF_INFERENCE_PROVIDER_SUFFIXES


def is_hf_hub_model_id(model_id: str) -> bool:
    name = str(model_id or "").strip()
    if not name or name.startswith(("./", "/")):
        return False
    return "/" in name


def strip_hf_inference_provider_suffix(model_id: str) -> str:
    """Remove HF provider suffixes such as ``:nscale`` (for OpenRouter and display)."""
    name = str(model_id or "").strip()
    if not name:
        return name
    base, suffix = _split_provider_suffix(name)
    if is_hf_inference_provider_suffix(suffix):
        return base
    return name


def to_openrouter_model_id(model_id: str) -> str:
    """Convert HF-style ids to OpenRouter slugs (lowercase org/model, no HF suffix)."""
    name = strip_hf_inference_provider_suffix(model_id)
    if not name or not is_hf_hub_model_id(name):
        return name
    org, model = name.split("/", 1)
    return f"{org.strip().lower()}/{model.strip().lower()}"


def ensure_hf_inference_provider_suffix(
    model_id: str,
    provider: str = DEFAULT_HF_INFERENCE_PROVIDER,
) -> str:
    """Ensure HF hub model ids include a provider suffix such as ``:nscale``."""
    name = str(model_id or "").strip()
    if not name or not is_hf_hub_model_id(name):
        return name
    base, suffix = _split_provider_suffix(name)
    if is_hf_inference_provider_suffix(suffix):
        return name
    provider_norm = str(provider or DEFAULT_HF_INFERENCE_PROVIDER).strip().lower()
    return f"{base}:{provider_norm}"


def adapt_model_id_for_mode(
    model_id: str,
    mode: str,
    *,
    hf_provider: str = DEFAULT_HF_INFERENCE_PROVIDER,
) -> str:
    """
    Normalize a stored model id for the active provider mode.

    - ``hub``: append ``:nscale`` (or ``hf_provider``) when missing
    - ``openrouter``: strip HF inference suffixes such as ``:nscale``
    - ``local`` and other modes: unchanged
    """
    name = str(model_id or "").strip()
    if not name:
        return name
    mode_norm = str(mode or "").strip().lower()
    if mode_norm == "openrouter":
        return to_openrouter_model_id(name)
    if mode_norm == "hub":
        return ensure_hf_inference_provider_suffix(name, hf_provider)
    return name
