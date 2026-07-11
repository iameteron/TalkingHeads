from __future__ import annotations


def short_model_name(raw: str) -> str:
    """
    Canonical model label for display and aggregation.

    Strips HuggingFace org prefix (``Qwen/``) and inference provider suffix
    (``:nscale``, ``:novita``, etc.). Case is preserved in the short name;
    use :func:`canonical_model_key` for grouping keys.
    """
    name = str(raw or "").strip()
    if not name:
        return "unknown"
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    if ":" in name:
        name = name.split(":", 1)[0]
    return name or "unknown"


def canonical_model_key(raw: str) -> str:
    """Case-insensitive aggregation key for a model id."""
    return short_model_name(raw).lower()
