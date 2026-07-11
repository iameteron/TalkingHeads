"""Shared OpenRouter HTTP client helpers (optional outbound proxy)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from openai import OpenAI

OPENROUTER_PROXY_ENV = "OPENROUTER_PROXY_URL"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_proxy_url() -> str | None:
    value = os.environ.get(OPENROUTER_PROXY_ENV, "").strip()
    return value or None


def _httpx_client(*, timeout: float):
    import httpx

    proxy = openrouter_proxy_url()
    kwargs: dict[str, Any] = {"timeout": timeout}
    if proxy:
        kwargs["proxy"] = proxy
    return httpx.Client(**kwargs)


def make_openai_client(
    *,
    api_key: str,
    base_url: str = DEFAULT_OPENROUTER_BASE_URL,
    timeout: float = 120.0,
) -> OpenAI:
    if openrouter_proxy_url():
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=_httpx_client(timeout=timeout),
        )
    return OpenAI(base_url=base_url, api_key=api_key)


def fetch_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 8.0,
) -> Any:
    proxy = openrouter_proxy_url()
    if proxy:
        import httpx

        response = httpx.get(url, headers=headers, timeout=timeout, proxy=proxy)
        response.raise_for_status()
        return response.json()

    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body[:300]}") from exc
