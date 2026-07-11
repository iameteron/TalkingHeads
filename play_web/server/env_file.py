"""Load and persist API keys in the project-root .env file."""

from __future__ import annotations

import os
import re
from pathlib import Path

MANAGED_KEYS = ("HF_TOKEN", "OPENROUTER_API_KEY")


def _bool_from_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def should_persist_api_keys_to_env() -> bool:
    """
    Controls whether API keys from UI are written into project .env.

    - local/dev default: enabled
    - production can disable by setting PLAY_WEB_PERSIST_KEYS_TO_ENV=false
    """
    return _bool_from_env("PLAY_WEB_PERSIST_KEYS_TO_ENV", default=True)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def env_path() -> Path:
    return project_root() / ".env"


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _quote(value: str) -> str:
    if re.search(r'[\s#="\']', value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None
    key, _, raw_value = stripped.partition("=")
    key = key.strip()
    if not key:
        return None
    return key, _unquote(raw_value)


def read_managed_keys_from_file() -> dict[str, str]:
    path = env_path()
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if key in MANAGED_KEYS and value:
            values[key] = value
    return values


def load_project_env(*, override: bool = False) -> None:
    for key, value in read_managed_keys_from_file().items():
        if override or not (os.environ.get(key) or "").strip():
            os.environ[key] = value


def get_api_secret(key: str) -> str:
    """Return a managed API key from the environment or .env file."""
    if key not in MANAGED_KEYS:
        raise ValueError(f"Unsupported env key: {key}")
    value = (os.environ.get(key) or "").strip()
    if value:
        return value
    file_value = read_managed_keys_from_file().get(key, "").strip()
    if file_value:
        os.environ[key] = file_value
    return file_value


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "••••"
    return f"{'•' * 8}…{value[-4:]}"


def api_keys_status(
    *,
    hf_token_override: str | None = None,
    openrouter_api_key_override: str | None = None,
) -> dict[str, object]:
    hf = (
        str(hf_token_override).strip()
        if hf_token_override is not None
        else get_api_secret("HF_TOKEN")
    )
    openrouter = (
        str(openrouter_api_key_override).strip()
        if openrouter_api_key_override is not None
        else get_api_secret("OPENROUTER_API_KEY")
    )
    return {
        "hf_token_configured": bool(hf),
        "hf_token_preview": mask_secret(hf),
        "openrouter_api_key_configured": bool(openrouter),
        "openrouter_api_key_preview": mask_secret(openrouter),
    }


def update_api_keys(
    *,
    hf_token: str | None = None,
    openrouter_api_key: str | None = None,
    persist_to_env: bool | None = None,
) -> dict[str, object]:
    updates: dict[str, str] = {}
    if hf_token is not None:
        token = str(hf_token).strip()
        if token:
            updates["HF_TOKEN"] = token
    if openrouter_api_key is not None:
        key = str(openrouter_api_key).strip()
        if key:
            updates["OPENROUTER_API_KEY"] = key
    if updates:
        persist = should_persist_api_keys_to_env() if persist_to_env is None else bool(persist_to_env)
        if persist:
            _write_env_updates(updates)
            os.environ.update(updates)
    return api_keys_status()


def _write_env_updates(updates: dict[str, str]) -> None:
    path = env_path()
    lines: list[str] = []
    replaced: set[str] = set()

    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = parse_env_line(line)
            if parsed and parsed[0] in updates:
                key = parsed[0]
                lines.append(f"{key}={_quote(updates[key])}")
                replaced.add(key)
            else:
                lines.append(line)

    for key, value in updates.items():
        if key not in replaced:
            lines.append(f"{key}={_quote(value)}")

    text = "\n".join(lines)
    if text:
        text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
