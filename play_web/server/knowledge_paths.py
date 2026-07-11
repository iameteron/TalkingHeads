"""Resolve world- and session-aware knowledge file paths for play_web."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from oracle.knowledge import (
    load_durable_knowledge_records,
    load_knowledge,
    read_starter_revision,
    use_knowledge_paths,
)

from .companion_bench import (
    _base_knowledge_paths,
    _bench_dir,
    _seed_knowledge_from_main_source,
    _slugify,
    _world_slug,
)


def session_knowledge_slug(sess: Any) -> str:
    """Short filesystem-safe id derived from the browser play session."""
    session_id = str(getattr(sess, "play_session_id", "") or "").strip()
    if not session_id:
        return "anonymous"
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]


def session_knowledge_dir(sess: Any) -> Path:
    path = _bench_dir() / "sessions" / session_knowledge_slug(sess)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_base_knowledge_paths(sess: Any) -> tuple[Path, Path]:
    world_slug = _world_slug(getattr(sess, "texture_theme", None))
    store = session_knowledge_dir(sess)
    return (
        store / f"knowledge_data__{world_slug}.json",
        store / f"knowledge_data__{world_slug}.txt",
    )


def _session_model_knowledge_paths(sess: Any) -> tuple[Path, Path]:
    model_slug = _slugify(getattr(sess, "active_agent_model", "") or "unknown_model")
    world_slug = _world_slug(getattr(sess, "texture_theme", None))
    store = session_knowledge_dir(sess)
    return (
        store / f"knowlage_data_{model_slug}__{world_slug}.json",
        store / f"knowlage_data_{model_slug}__{world_slug}.txt",
    )


def play_knowledge_paths_for_session(sess: Any) -> tuple[Path, Path]:
    """Knowledge files the active play session should read and write."""
    if getattr(sess, "is_arc_game", lambda: False)():
        return _base_knowledge_paths("craftax")
    if _uses_model_knowledge_store(sess):
        return _session_model_knowledge_paths(sess)
    return _session_base_knowledge_paths(sess)


def _uses_model_knowledge_store(sess: Any) -> bool:
    return bool(
        getattr(sess, "companion_research_active", False)
        or getattr(getattr(sess, "campaign_state", None), "enabled", False)
    )


def _ensure_session_knowledge_current(
    json_path: Path,
    txt_path: Path,
    *,
    world_mode: str | None,
) -> None:
    """Refresh starter rows in a session file when the world base revision is newer."""
    base_json, _base_txt = _base_knowledge_paths(world_mode)
    base_revision = read_starter_revision(base_json)
    if base_revision <= 0:
        if not (json_path.exists() or txt_path.exists()):
            _seed_knowledge_from_main_source(json_path, txt_path, world_mode=world_mode)
        return
    model_revision = read_starter_revision(json_path)
    if json_path.exists() and model_revision >= base_revision:
        return
    existing_model_entries: list[dict[str, Any]] = []
    if json_path.exists() or txt_path.exists():
        with use_knowledge_paths(json_path=json_path, txt_path=txt_path):
            existing_model_entries = load_durable_knowledge_records()
    _seed_knowledge_from_main_source(
        json_path,
        txt_path,
        world_mode=world_mode,
        existing_model_entries=existing_model_entries,
        model_revision=model_revision,
    )


def ensure_play_session_knowledge(sess: Any, json_path: Path, txt_path: Path) -> None:
    """Create or refresh session-scoped knowledge files before agent reads/writes."""
    world_mode = getattr(sess, "texture_theme", None)
    if not (json_path.exists() or txt_path.exists()):
        _seed_knowledge_from_main_source(json_path, txt_path, world_mode=world_mode)
        return
    if _uses_model_knowledge_store(sess):
        _ensure_session_knowledge_current(json_path, txt_path, world_mode=world_mode)


@contextmanager
def play_knowledge_context(sess: Any) -> Iterator[tuple[Path, Path]]:
    """Activate session knowledge paths and refresh stale companion starter rows."""
    json_path, txt_path = play_knowledge_paths_for_session(sess)
    ensure_play_session_knowledge(sess, json_path, txt_path)
    with use_knowledge_paths(json_path=json_path, txt_path=txt_path):
        yield json_path, txt_path


def load_session_knowledge(sess: Any) -> str:
    with play_knowledge_context(sess):
        return load_knowledge()
