"""Per-client play sessions keyed by browser tab session id."""

from __future__ import annotations

import os
import re
import time
import uuid
from threading import Lock
from typing import TYPE_CHECKING

from fastapi import Header, Response

if TYPE_CHECKING:
    from .runtime import Session

SESSION_ID_HEADER = "X-Play-Session-Id"
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
DEFAULT_TTL_SECONDS = int(os.environ.get("PLAY_WEB_SESSION_TTL_SECONDS", "7200"))
MAX_SESSIONS = int(os.environ.get("PLAY_WEB_MAX_SESSIONS", "200"))


def normalize_session_id(value: str | None) -> str | None:
    if not value:
        return None
    candidate = str(value).strip()
    if SESSION_ID_PATTERN.fullmatch(candidate):
        return candidate
    return None


class PlaySessionRegistry:
    def __init__(self, *, ttl_seconds: int = DEFAULT_TTL_SECONDS, max_sessions: int = MAX_SESSIONS) -> None:
        self._ttl_seconds = max(60, int(ttl_seconds))
        self._max_sessions = max(1, int(max_sessions))
        self._sessions: dict[str, tuple[Session, float]] = {}
        self._lock = Lock()

    def _create_session(self) -> Session:
        from .features import apply_demo_runtime_defaults
        from .runtime import create_isolated_session

        sess = create_isolated_session()
        apply_demo_runtime_defaults(sess, apply_model_default=True)
        return sess

    def _purge_expired_unlocked(self, now: float) -> None:
        expired = [
            session_id
            for session_id, (_, last_access) in self._sessions.items()
            if now - last_access > self._ttl_seconds
        ]
        for session_id in expired:
            del self._sessions[session_id]

    def _evict_oldest_unlocked(self) -> None:
        if not self._sessions:
            return
        oldest_id = min(self._sessions, key=lambda key: self._sessions[key][1])
        del self._sessions[oldest_id]

    def get_or_create(self, session_id: str | None) -> tuple[str, Session]:
        now = time.monotonic()
        normalized = normalize_session_id(session_id)
        with self._lock:
            self._purge_expired_unlocked(now)
            if normalized and normalized in self._sessions:
                sess, _ = self._sessions[normalized]
                sess.play_session_id = normalized
                self._sessions[normalized] = (sess, now)
                return normalized, sess

            while len(self._sessions) >= self._max_sessions:
                self._evict_oldest_unlocked()

            resolved_id = normalized or str(uuid.uuid4())
            sess = self._create_session()
            sess.play_session_id = resolved_id
            self._sessions[resolved_id] = (sess, now)
            return resolved_id, sess


play_session_registry = PlaySessionRegistry()


def resolve_play_session(session_id: str | None) -> tuple[str, Session]:
    return play_session_registry.get_or_create(session_id)


def attach_session_id_header(response: Response, session_id: str) -> None:
    response.headers[SESSION_ID_HEADER] = session_id


def session_id_from_header(
    x_play_session_id: str | None = Header(default=None, alias=SESSION_ID_HEADER),
) -> str | None:
    return normalize_session_id(x_play_session_id)
