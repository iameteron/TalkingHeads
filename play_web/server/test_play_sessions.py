from play_web.server.play_sessions import (
    PlaySessionRegistry,
    normalize_session_id,
    resolve_play_session,
)


def test_normalize_session_id_accepts_uuid() -> None:
    session_id = "550e8400-e29b-41d4-a716-446655440000"
    assert normalize_session_id(session_id) == session_id


def test_normalize_session_id_rejects_invalid() -> None:
    assert normalize_session_id("bad id!") is None
    assert normalize_session_id("") is None


def test_registry_creates_isolated_sessions() -> None:
    registry = PlaySessionRegistry(ttl_seconds=3600, max_sessions=10)
    first_id, first_sess = registry.get_or_create("client-session-a")
    second_id, second_sess = registry.get_or_create("client-session-b")
    third_id, third_sess = registry.get_or_create("client-session-a")

    assert first_id == "client-session-a"
    assert second_id == "client-session-b"
    assert third_id == "client-session-a"
    assert first_sess is third_sess
    assert first_sess is not second_sess
    assert first_sess.play_session_id == "client-session-a"
    assert second_sess.play_session_id == "client-session-b"


def test_resolve_play_session_creates_client_id_when_missing() -> None:
    session_id, _sess = resolve_play_session(None)
    assert normalize_session_id(session_id) == session_id
