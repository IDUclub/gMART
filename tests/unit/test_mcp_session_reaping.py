"""A closed MCP session must be forgotten, or the server grows one per call.

Each session starts a task that sits inside app.run(); the SDK ends it only on
an idle timeout that ships disabled. A client that opens a session per tool call
— which the agents do — then leaves a live transport behind every time: 802 of
them within half an hour of benchmark load, while the process grew until the host
started killing workers in other containers.
"""

from __future__ import annotations

import pytest

from src.idu_mcp.main import _BodyLimitSessionManager, _session_idle_timeout


class FakeIdleScope:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class FakeTransport:
    def __init__(self) -> None:
        self.idle_scope = FakeIdleScope()


def _manager(**instances) -> _BodyLimitSessionManager:
    """A manager without the SDK base's constructor, which needs a live app."""

    manager = _BodyLimitSessionManager.__new__(_BodyLimitSessionManager)
    manager._server_instances = dict(instances)
    manager._session_owners = {name: "owner" for name in instances}
    return manager


@pytest.mark.parametrize(
    "value,expected",
    [(None, 600.0), ("900", 900.0), ("0", None), ("off", None), ("нет", 600.0)],
)
def test_idle_timeout_setting(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("MCP_SESSION_IDLE_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("MCP_SESSION_IDLE_TIMEOUT", value)

    assert _session_idle_timeout() == expected


def test_the_fallback_timeout_outlives_the_longest_tool_call(monkeypatch):
    """The deadline only moves on incoming requests, and a call in progress
    sends none — a timeout below the call timeout would cancel real work."""

    monkeypatch.delenv("MCP_SESSION_IDLE_TIMEOUT", raising=False)
    from src.agents.mcp_clients.mcp_http import DEFAULT_MCP_HTTP_TIMEOUT

    assert _session_idle_timeout() > DEFAULT_MCP_HTTP_TIMEOUT


def test_session_id_is_read_from_the_headers():
    scope = {
        "method": "DELETE",
        "headers": [
            (b"content-type", b"application/json"),
            (b"MCP-Session-Id", b"abc"),
        ],
    }

    assert _BodyLimitSessionManager._session_id(scope) == "abc"
    assert _BodyLimitSessionManager._session_id({"headers": []}) is None
    assert _BodyLimitSessionManager._session_id({}) is None


def test_dropping_a_session_ends_the_task_behind_it():
    transport = FakeTransport()
    manager = _manager(abc=transport)

    manager._drop_session("abc")

    assert manager._server_instances == {}
    assert manager._session_owners == {}
    assert transport.idle_scope.cancelled, "the session task must be cancelled too"


def test_dropping_an_unknown_session_is_harmless():
    manager = _manager(abc=FakeTransport())

    manager._drop_session("other")
    manager._drop_session(None)

    assert set(manager._server_instances) == {"abc"}


def test_a_transport_without_an_idle_scope_is_still_forgotten():
    """Nothing to cancel when the timeout is switched off, but the instance
    must go all the same."""

    class Bare:
        idle_scope = None

    manager = _manager(abc=Bare())

    manager._drop_session("abc")

    assert manager._server_instances == {}


@pytest.mark.asyncio
async def test_delete_drops_the_session_and_other_methods_do_not(monkeypatch):
    transport = FakeTransport()
    manager = _manager(abc=transport)
    handled: list[str] = []

    async def fake_super(scope, receive, send):
        handled.append(scope["method"])

    monkeypatch.setattr(
        type(manager).__mro__[1], "handle_request", staticmethod(fake_super)
    )

    headers = [(b"mcp-session-id", b"abc")]
    await manager.handle_request({"method": "POST", "headers": headers}, None, None)
    assert set(manager._server_instances) == {"abc"}, "a call must keep its session"

    await manager.handle_request({"method": "DELETE", "headers": headers}, None, None)

    assert manager._server_instances == {}
    assert transport.idle_scope.cancelled
    assert handled == ["POST", "DELETE"], "the SDK still handles the request itself"
