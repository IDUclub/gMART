"""MCP clients must not inherit httpx's 5-second defaults.

A tool call on a large scenario ships tens of megabytes to the MCP server; while
the server is busy it stops reading its socket and the write blocks past the
default write timeout, so the call dies with httpx.WriteTimeout and takes the
pipeline row with it.
"""

from __future__ import annotations

import httpx

from src.agents.mcp_clients.mcp_http import (
    DEFAULT_MCP_HTTP_TIMEOUT,
    build_mcp_client,
    mcp_http_timeout,
    mcp_httpx_client,
)


def test_default_timeout_is_minutes_not_seconds(monkeypatch):
    monkeypatch.delenv("MCP_HTTP_TIMEOUT", raising=False)

    assert mcp_http_timeout() == DEFAULT_MCP_HTTP_TIMEOUT
    assert DEFAULT_MCP_HTTP_TIMEOUT >= 60


def test_timeout_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("MCP_HTTP_TIMEOUT", "45")
    assert mcp_http_timeout() == 45.0


def test_unparsable_timeout_falls_back(monkeypatch):
    monkeypatch.setenv("MCP_HTTP_TIMEOUT", "нет")
    assert mcp_http_timeout() == DEFAULT_MCP_HTTP_TIMEOUT


def test_every_phase_gets_the_budget(monkeypatch):
    """Read alone is not enough — the failure was on write."""

    monkeypatch.setenv("MCP_HTTP_TIMEOUT", "120")

    client = mcp_httpx_client(headers={"X-Test": "1"})

    assert client.timeout.write == 120.0
    assert client.timeout.read == 120.0
    assert client.timeout.pool == 120.0
    # connect is capped: a server that is down should fail fast, not in minutes
    assert client.timeout.connect == 30.0
    assert client.headers["X-Test"] == "1"


def test_factory_accepts_both_caller_signatures():
    """The two callers disagree, and rejecting an argument kills the connection
    with a bare ``Client failed to connect``, not a signature error."""

    # fastmcp/client/transports/http.py — adds follow_redirects
    client = mcp_httpx_client(
        headers={}, auth=None, follow_redirects=True, timeout=httpx.Timeout(30.0)
    )
    assert client.follow_redirects is True

    # mcp/client/streamable_http.py — the SDK's own, positional-compatible shape
    client = mcp_httpx_client({}, httpx.Timeout(30.0), None)
    assert client.follow_redirects is True


def test_the_sdk_default_timeout_is_ignored(monkeypatch):
    """The transport passes the MCP SDK's own default in — that is what we replace."""

    monkeypatch.setenv("MCP_HTTP_TIMEOUT", "120")

    client = mcp_httpx_client(timeout=httpx.Timeout(5.0))

    assert client.timeout.write == 120.0


def test_http_client_is_built_with_the_factory():
    client = build_mcp_client("http://idumcp:8000/mcp", auth="token")

    assert client.transport.httpx_client_factory is mcp_httpx_client
    assert client.transport.auth.token.get_secret_value() == "token"


def test_non_http_target_keeps_fastmcp_transport_inference():
    """There is no socket to time out on an in-memory server."""

    from fastmcp import FastMCP

    client = build_mcp_client(FastMCP(name="test"))

    assert not hasattr(client.transport, "httpx_client_factory")
