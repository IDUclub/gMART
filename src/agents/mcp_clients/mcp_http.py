"""FastMCP clients built with timeouts sized for this platform's payloads.

httpx defaults every phase — connect, read, write, pool — to 5 seconds. A tool
call on a large scenario ships tens of megabytes of GeoJSON to the MCP server,
and while the server is busy with the previous chunk it stops reading its socket;
the write then blocks for longer than 5 s and the call dies with
``httpx.WriteTimeout``, taking the whole pipeline row with it. The same holds for
the answer, which the server may take minutes to compute.

``MCP_HTTP_TIMEOUT`` (seconds) sets that budget for every MCP client at once.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastmcp import Client as McpClient
from fastmcp.client.transports import StreamableHttpTransport

DEFAULT_MCP_HTTP_TIMEOUT = 300.0
MAX_CONNECT_TIMEOUT = 30.0


def mcp_http_timeout() -> float:
    """Timeout in seconds for MCP HTTP traffic, from ``MCP_HTTP_TIMEOUT``."""

    raw = os.getenv("MCP_HTTP_TIMEOUT", "")
    try:
        return max(1.0, float(raw)) if raw else DEFAULT_MCP_HTTP_TIMEOUT
    except ValueError:
        return DEFAULT_MCP_HTTP_TIMEOUT


def mcp_httpx_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,  # noqa: ARG001 — the caller's default
    auth: httpx.Auth | None = None,
    **kwargs: Any,
) -> httpx.AsyncClient:
    """``McpHttpClientFactory``: the transport calls this to get its client.

    The ``timeout`` the caller passes is the MCP SDK's own default, which is what
    we are here to replace, so it is deliberately ignored. Everything else is
    forwarded: the two callers disagree about the signature — the FastMCP
    transport adds ``follow_redirects``, the SDK's own path does not — and a
    factory that rejects an argument fails the connection outright.
    """

    seconds = mcp_http_timeout()
    kwargs.setdefault("follow_redirects", True)
    return httpx.AsyncClient(
        headers=headers,
        auth=auth,
        timeout=httpx.Timeout(seconds, connect=min(MAX_CONNECT_TIMEOUT, seconds)),
        **kwargs,
    )


def build_mcp_client(url: str, auth: str | None = None) -> McpClient:
    """An MCP client for ``url``, with the timeouts above applied.

    Non-HTTP targets (in-memory servers, local scripts) keep FastMCP's own
    transport inference — there is no socket to time out there.
    """

    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return McpClient(url, auth=auth) if auth else McpClient(url)
    # The read budget rides on the httpx client too: FastMCP 3.4 dropped
    # sse_read_timeout and points at the factory for exactly this.
    return McpClient(
        StreamableHttpTransport(url, auth=auth, httpx_client_factory=mcp_httpx_client)
    )
