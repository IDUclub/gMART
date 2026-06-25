"""Integration fixtures: probe the live local stack and skip cleanly when a service is down.

Each ``require_*`` fixture skips the test when its service (Redis / Ollama / IDU_DVD MCP) is
unavailable, so the unit suite and partial stacks never produce failures. Mirrors the tiered
approach used in IDU_DVD.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture
async def require_redis():
    """A live async Redis client (decode_responses), or skip if Redis is down."""
    import redis.asyncio as aioredis

    client = aioredis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    try:
        await client.ping()
    except Exception as exc:  # noqa: BLE001
        await client.aclose()
        pytest.skip(f"Redis unavailable at {os.environ['REDIS_URL']}: {exc}")
    yield client
    await client.aclose()


@pytest.fixture
def require_ollama() -> str:
    """The Ollama base URL, or skip if Ollama is not reachable."""
    import httpx

    url = os.environ["OLLAMA_API_URL"].rstrip("/")
    try:
        httpx.get(f"{url}/api/tags", timeout=2.0).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Ollama unavailable at {url}: {exc}")
    return url


@pytest.fixture
async def require_dvd_mcp():
    """A live DvdMcpClient (tools listable), or skip if the IDU_DVD MCP is down."""
    from fastmcp import Client

    from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient

    url = os.environ["DVD_MCP_SERVER"]
    client = DvdMcpClient(Client(url), mcp_url=url)
    try:
        await client.load_ollama_tools()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"IDU_DVD MCP unavailable at {url}: {exc}")
    return client
