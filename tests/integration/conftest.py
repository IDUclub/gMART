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
def require_openai_backend() -> tuple[str, str]:
    """``(base_url, model)`` of a live OpenAI-compatible server, or skip.

    Points at vLLM through ``VLLM_BASE_URL``; any other OpenAI-compatible server
    works, including Ollama's own ``/v1``. The model is the first one served
    unless ``VLLM_MODEL`` names it.
    """

    import httpx

    url = os.environ.get("VLLM_BASE_URL", "").rstrip("/")
    if not url:
        pytest.skip("VLLM_BASE_URL is not set")
    try:
        served = httpx.get(f"{url}/models", timeout=5.0).raise_for_status().json()
        models = [m["id"] for m in served["data"]]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"OpenAI-compatible server unavailable at {url}: {exc}")
    model = os.environ.get("VLLM_MODEL") or (models[0] if models else "")
    if not model:
        pytest.skip(f"no models served at {url}")
    return url, model


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
