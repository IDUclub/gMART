"""Shared fixtures for the gMART agents test-suite.

Service URLs are set in the environment *before* any ``src.agents`` module is imported, so
``app_config_loader.load_config`` (executed at import time by ``dependencies.py``) resolves
without a live stack. Real env vars win (``setdefault``) so CI / a local ``.env`` can override.
"""

from __future__ import annotations

import os

os.environ.setdefault("OLLAMA_API_URL", "http://localhost:11434")
os.environ.setdefault("IDU_MCP_SERVER", "http://localhost:8000/mcp")
os.environ.setdefault("OBJECTS_EFFECTS_MCP_SERVER", "http://localhost:8080/mcp")
os.environ.setdefault("DVD_MCP_SERVER", "http://localhost:8000/mcp")
os.environ.setdefault("CHAT_STORAGE", "http://localhost:8010")
os.environ.setdefault("URBAN_API_URL", "http://localhost/api")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.helpers import FakeDvdMcpClient, FakeLlmClient, FakeUrbanApiClient


@pytest.fixture
def fake_llm() -> FakeLlmClient:
    return FakeLlmClient()


@pytest.fixture
def fake_urban() -> FakeUrbanApiClient:
    return FakeUrbanApiClient()


@pytest.fixture
def fake_mcp() -> FakeDvdMcpClient:
    return FakeDvdMcpClient()


@pytest.fixture
def state_store():
    """A real PipelineStateStore backed by fakeredis (async) — exercises the genuine
    buffering / checkpoint / replay code paths without a live Redis."""
    import fakeredis.aioredis

    from src.agents.services.pipeline_state import PipelineStateStore

    return PipelineStateStore(fakeredis.aioredis.FakeRedis(decode_responses=True))


@pytest.fixture
def service(monkeypatch, fake_llm, fake_urban, state_store):
    """A ``DvdRagService`` wired with the fakes above.

    ``create_chat`` / ``get_chat_messages`` / ``_schedule_persist_answer`` are stubbed by
    default so loop tests stay focused; tests that exercise those paths override the stubs.
    """
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.AsyncOllamaClient",
        lambda *a, **k: fake_llm,
    )
    from src.agents.services.dvd_rag_service import DvdRagService

    svc = DvdRagService("http://ollama", Mock(), fake_urban, state_store)
    svc.create_chat = AsyncMock(return_value=("chat-xyz", "Тестовый чат"))
    svc.get_chat_messages = AsyncMock(return_value=SimpleNamespace(messages=[]))
    svc.add_single_message = AsyncMock()
    svc._schedule_persist_answer = Mock()
    return svc
