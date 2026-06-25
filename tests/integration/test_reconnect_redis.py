"""Integration: the reconnect cycle (buffer → checkpoint → replay) against a *real* Redis.

Exercises the genuine fire-and-forget buffering and ``PipelineStateStore`` persistence end to
end: a first run completes (writing state to Redis), then a reconnect with the same request_id
replays the buffered events and skips all work. LLM and IDU_DVD MCP are faked — the live
dependency under test is Redis.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.helpers import (
    FakeDvdMcpClient,
    FakeLlmClient,
    FakeUrbanApiClient,
    final_chunk,
    plan_json,
    verdict_json,
)

pytestmark = pytest.mark.integration

_RID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def _build_service(monkeypatch, fake_llm, state_store):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.AsyncOllamaClient",
        lambda *a, **k: fake_llm,
    )
    from src.agents.services.dvd_rag_service import DvdRagService

    svc = DvdRagService("http://ollama", Mock(), FakeUrbanApiClient(), state_store)
    svc.create_chat = AsyncMock(return_value=("chat-1", "Тест"))
    svc.get_chat_messages = AsyncMock(return_value=SimpleNamespace(messages=[]))
    svc._schedule_persist_answer = Mock()
    return svc


async def _clear(redis):
    for suffix in ("state", "checkpoint", "events"):
        await redis.delete(f"pipeline:{_RID}:{suffix}")


async def test_reconnect_replays_against_real_redis(require_redis, monkeypatch):
    from src.agents.services.pipeline_state import PipelineStateStore

    store = PipelineStateStore(require_redis)
    await _clear(require_redis)

    # First run to completion — buffers events + checkpoints "accepted" in real Redis.
    fake_llm = FakeLlmClient()
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Ответ против реального Redis [1]"]
    svc = _build_service(monkeypatch, fake_llm, store)
    first = [
        event
        async for event in svc.run_document_qa_pipeline(
            dvd_mcp_client=FakeDvdMcpClient(),
            token="t",
            model="m",
            temperature=0.0,
            user_query="q",
            chat_id="chat-1",
            request_id=_RID,
        )
    ]
    assert final_chunk(first) is not None
    # let the fire-and-forget buffer tasks flush to Redis
    await asyncio.sleep(0.3)

    # Reconnect with the same request_id and a *fresh* fake with no programmed responses:
    # if any work were redone it would have nothing to consume.
    fake_llm2 = FakeLlmClient()
    mcp2 = FakeDvdMcpClient()
    svc2 = _build_service(monkeypatch, fake_llm2, store)
    second = [
        event
        async for event in svc2.run_document_qa_pipeline(
            dvd_mcp_client=mcp2,
            token="t",
            model="m",
            temperature=0.0,
            user_query="q",
            chat_id="chat-1",
            request_id=_RID,
        )
    ]

    assert mcp2.search_calls == []  # no new searches
    assert fake_llm2.chat_calls == []  # no new LLM calls
    assert final_chunk(second) is not None  # the done chunk came back from the replay
    svc2._schedule_persist_answer.assert_not_called()  # no duplicate persistence

    await _clear(require_redis)
