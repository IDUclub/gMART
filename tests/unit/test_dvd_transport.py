"""Stage 6: transport reuse — HTTP pool, one MCP session per run, cached metadata."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.common.api_handlers import json_api_handler
from src.agents.common.api_handlers.json_api_handler import JsonApiHandler
from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter
from tests.helpers import FakeDvdMcpClient, plan_json, verdict_json
from tests.unit.test_json_api_handler import FakeResponse, FakeSession


class PooledSession(FakeSession):
    created = 0

    def __init__(self):
        super().__init__([FakeResponse(200, json_body={"ok": n}) for n in range(5)])
        self.closed = False
        PooledSession.created += 1

    async def close(self):
        self.closed = True


async def test_handler_reuses_one_session_and_closes_it(monkeypatch):
    PooledSession.created = 0
    monkeypatch.setattr(json_api_handler.aiohttp, "ClientSession", PooledSession)
    handler = JsonApiHandler("http://chat-storage", backoff_base=0)
    assert await handler.get("/a") == {"ok": 0}
    assert await handler.post("/b", data={}) == {"ok": 1}
    assert await handler.patch("/c", data={}) == {"ok": 2}
    assert PooledSession.created == 1
    session = handler._pooled
    await handler.close()
    assert session.closed
    # A closed pool is replaced on the next call.
    assert await handler.get("/d") == {"ok": 0}
    assert PooledSession.created == 2
    await handler.close()


async def test_model_window_is_cached():
    adapter = OpenAiCompatAdapter("http://vllm:8000/v1")
    adapter.client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(id="gpt-oss-20b", max_model_len=32768)]
        )
    )
    try:
        assert await adapter.model_context_window("gpt-oss-20b") == 32768
        assert await adapter.model_context_window("gpt-oss-20b") == 32768
        assert adapter.client.models.list.await_count == 1
    finally:
        await adapter.client.close()


async def test_identical_prompts_are_tokenized_once():
    adapter = OpenAiCompatAdapter("http://vllm:8000/v1")
    adapter.client.post = AsyncMock(return_value={"count": 77})
    first = [{"role": "user", "content": "свидетельства"}]
    try:
        assert await adapter.model_input_tokens("m", first) == 77
        assert await adapter.model_input_tokens("m", [dict(first[0])]) == 77
        assert adapter.client.post.await_count == 1
        await adapter.model_input_tokens("m", first, reasoning_effort="low")
        await adapter.model_input_tokens("m", [{"role": "user", "content": "другое"}])
        assert adapter.client.post.await_count == 3
    finally:
        await adapter.client.close()


async def test_failed_count_is_not_cached():
    adapter = OpenAiCompatAdapter("http://vllm:8000/v1")
    adapter.client.post = AsyncMock(return_value={"count": None})
    messages = [{"role": "user", "content": "вопрос"}]
    try:
        assert await adapter.model_input_tokens("m", messages) is None
        assert await adapter.model_input_tokens("m", messages) is None
        assert adapter.client.post.await_count == 2
    finally:
        await adapter.client.close()


class SessionRecorder:
    def __init__(self):
        self.entered = self.exited = 0

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, *exc):
        self.exited += 1


class SessionMcp(FakeDvdMcpClient):
    def __init__(self):
        super().__init__()
        self.mcp_client = SessionRecorder()
        self.open_during_search = []

    async def search(self, *args, **kwargs):
        recorder = self.mcp_client
        self.open_during_search.append(recorder.entered > recorder.exited)
        return await super().search(*args, **kwargs)


async def _run(service, mcp):
    return [
        event
        async for event in service.run_document_qa_pipeline(
            dvd_mcp_client=mcp,
            token="tok",
            model="m",
            temperature=0.0,
            user_query="нормы озеленения",
            chat_id="chat-1",
        )
    ]


@pytest.mark.parametrize("enabled", [True, False])
async def test_run_keeps_one_mcp_session(service, fake_llm, monkeypatch, enabled):
    monkeypatch.setenv("DVD_PERSISTENT_MCP_SESSION", "true" if enabled else "false")
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Ответ [1]."]
    mcp = SessionMcp()
    await _run(service, mcp)
    assert mcp.open_during_search == [enabled]
    assert (mcp.mcp_client.entered, mcp.mcp_client.exited) == (
        (1, 1) if enabled else (0, 0)
    )


async def test_failed_session_open_falls_back_to_per_call(service, fake_llm):
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Ответ [1]."]
    mcp = SessionMcp()
    mcp.mcp_client.__aenter__ = AsyncMock(side_effect=ConnectionError("down"))
    events = await _run(service, mcp)
    assert events[-1]["content"]["done"]
    assert mcp.mcp_client.exited == 0
