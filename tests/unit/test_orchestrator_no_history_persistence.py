"""``persist_history=False`` on the orchestrator must leave no ChatStorage trace
(mirror of ``test_a2a_no_history_persistence.py`` for the sub-agent pipelines)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def orchestrator(monkeypatch, fake_llm, fake_urban, state_store):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.AsyncOllamaClient",
        lambda *a, **k: fake_llm,
    )
    from src.agents.services.orchestrator_service import OrchestratorService

    app_config = SimpleNamespace(
        DVD_MCP_URL="http://dvd", NORM_GRAPH_MCP_URL="http://norms"
    )
    svc = OrchestratorService(
        "http://ollama",
        Mock(),
        fake_urban,
        state_store,
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        app_config,
    )
    svc.create_chat = AsyncMock(return_value=("chat-xyz", "Тестовый чат"))
    svc.get_chat_messages = AsyncMock(return_value=SimpleNamespace(messages=[]))
    svc.add_single_message = AsyncMock()
    svc.add_complex_message = AsyncMock()
    return svc


async def _canned_pipeline(*args, **kwargs):
    yield {"type": "chunk", "content": {"text": "ответ", "done": False}}
    yield {"type": "chunk", "content": {"text": "", "done": True}}


@pytest.mark.asyncio
async def test_no_chat_storage_calls_without_persistence(orchestrator, fake_llm):
    fake_llm.json_responses = [
        json.dumps(
            {
                "mode": "execute",
                "steps": [{"agent": "provision", "task": "задача"}],
                "clarification_question": None,
            },
            ensure_ascii=False,
        )
    ]
    orchestrator.provision_service.run_provision_pipeline = (
        lambda *a, **k: _canned_pipeline()
    )

    events = [
        event
        async for event in orchestrator.run_orchestration_pipeline(
            idu_mcp_client=Mock(),
            effects_mcp_client=Mock(),
            dvd_mcp_client=Mock(),
            normgraph_mcp_client=Mock(),
            token="tok",
            model="m",
            temperature=0.5,
            user_query="запрос",
            scenario_id=772,
            persist_history=False,
        )
    ]
    await asyncio.sleep(0)  # let any (wrongly) scheduled persistence tasks run

    assert [e["type"] for e in events if e["type"] == "orchestrator_final"]
    orchestrator.create_chat.assert_not_awaited()
    orchestrator.add_single_message.assert_not_awaited()
    orchestrator.add_complex_message.assert_not_awaited()
    # no chat_created event either
    assert not [e for e in events if e["type"] == "service_event"]
