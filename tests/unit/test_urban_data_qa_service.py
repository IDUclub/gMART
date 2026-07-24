"""Unit tests for UrbanDataQaService — the native Ollama tool-calling QA loop.

The shared ``fake_llm``/``FakeLlmClient`` (tests/helpers.py) only models the dict-shaped,
non-tool-calling planner/critic pattern used elsewhere, so this file uses a small local
fake that also carries ``message.tool_calls`` (attribute access, mirroring the real
``ollama.ChatResponse``/``Message`` objects).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.agents.services.urban_data_qa_service import UrbanDataQaService
from tests.helpers import answer_text, events_of_type, final_chunk, types_of


class _ToolCall:
    def __init__(self, name: str, arguments: dict):
        self.function = SimpleNamespace(name=name, arguments=arguments)


class _ChatResponse:
    def __init__(self, content: str = "", tool_calls=None, done: bool = True):
        self.message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
        self.done = done


class FakeToolCallingLlmClient:
    """Stand-in for the ollama AsyncClient exposed as ``self.llm_client``.

    - non-stream ``chat`` (the tool-calling rounds): returns the next ``_ChatResponse``
      popped from ``responses``.
    - stream ``chat`` (final answer drafting): returns an async iterator built from the
      next string in ``answer_texts``.
    """

    def __init__(self) -> None:
        self.responses: list[_ChatResponse] = []
        self.answer_texts: list[str] = []
        self.chat_calls: list[SimpleNamespace] = []

    async def chat(
        self,
        model=None,
        messages=None,
        tools=None,
        options=None,
        stream=False,
        **kwargs,
    ):
        self.chat_calls.append(
            SimpleNamespace(
                model=model,
                messages=messages,
                tools=tools,
                options=options,
                stream=stream,
            )
        )
        if stream:
            text = self.answer_texts.pop(0) if self.answer_texts else ""
            return self._stream(text)
        return self.responses.pop(0)

    @staticmethod
    async def _stream(text: str):
        if text:
            mid = len(text) // 2
            yield _ChatResponse(content=text[:mid], done=False)
            yield _ChatResponse(content=text[mid:], done=True)
        else:
            yield _ChatResponse(content="", done=True)


class FakeUrbanDataMcpClient:
    """Stand-in for ``UrbanDataMcpClient``: records tool calls, returns programmed results."""

    def __init__(self, tools=None, tool_results=None):
        self.tools = (
            tools
            if tools is not None
            else [
                {
                    "type": "function",
                    "function": {"name": "GetTerritories", "parameters": {}},
                }
            ]
        )
        self._tool_results = tool_results or {}
        self.execute_calls: list[tuple[str, dict]] = []
        self.update_token_calls: list[str] = []

    async def get_tools(self):
        return self.tools

    async def execute_tool(self, tool_name, arguments, meta=None):
        self.execute_calls.append((tool_name, dict(arguments)))
        return self._tool_results.get(tool_name)

    def update_token(self, new_token):
        self.update_token_calls.append(new_token)


@pytest.fixture
def fake_urban_llm() -> FakeToolCallingLlmClient:
    return FakeToolCallingLlmClient()


@pytest.fixture
def fake_urban_mcp() -> FakeUrbanDataMcpClient:
    return FakeUrbanDataMcpClient()


@pytest.fixture
def service(monkeypatch, fake_urban_llm, fake_urban, state_store):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.AsyncOllamaClient",
        lambda *a, **k: fake_urban_llm,
    )
    svc = UrbanDataQaService("http://ollama", Mock(), fake_urban, state_store)
    svc.create_chat = AsyncMock(return_value=("chat-xyz", "Тестовый чат"))
    svc.get_chat_messages = AsyncMock(return_value=SimpleNamespace(messages=[]))
    svc.add_single_message = AsyncMock()
    svc._schedule_persist_answer = Mock()
    return svc


async def _run(service, mcp, **overrides):
    kwargs = dict(
        urban_data_mcp_client=mcp,
        token="tok",
        model="m",
        temperature=0.0,
        user_query="Какие территории входят в проект?",
        chat_id="chat-1",
    )
    kwargs.update(overrides)
    return [event async for event in service.run_urban_data_qa_pipeline(**kwargs)]


FEATURE_COLLECTION = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "Территория 1"},
            "geometry": {"type": "Point", "coordinates": [30.3, 59.9]},
        }
    ],
}


class TestToolCallingLoop:
    async def test_single_tool_call_then_final_answer(
        self, service, fake_urban_llm, fake_urban_mcp
    ):
        fake_urban_mcp._tool_results["GetTerritories"] = FEATURE_COLLECTION
        fake_urban_llm.responses = [
            _ChatResponse(
                tool_calls=[_ToolCall("GetTerritories", {"scenario_id": 772})]
            ),
            _ChatResponse(content="Готово"),
        ]
        fake_urban_llm.answer_texts = ["В проект входит одна территория."]

        events = await _run(service, fake_urban_mcp)

        assert types_of(events)[0] == "pipeline_started"
        assert fake_urban_mcp.execute_calls == [
            ("GetTerritories", {"scenario_id": 772})
        ]

        tool_call_events = events_of_type(events, "tool_call")
        assert len(tool_call_events) == 1
        assert tool_call_events[0]["content"]["mcp_source"] == "URBAN_DATA_MCP_URL"
        assert (
            tool_call_events[0]["content"]["tool_calls"][0]["function"]["name"]
            == "GetTerritories"
        )

        fc_events = events_of_type(events, "feature_collection")
        assert len(fc_events) == 1
        assert fc_events[0]["content"]["name"] == "GetTerritories"
        assert fc_events[0]["content"]["feature_collection"] == FEATURE_COLLECTION

        assert answer_text(events) == "В проект входит одна территория."
        fc = final_chunk(events)
        assert fc is not None and fc["done"] is True

        collected = service._schedule_persist_answer.call_args.args[2]
        assert collected["final_answer"] == "В проект входит одна территория."
        assert collected["newly_completed"] is True
        assert collected["tool_calls"] == [
            {"function": {"name": "GetTerritories", "arguments": {"scenario_id": 772}}}
        ]

    async def test_no_tool_calls_needed_answers_directly(
        self, service, fake_urban_llm, fake_urban_mcp
    ):
        fake_urban_llm.responses = [_ChatResponse(content="")]
        fake_urban_llm.answer_texts = ["Общий ответ без инструментов."]

        events = await _run(service, fake_urban_mcp)

        assert fake_urban_mcp.execute_calls == []
        assert events_of_type(events, "tool_call") == []
        assert answer_text(events) == "Общий ответ без инструментов."

    async def test_loop_stops_at_max_iterations(
        self, service, fake_urban_llm, fake_urban_mcp
    ):
        fake_urban_mcp._tool_results["GetTerritories"] = {"count": 0}
        fake_urban_llm.responses = [
            _ChatResponse(tool_calls=[_ToolCall("GetTerritories", {"page": i})])
            for i in range(UrbanDataQaService.MAX_TOOL_ITERATIONS)
        ]
        fake_urban_llm.answer_texts = ["Ответ после исчерпания попыток."]

        events = await _run(service, fake_urban_mcp)

        assert (
            len(fake_urban_mcp.execute_calls) == UrbanDataQaService.MAX_TOOL_ITERATIONS
        )
        assert answer_text(events) == "Ответ после исчерпания попыток."


class TestSystemPrompt:
    async def test_scenario_id_is_surfaced_to_the_model(
        self, service, fake_urban_llm, fake_urban_mcp
    ):
        fake_urban_llm.responses = [_ChatResponse(content="")]
        fake_urban_llm.answer_texts = ["Ответ."]

        await _run(service, fake_urban_mcp, scenario_id=772)

        system_message = fake_urban_llm.chat_calls[0].messages[0]
        assert system_message["role"] == "system"
        assert "772" in system_message["content"]

    async def test_missing_scenario_id_asks_to_clarify_if_needed(
        self, service, fake_urban_llm, fake_urban_mcp
    ):
        fake_urban_llm.responses = [_ChatResponse(content="")]
        fake_urban_llm.answer_texts = ["Ответ."]

        await _run(service, fake_urban_mcp, scenario_id=None)

        system_message = fake_urban_llm.chat_calls[0].messages[0]
        assert "не выбран" in system_message["content"]


class TestReconnect:
    async def test_reconnect_after_completion_replays_only(
        self, service, fake_urban_llm, fake_urban_mcp
    ):
        fake_urban_llm.responses = [_ChatResponse(content="")]
        fake_urban_llm.answer_texts = ["Финальный ответ."]

        first = await _run(service, fake_urban_mcp)
        request_id = first[0]["content"]["request_id"]

        # ``_buf`` buffers events via fire-and-forget ``asyncio.create_task`` — give the
        # event loop a few ticks so every buffered-event write actually lands before the
        # reconnect run reads them back.
        for _ in range(50):
            await asyncio.sleep(0)

        service._schedule_persist_answer.reset_mock()
        second = await _run(
            service, fake_urban_mcp, request_id=request_id, chat_id=None
        )

        assert types_of(second) == types_of(first)
        assert answer_text(second) == "Финальный ответ."
        # Nothing was re-executed: no new LLM/tool calls were made on replay.
        assert len(fake_urban_llm.chat_calls) == 2


class TestGeometryShaping:
    def test_feature_collections_found_at_any_depth(self):
        data = {"outer": {"inner": [1, {"layer": FEATURE_COLLECTION}]}}
        events = list(UrbanDataQaService._feature_collections("Tool", data))
        assert len(events) == 1
        assert events[0]["content"]["name"] == "Tool.outer.inner[1].layer"
        assert events[0]["content"]["feature_collection"] == FEATURE_COLLECTION

    def test_strip_geometries_replaces_coordinates_and_summarizes_features(self):
        stripped = UrbanDataQaService._strip_geometries(FEATURE_COLLECTION)
        assert stripped["feature_count"] == 1
        assert stripped["sample_properties"] == [{"name": "Территория 1"}]
        assert "coordinates" not in stripped

    def test_strip_geometries_leaves_plain_data_untouched(self):
        data = {"name": "test", "count": 3, "items": [1, 2, 3]}
        assert UrbanDataQaService._strip_geometries(data) == data
