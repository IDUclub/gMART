"""The autonomous orchestrator (Google ADK): sub-agents as tools, same SSE contract."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.agents.model_clients.llm_base import LlmResponseError
from src.agents.services.orchestrator.orchestrator_adk import OrderedStepBus
from tests.helpers import events_of_type, types_of


def call(name: str, call_id: str, **args) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def turn(*calls: dict, text: str = "") -> dict:
    return {
        "message": {"content": text, "tool_calls": list(calls) or None},
        "done_reason": "tool_calls" if calls else "stop",
    }


class ToolScriptedLlm:
    """Adapter double answering the orchestrator's turns from a script.

    A scripted ``Exception`` is raised instead of answered. Non-tool requests
    (the JSON planner, chat titles) are served from ``plans``.
    """

    def __init__(self, *turns, plans: list[str] | None = None) -> None:
        self.turns = list(turns)
        self.plans = list(plans or [])
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" not in kwargs:
            return {"message": {"content": self.plans.pop(0)}, "done_reason": "stop"}
        reply = self.turns.pop(0) if self.turns else turn(text="ГОТОВО")
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def generate(self, **kwargs):
        return SimpleNamespace(response="Тестовый чат")

    @property
    def tool_calls(self) -> list[dict]:
        return [c for c in self.calls if "tools" in c]


class FakePipeline:
    def __init__(self, events=None, delay: float = 0.0, log=None, name: str = ""):
        self.events = events or []
        self.delay = delay
        self.log = log
        self.name = name
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self._run()

    async def _run(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        for event in self.events:
            yield event
        if self.log is not None:
            self.log.append(f"{self.name}:end")


def answer(text: str) -> list[dict]:
    return [
        {"type": "chunk", "content": {"text": text, "done": False}},
        {"type": "chunk", "content": {"text": "", "done": True}},
    ]


@pytest.fixture
def make_orchestrator(monkeypatch, fake_urban, state_store):
    monkeypatch.setenv("ORCHESTRATOR_ENGINE", "adk")
    monkeypatch.setattr(
        "src.agents.services.orchestrator.orchestrator_adk._tools_unsupported", False
    )

    def build(llm):
        monkeypatch.setattr(
            "src.agents.model_clients.base_client.build_llm_adapter",
            lambda *a, **k: llm,
        )
        from src.agents.services.orchestrator.orchestrator_service import (
            OrchestratorService,
        )

        app_config = SimpleNamespace(
            DVD_MCP_URL="http://dvd",
            NORM_GRAPH_MCP_URL="http://norms",
            URBAN_MCP_URL="http://urban-mcp",
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

    return build


async def run_pipeline(svc, **overrides) -> list[dict]:
    kwargs = dict(
        idu_mcp_client=Mock(),
        effects_mcp_client=Mock(),
        dvd_mcp_client=Mock(),
        normgraph_mcp_client=Mock(),
        token="tok",
        model="m",
        temperature=0.5,
        user_query="запрос",
        scenario_id=772,
    )
    kwargs.update(overrides)
    return [event async for event in svc.run_orchestration_pipeline(**kwargs)]


async def test_tool_calls_become_plan_and_steps_with_unchanged_contract(
    make_orchestrator,
):
    llm = ToolScriptedLlm(
        turn(
            call("run_provision", "c1", task="Рассчитай обеспеченность школами"),
            call("run_documents", "c2", task="Объясни норматив обеспеченности"),
        ),
        turn(text="ГОТОВО"),
    )
    svc = make_orchestrator(llm)
    provision = FakePipeline(answer("Обеспеченность 82%"))
    documents = FakePipeline(answer("Норматив — 120 мест"))
    svc.provision_service.run_provision_pipeline = provision
    svc.dvd_service.run_document_qa_pipeline = documents

    events = await run_pipeline(svc)

    assert types_of(events) == [
        "pipeline_started",
        "service_event",
        "status",
        "plan",
        "step_started",
        "step_event",
        "step_event",
        "step_finished",
        "step_started",
        "step_event",
        "step_event",
        "step_finished",
        "orchestrator_final",
    ]
    plan = events_of_type(events, "plan")[0]["content"]["steps"]
    assert [(s["step"], s["agent"]) for s in plan] == [
        (1, "provision"),
        (2, "documents"),
    ]
    final = events_of_type(events, "orchestrator_final")[0]["content"]["steps"]
    assert [s["status"] for s in final] == ["completed", "completed"]
    # The planned steps run in order: the second one sees the first digest.
    assert "Обеспеченность 82%" in documents.calls[0]["context_note"]


async def test_agent_requests_go_through_the_adapter_with_tools(make_orchestrator):
    llm = ToolScriptedLlm(
        turn(call("run_provision", "c1", task="Обеспеченность школами")),
        turn(text="ГОТОВО"),
    )
    svc = make_orchestrator(llm)
    svc.provision_service.run_provision_pipeline = FakePipeline(answer("82%"))

    await run_pipeline(svc, user_query="обеспеченность школами")

    first, second = llm.tool_calls
    assert first["request_kind"] == "orchestrator.route"
    assert first["think"] is False
    assert first["messages"][0]["role"] == "system"
    assert "маршрутизатор" in first["messages"][0]["content"]
    assert "ask_clarification" in first["messages"][0]["content"]
    assert first["messages"][-1] == {
        "role": "user",
        "content": "обеспеченность школами",
    }
    names = {tool["function"]["name"] for tool in first["tools"]}
    assert {"run_provision", "run_documents", "ask_clarification"} <= names
    # The second turn carries the call and its result, paired by id.
    assistant, result = second["messages"][-2:]
    assert assistant["tool_calls"][0]["id"] == "c1"
    assert result["role"] == "tool" and result["tool_call_id"] == "c1"
    assert json.loads(result["content"])["status"] == "completed"


async def test_ask_clarification_calls_no_agents(make_orchestrator):
    llm = ToolScriptedLlm(
        turn(call("ask_clarification", "c1", question="Что именно рассчитать?"))
    )
    svc = make_orchestrator(llm)
    provision = FakePipeline(answer("x"))
    svc.provision_service.run_provision_pipeline = provision

    events = await run_pipeline(svc)

    assert events_of_type(events, "clarification")[0]["content"] == {
        "question": "Что именно рассчитать?"
    }
    assert not events_of_type(events, "plan")
    assert not provision.calls
    assert len(llm.tool_calls) == 1


async def test_plain_text_answer_is_a_clarification(make_orchestrator):
    llm = ToolScriptedLlm(turn(text="Уточните, какой сервис вас интересует."))
    svc = make_orchestrator(llm)

    events = await run_pipeline(svc)

    question = events_of_type(events, "clarification")[0]["content"]["question"]
    assert question == "Уточните, какой сервис вас интересует."


async def test_unavailable_agent_turns_the_plan_into_a_clarification(
    make_orchestrator,
):
    # Without a scenario the calculation agents are not offered.
    llm = ToolScriptedLlm(turn(call("run_provision", "c1", task="Обеспеченность")))
    svc = make_orchestrator(llm)
    provision = FakePipeline(answer("x"))
    svc.provision_service.run_provision_pipeline = provision

    events = await run_pipeline(svc, scenario_id=None)

    assert events_of_type(events, "clarification")
    assert not provision.calls


async def test_agent_adds_a_step_after_seeing_the_first_result(make_orchestrator):
    llm = ToolScriptedLlm(
        turn(call("run_norms", "c1", task="Ограничения для школ")),
        turn(call("run_documents", "c2", task="Тексты норм для школ")),
        turn(text="ГОТОВО"),
    )
    svc = make_orchestrator(llm)
    svc.normgraph_service.run_norms_qa_pipeline = FakePipeline(answer("граф"))
    documents = FakePipeline(answer("документы"))
    svc.dvd_service.run_document_qa_pipeline = documents

    events = await run_pipeline(svc, user_query="Нормы для школ по СП 42")

    assert len(events_of_type(events, "plan")[0]["content"]["steps"]) == 1
    started = events_of_type(events, "step_started")
    assert [(e["content"]["step"], e["content"]["agent"]) for e in started] == [
        (1, "norms"),
        (2, "documents"),
    ]
    assert "граф" in documents.calls[0]["context_note"]
    final = events_of_type(events, "orchestrator_final")[0]["content"]["steps"]
    assert [s["step"] for s in final] == [1, 2]


async def test_step_waiting_for_the_user_ends_the_run(make_orchestrator):
    llm = ToolScriptedLlm(
        turn(
            call("run_provision", "c1", task="Обеспеченность"),
            call("run_documents", "c2", task="Объясни результат"),
        )
    )
    svc = make_orchestrator(llm)
    svc.provision_service.run_provision_pipeline = FakePipeline(
        [{"type": "clarification", "content": {"question": "Какой сервис?"}}]
    )
    documents = FakePipeline(answer("x"))
    svc.dvd_service.run_document_qa_pipeline = documents

    events = await run_pipeline(svc)

    final = events_of_type(events, "orchestrator_final")[0]["content"]["steps"]
    assert [s["status"] for s in final] == ["needs_clarification", "skipped"]
    assert not documents.calls
    assert len(llm.tool_calls) == 1  # no further model turn after the stop


async def test_independent_qa_steps_run_together_but_stream_in_order(
    make_orchestrator,
):
    llm = ToolScriptedLlm(
        turn(
            call("run_norms", "c1", task="Ограничения для школ"),
            call("run_documents", "c2", task="Документы о школах"),
        ),
        turn(text="ГОТОВО"),
    )
    svc = make_orchestrator(llm)
    log: list[str] = []
    svc.normgraph_service.run_norms_qa_pipeline = FakePipeline(
        answer("граф"), 0.2, log, "norms"
    )
    documents = FakePipeline(answer("документы"), 0.0, log, "documents")
    svc.dvd_service.run_document_qa_pipeline = documents

    events = await run_pipeline(svc, user_query="Какие регламенты школ?")

    assert log.index("documents:end") < log.index("norms:end")
    steps = [
        e["content"]["step"]
        for e in events
        if e["type"] in {"step_started", "step_event", "step_finished"}
    ]
    assert steps == sorted(steps)
    assert documents.calls[0]["context_note"] is None


async def test_server_without_tool_calling_falls_back_to_the_planner(
    make_orchestrator,
):
    plan = json.dumps(
        {
            "mode": "execute",
            "steps": [{"agent": "provision", "task": "Обеспеченность школами"}],
            "clarification_question": None,
        },
        ensure_ascii=False,
    )
    llm = ToolScriptedLlm(
        LlmResponseError('"auto" tool choice requires --enable-auto-tool-choice', 400),
        plans=[plan],
    )
    svc = make_orchestrator(llm)
    provision = FakePipeline(answer("82%"))
    svc.provision_service.run_provision_pipeline = provision

    events = await run_pipeline(svc)

    assert events_of_type(events, "plan")[0]["content"]["steps"][0]["agent"] == (
        "provision"
    )
    assert provision.calls
    assert any(c.get("request_kind") == "orchestrator.plan" for c in llm.calls)

    # The next request goes to the planner without a failing tool call first.
    llm.plans.append(plan)
    calls_before = len(llm.tool_calls)
    await run_pipeline(svc)
    assert len(llm.tool_calls) == calls_before


async def test_transient_failure_does_not_disable_the_agent(make_orchestrator):
    from src.agents.services.orchestrator import orchestrator_adk

    plan = json.dumps(
        {
            "mode": "execute",
            "steps": [{"agent": "provision", "task": "Обеспеченность"}],
            "clarification_question": None,
        },
        ensure_ascii=False,
    )
    llm = ToolScriptedLlm(LlmResponseError("timeout", 503), plans=[plan])
    svc = make_orchestrator(llm)
    svc.provision_service.run_provision_pipeline = FakePipeline(answer("82%"))

    events = await run_pipeline(svc)

    assert events_of_type(events, "plan")
    assert orchestrator_adk.adk_available()


async def test_model_failure_after_steps_keeps_their_results(make_orchestrator):
    llm = ToolScriptedLlm(
        turn(call("run_provision", "c1", task="Обеспеченность")),
        LlmResponseError("server went away", 503),
    )
    svc = make_orchestrator(llm)
    svc.provision_service.run_provision_pipeline = FakePipeline(answer("82%"))

    events = await run_pipeline(svc)

    final = events_of_type(events, "orchestrator_final")[0]["content"]["steps"]
    assert [s["status"] for s in final] == ["completed"]


def test_ordered_bus_holds_later_steps_until_earlier_finish():
    bus = OrderedStepBus()
    bus.open(1)
    bus.open(2)
    bus.emit(2, {"e": "2a"})
    bus.emit(1, {"e": "1a"})
    bus.close(2)
    bus.emit(1, {"e": "1b"})
    bus.close(1)
    out = []
    while not bus.queue.empty():
        out.append(bus.queue.get_nowait()["e"])
    assert out == ["1a", "1b", "2a"]
