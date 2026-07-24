"""Unit tests for ``OrchestratorService`` — event flow, digests, failure policy."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.helpers import events_of_type, types_of


def orchestration_plan_json(
    steps: list[dict] | None = None,
    mode: str = "execute",
    clarification_question: str | None = None,
) -> str:
    return json.dumps(
        {
            "mode": mode,
            "steps": steps or [],
            "clarification_question": clarification_question,
        },
        ensure_ascii=False,
    )


class FakePipeline:
    """A canned sub-agent pipeline: records call kwargs, replays events, may raise."""

    def __init__(
        self, events: list[dict] | None = None, raise_exc: Exception | None = None
    ) -> None:
        self.events = events or []
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self._run()

    async def _run(self):
        for event in self.events:
            yield event
        if self.raise_exc is not None:
            raise self.raise_exc


PROVISION_EVENTS = [
    {"type": "pipeline_started", "content": {"request_id": "inner-prov"}},
    {"type": "status", "content": {"status": "effects_calculation", "text": "Считаю…"}},
    {"type": "chunk", "content": {"text": "Обеспеченность школами 82%", "done": False}},
    {
        "type": "feature_collection",
        "content": {"name": "Слой обеспеченности", "feature_collection": {}},
    },
    {"type": "chunk", "content": {"text": "", "done": True}},
]

RESTRICTION_EVENTS = [
    {"type": "pipeline_started", "content": {"request_id": "inner-restr"}},
    {"type": "chunk", "content": {"text": "Ограничения построены", "done": False}},
    {
        "type": "feature_collection",
        "content": {"name": "Зоны ограничений", "feature_collection": {}},
    },
    {"type": "chunk", "content": {"text": "", "done": True}},
]


@pytest.fixture
def orchestrator(monkeypatch, fake_llm, fake_urban, state_store):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.AsyncOllamaClient",
        lambda *a, **k: fake_llm,
    )
    from src.agents.services.orchestrator_service import OrchestratorService

    app_config = SimpleNamespace(
        DVD_MCP_URL="http://dvd",
        NORM_GRAPH_MCP_URL="http://norms",
        URBAN_DATA_MCP_URL="http://urban-data",
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
        Mock(),
        app_config,
    )
    svc.create_chat = AsyncMock(return_value=("chat-xyz", "Тестовый чат"))
    svc.get_chat_messages = AsyncMock(return_value=SimpleNamespace(messages=[]))
    svc.add_single_message = AsyncMock()
    svc.add_complex_message = AsyncMock()
    return svc


async def run_pipeline(svc, **overrides) -> list[dict]:
    kwargs = dict(
        idu_mcp_client=Mock(),
        effects_mcp_client=Mock(),
        dvd_mcp_client=Mock(),
        normgraph_mcp_client=Mock(),
        urban_data_mcp_client=Mock(),
        token="tok",
        model="m",
        temperature=0.5,
        user_query="запрос",
        scenario_id=772,
    )
    kwargs.update(overrides)
    return [event async for event in svc.run_orchestration_pipeline(**kwargs)]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_step_event_order(orchestrator, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json(
            [{"agent": "provision", "task": "Рассчитай обеспеченность школами"}]
        )
    ]
    pipeline = FakePipeline(PROVISION_EVENTS)
    orchestrator.provision_service.run_provision_pipeline = pipeline

    events = await run_pipeline(orchestrator)

    assert types_of(events) == [
        "pipeline_started",
        "service_event",  # chat_created
        "status",  # planning
        "plan",
        "step_started",
        "step_event",  # status
        "step_event",  # chunk
        "step_event",  # feature_collection
        "step_event",  # done chunk
        "step_finished",
        "orchestrator_final",
    ]
    step_events = events_of_type(events, "step_event")
    assert all(e["content"]["step"] == 1 for e in step_events)
    assert all(e["content"]["agent"] == "provision" for e in step_events)
    # the inner pipeline_started is suppressed
    inner_types = [e["content"]["event"]["type"] for e in step_events]
    assert "pipeline_started" not in inner_types

    finished = events_of_type(events, "step_finished")[0]["content"]
    assert finished["status"] == "completed"
    assert "Обеспеченность школами 82%" in finished["summary"]
    assert "Слой обеспеченности" in finished["summary"]

    final = events_of_type(events, "orchestrator_final")[0]["content"]
    assert [s["status"] for s in final["steps"]] == ["completed"]


@pytest.mark.asyncio
async def test_sub_agents_run_without_persistence_and_own_request_ids(
    orchestrator, fake_llm
):
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": "provision", "task": "задача"}])
    ]
    pipeline = FakePipeline(PROVISION_EVENTS)
    orchestrator.provision_service.run_provision_pipeline = pipeline

    events = await run_pipeline(orchestrator)

    outer_request_id = events_of_type(events, "pipeline_started")[0]["content"][
        "request_id"
    ]
    call = pipeline.calls[0]
    assert call["persist_history"] is False
    assert call["request_id"] != outer_request_id
    assert (
        call["request_id"]
        == events_of_type(events, "step_started")[0]["content"]["step_request_id"]
    )


@pytest.mark.asyncio
async def test_second_step_receives_first_step_digest(orchestrator, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json(
            [
                {"agent": "restriction", "task": "Построй ограничения"},
                {"agent": "provision", "task": "Оцени обеспеченность"},
            ]
        )
    ]
    restriction = FakePipeline(RESTRICTION_EVENTS)
    provision = FakePipeline(PROVISION_EVENTS)
    orchestrator.restriction_service.run_restriction_execution_pipline = restriction
    orchestrator.provision_service.run_provision_pipeline = provision

    events = await run_pipeline(orchestrator)

    assert restriction.calls[0]["user_query"] == "Построй ограничения"
    second_query = provision.calls[0]["user_query"]
    assert second_query.startswith("Оцени обеспеченность")
    assert "Контекст — результаты предыдущих шагов" in second_query
    assert "Ограничения построены" in second_query
    assert "Зоны ограничений" in second_query

    final = events_of_type(events, "orchestrator_final")[0]["content"]
    assert [s["status"] for s in final["steps"]] == ["completed", "completed"]


# ---------------------------------------------------------------------------
# Clarification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clarification_plan_calls_no_agents(orchestrator, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json(
            mode="needs_clarification",
            clarification_question="Уточните, что нужно сделать.",
        )
    ]
    pipeline = FakePipeline(PROVISION_EVENTS)
    orchestrator.provision_service.run_provision_pipeline = pipeline

    events = await run_pipeline(orchestrator)

    clarifications = events_of_type(events, "clarification")
    assert len(clarifications) == 1
    assert clarifications[0]["content"]["question"] == "Уточните, что нужно сделать."
    assert not events_of_type(events, "plan")
    assert not events_of_type(events, "step_started")
    assert not pipeline.calls
    # the clarification is persisted as the assistant answer
    await asyncio.sleep(0)
    assert orchestrator.add_complex_message.await_count == 1


# ---------------------------------------------------------------------------
# Failure policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_step_aborts_remaining_steps(orchestrator, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json(
            [
                {"agent": "restriction", "task": "Построй ограничения"},
                {"agent": "provision", "task": "Оцени обеспеченность"},
            ]
        )
    ]
    failing = FakePipeline(
        [
            {"type": "pipeline_started", "content": {"request_id": "inner"}},
            {"type": "error", "content": {"message": "boom", "traceback": "tb"}},
        ]
    )
    provision = FakePipeline(PROVISION_EVENTS)
    orchestrator.restriction_service.run_restriction_execution_pipline = failing
    orchestrator.provision_service.run_provision_pipeline = provision

    events = await run_pipeline(orchestrator)

    finished = events_of_type(events, "step_finished")
    assert len(finished) == 1
    assert finished[0]["content"]["status"] == "failed"
    assert not provision.calls
    final = events_of_type(events, "orchestrator_final")[0]["content"]
    assert [s["status"] for s in final["steps"]] == ["failed", "skipped"]
    # the inner error event is forwarded so the client sees the reason
    inner_types = [
        e["content"]["event"]["type"] for e in events_of_type(events, "step_event")
    ]
    assert "error" in inner_types


@pytest.mark.asyncio
async def test_step_exception_is_contained(orchestrator, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": "provision", "task": "задача"}])
    ]
    orchestrator.provision_service.run_provision_pipeline = FakePipeline(
        [{"type": "status", "content": {"status": "service_lookup", "text": "…"}}],
        raise_exc=RuntimeError("downstream exploded"),
    )

    events = await run_pipeline(orchestrator)

    finished = events_of_type(events, "step_finished")[0]["content"]
    assert finished["status"] == "failed"
    final = events_of_type(events, "orchestrator_final")[0]["content"]
    assert [s["status"] for s in final["steps"]] == ["failed"]


@pytest.mark.asyncio
async def test_token_expired_forwarded_verbatim(orchestrator, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": "provision", "task": "задача"}])
    ]
    token_expired = {
        "type": "token_expired",
        "content": {"request_id": "inner-id", "message": "Токен истёк"},
    }
    orchestrator.provision_service.run_provision_pipeline = FakePipeline(
        [token_expired, *PROVISION_EVENTS[1:]]
    )

    events = await run_pipeline(orchestrator)

    forwarded = [
        e["content"]["event"]
        for e in events_of_type(events, "step_event")
        if e["content"]["event"]["type"] == "token_expired"
    ]
    assert forwarded == [token_expired]


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_digest_keeps_only_last_iteration(orchestrator, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": "documents", "task": "вопрос"}])
    ]
    orchestrator.dvd_service.run_document_qa_pipeline = FakePipeline(
        [
            {
                "type": "chunk",
                "content": {"text": "черновик", "done": False, "iteration": 1},
            },
            {
                "type": "chunk",
                "content": {"text": "итоговый ответ", "done": False, "iteration": 2},
            },
            {"type": "chunk", "content": {"text": "", "done": True, "iteration": 2}},
        ]
    )

    events = await run_pipeline(orchestrator)

    summary = events_of_type(events, "step_finished")[0]["content"]["summary"]
    assert summary == "итоговый ответ"


@pytest.mark.asyncio
async def test_digest_is_capped(orchestrator, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": "provision", "task": "задача"}])
    ]
    orchestrator.provision_service.run_provision_pipeline = FakePipeline(
        [{"type": "chunk", "content": {"text": "х" * 5000, "done": False}}]
    )

    events = await run_pipeline(orchestrator)

    summary = events_of_type(events, "step_finished")[0]["content"]["summary"]
    assert len(summary) <= orchestrator.DIGEST_MAX_CHARS
    assert summary.endswith("…")


# ---------------------------------------------------------------------------
# Reconnect (v1: replay-only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconnect_replays_buffered_events_only(
    orchestrator, fake_llm, state_store
):
    request_id = state_store.new_request_id()
    await state_store.create(
        request_id,
        chat_id="chat-xyz",
        user_query="запрос",
        scenario_id=772,
        model="m",
        temperature=0.5,
    )
    buffered = [
        {"type": "pipeline_started", "content": {"request_id": request_id}},
        {"type": "status", "content": {"status": "planning", "text": "…"}},
    ]
    for event in buffered:
        await state_store.buffer_event(request_id, event)

    events = await run_pipeline(orchestrator, request_id=request_id)

    assert events == buffered
    assert not fake_llm.chat_calls  # the planner is not re-run
    orchestrator.create_chat.assert_not_awaited()
