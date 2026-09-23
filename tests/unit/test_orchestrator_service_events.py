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
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *a, **k: fake_llm,
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


@pytest.mark.asyncio
async def test_compliance_dispatches_normative_pipeline(orchestrator, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json(
            [
                {
                    "agent": "compliance",
                    "task": "Проверь нарушения отступов домов от дорог",
                }
            ]
        )
    ]
    pipeline = FakePipeline(RESTRICTION_EVENTS)
    orchestrator.restriction_service.run_compliance_pipeline = pipeline
    normgraph = Mock()
    events = await run_pipeline(orchestrator, normgraph_mcp_client=normgraph)
    assert pipeline.calls[0]["normgraph_mcp_client"] is normgraph
    assert pipeline.calls[0]["scenario_id"] == 772
    assert pipeline.calls[0]["persist_history"] is False
    assert (
        events_of_type(events, "step_finished")[0]["content"]["status"] == "completed"
    )


@pytest.mark.asyncio
async def test_inner_clarification_is_not_success_or_downstream_evidence(
    orchestrator, fake_llm
):
    fake_llm.json_responses = [
        orchestration_plan_json(
            [
                {"agent": "provision", "task": "Обеспеченность"},
                {"agent": "documents", "task": "Объясни результат"},
            ]
        )
    ]
    orchestrator.provision_service.run_provision_pipeline = FakePipeline(
        [{"type": "clarification", "content": {"question": "Какой сервис рассчитать?"}}]
    )
    downstream = FakePipeline([])
    orchestrator.dvd_service.run_document_qa_pipeline = downstream
    events = await run_pipeline(orchestrator)
    final = events_of_type(events, "orchestrator_final")[0]["content"]
    assert [s["status"] for s in final["steps"]] == ["needs_clarification", "skipped"]
    assert "Какой сервис" in final["steps"][0]["summary"]
    assert not downstream.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["error", "pipeline_failed"])
async def test_failed_draft_is_not_persisted_as_answer(
    orchestrator, fake_llm, terminal
):
    fake_llm.json_responses = [
        orchestration_plan_json(
            [{"agent": "documents", "task": "Найди подтверждённую норму"}]
        )
    ]
    orchestrator.dvd_service.run_document_qa_pipeline = FakePipeline(
        [
            {
                "type": "chunk",
                "content": {"text": "Выдуманная норма 999 м", "done": False},
            },
            {"type": terminal, "content": {"message": "Источник не получен"}},
        ]
    )
    events = await run_pipeline(orchestrator)
    await asyncio.sleep(0)
    finished = events_of_type(events, "step_finished")[0]["content"]
    assert finished["status"] == "failed"
    assert "Выдуманная" not in finished["summary"]
    parts = orchestrator.add_complex_message.await_args.args[3]
    assert all("Выдуманная" not in part.payload.text for part in parts)


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
async def test_table_event_is_persisted_with_the_orchestrator_answer(
    orchestrator, fake_llm
):
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": "scenario_data", "task": "Объекты"}])
    ]
    table = {
        "name": "objects",
        "title": "Объекты",
        "columns": [{"key": "name", "label": "Название"}],
        "rows": [{"name": "Школа"}],
        "total_rows": 1,
        "complete": True,
    }
    orchestrator.scenario_data_service = SimpleNamespace(
        run_scenario_data_pipeline=FakePipeline(
            [
                {"type": "table", "content": table},
                {
                    "type": "chunk",
                    "content": {"text": "Полный перечень в таблице.", "done": True},
                },
            ]
        )
    )

    await run_pipeline(orchestrator, urban_mcp_client=Mock())
    await asyncio.sleep(0)

    orchestrator.add_complex_message.assert_awaited_once()
    parts = orchestrator.add_complex_message.await_args.args[3]
    assert any(
        part.kind == "table" and part.payload.name == "objects" for part in parts
    )


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
async def test_scenario_data_agent_runs_without_scenario_id(orchestrator, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json(
            [{"agent": "scenario_data", "task": "Перечисли типы сервисов"}]
        )
    ]
    pipeline = FakePipeline(
        [{"type": "chunk", "content": {"text": "Школы", "done": True}}]
    )
    orchestrator.scenario_data_service = SimpleNamespace(
        run_scenario_data_pipeline=pipeline
    )

    events = await run_pipeline(
        orchestrator,
        scenario_id=None,
        urban_mcp_client=Mock(),
    )

    assert (
        events_of_type(events, "step_finished")[0]["content"]["status"] == "completed"
    )
    assert pipeline.calls[0]["scenario_id"] is None
    assert pipeline.calls[0]["persist_history"] is False


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
    assert "normgraph_mcp_client" not in restriction.calls[0]
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


async def test_pzz_dispatch_forwards_inputs_and_wraps_report(orchestrator, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": "pzz", "task": "Проверь ПЗЗ"}])
    ]
    orchestrator.app_config.PZZ_MCP_URL = "http://pzz/mcp"
    pipeline = FakePipeline(
        [
            {"type": "object_zone_fit", "content": {"summary": {"total": 1}}},
            {
                "type": "chunk",
                "content": {"text": "Один объект проверен", "done": True},
            },
        ]
    )
    orchestrator.pzz_service = SimpleNamespace(run_pzz_pipeline=pipeline)
    client = SimpleNamespace(api_client=object())
    inputs = {"mode": "scenario", "year": 2026, "source": "PZZ"}
    events = await run_pipeline(orchestrator, pzz_mcp_client=client, pzz_inputs=inputs)
    assert pipeline.calls[0]["pzz_mcp_client"] is client
    assert pipeline.calls[0]["pzz_api_client"] is client.api_client
    assert pipeline.calls[0]["inputs"] is inputs
    assert pipeline.calls[0]["persist_history"] is False
    assert any(
        e["type"] == "step_event" and e["content"]["event"]["type"] == "object_zone_fit"
        for e in events
    )
    assert events[-1]["content"]["steps"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_file_event_closes_the_stream_and_is_persisted(orchestrator, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": "compliance", "task": "Проверь нормы"}])
    ]
    descriptor = {
        "name": "compliance_report",
        "title": "Отчёт о проверке соответствия нормам",
        "role": "result",
        "url": "http://gmart/files/compliance_report/abc",
        "download_url": "http://gmart/files/compliance_report/abc?download=1",
        "filename": "compliance_report_772_20260923-1200.md",
        "mime_type": "text/markdown",
        "source_service": "gmart",
    }
    orchestrator.restriction_service.run_compliance_pipeline = FakePipeline(
        [
            {"type": "chunk", "content": {"text": "Проверка завершена.", "done": True}},
            {"type": "file", "content": descriptor},
        ]
    )

    events = await run_pipeline(orchestrator, normgraph_mcp_client=Mock())
    await asyncio.sleep(0)

    assert types_of(events)[-2:] == ["orchestrator_final", "file"]
    assert events[-1]["content"] == descriptor
    assert all(
        event["content"]["event"]["type"] != "file"
        for event in events_of_type(events, "step_event")
    )
    parts = orchestrator.add_complex_message.await_args.args[3]
    file_parts = [part for part in parts if part.kind == "file"]
    assert len(file_parts) == 1
    assert "download_url" not in file_parts[0].payload
    assert file_parts[0].payload["url"] == descriptor["url"]


@pytest.mark.asyncio
async def test_documents_step_gets_the_user_question_and_router_task(
    orchestrator, fake_llm
):
    fake_llm.json_responses = [
        orchestration_plan_json(
            [{"agent": "documents", "task": "Найти документы о школах"}]
        )
    ]
    documents = FakePipeline([{"type": "chunk", "content": {"text": "", "done": True}}])
    orchestrator.dvd_service.run_document_qa_pipeline = documents

    await run_pipeline(orchestrator, user_query="Какие регламенты застройки школ?")

    call = documents.calls[0]
    assert call["user_query"] == "Какие регламенты застройки школ?"
    assert call["task"] == "Найти документы о школах"
    assert call["context_note"] is None


class SlowPipeline(FakePipeline):
    """Records when its producer starts and finishes; yields after ``delay``."""

    def __init__(self, events, delay, log, name):
        super().__init__(events)
        self.delay, self.log, self.name = delay, log, name

    async def _run(self):
        self.log.append(f"{self.name}:start")
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.log.append(f"{self.name}:cancelled")
            raise
        for event in self.events:
            yield event
        self.log.append(f"{self.name}:end")


@pytest.mark.asyncio
async def test_independent_qa_steps_run_together_but_stream_in_order(
    orchestrator, fake_llm
):
    fake_llm.json_responses = [
        orchestration_plan_json(
            [
                {"agent": "norms", "task": "Ограничения для школ"},
                {"agent": "documents", "task": "Документы о школах"},
            ]
        )
    ]
    log: list[str] = []
    done = {"type": "chunk", "content": {"text": "", "done": True}}
    norms = SlowPipeline(
        [{"type": "chunk", "content": {"text": "граф", "done": False}}, done],
        0.2,
        log,
        "norms",
    )
    documents = SlowPipeline(
        [{"type": "chunk", "content": {"text": "документы", "done": False}}, done],
        0.0,
        log,
        "documents",
    )
    orchestrator.normgraph_service.run_norms_qa_pipeline = norms
    orchestrator.dvd_service.run_document_qa_pipeline = documents

    started = asyncio.get_running_loop().time()
    events = await run_pipeline(orchestrator, user_query="Какие регламенты школ?")

    # The documents producer finished while norms was still working...
    assert log.index("documents:end") < log.index("norms:end")
    # ...yet the client sees step 1 fully before step 2.
    steps = [
        e["content"]["step"]
        for e in events
        if e["type"] in {"step_started", "step_event", "step_finished"}
    ]
    assert steps == sorted(steps)
    final = events_of_type(events, "orchestrator_final")[0]["content"]
    assert [s["status"] for s in final["steps"]] == ["completed", "completed"]
    assert documents.calls[0]["context_note"] is None
    assert asyncio.get_running_loop().time() - started < 0.4


@pytest.mark.asyncio
async def test_failed_first_step_cancels_the_step_running_ahead(orchestrator, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json(
            [
                {"agent": "norms", "task": "Ограничения"},
                {"agent": "documents", "task": "Документы"},
            ]
        )
    ]
    log: list[str] = []
    norms = FakePipeline(raise_exc=RuntimeError("boom"))
    documents = SlowPipeline([], 5.0, log, "documents")
    orchestrator.normgraph_service.run_norms_qa_pipeline = norms
    orchestrator.dvd_service.run_document_qa_pipeline = documents

    events = await run_pipeline(orchestrator)

    final = events_of_type(events, "orchestrator_final")[0]["content"]
    assert [s["status"] for s in final["steps"]] == ["failed", "skipped"]
    await asyncio.sleep(0)
    assert log == ["documents:start", "documents:cancelled"]
