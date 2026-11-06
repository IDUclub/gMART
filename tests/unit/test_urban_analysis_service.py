"""Hermetic concurrency, streaming order, failure isolation and cancellation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.agents.common.exceptions.token_exceptions import PipelineSuspendedError
from src.agents.dto.pzz_request_dto import PzzInputs
from src.agents.schema.orchestrator_response import OrchestratorResponse
from src.agents.services.pipeline_state import PipelineStatus
from src.agents.services.urban_analysis.urban_analysis_service import (
    ANALYSIS_STEPS,
    UrbanAnalysisResult,
    UrbanAnalysisService,
)
from tests.helpers import events_of_type


def chunk(text):
    return {"type": "chunk", "content": {"text": text, "done": True}}


@pytest.fixture
def analysis(fake_llm, state_store):
    async def success(**kwargs):
        yield chunk("Проверка выполнена")

    pzz = SimpleNamespace(
        run_pzz_pipeline=success,
        resolve_inputs=AsyncMock(return_value=PzzInputs(mode="scenario")),
    )
    return UrbanAnalysisService(
        SimpleNamespace(run_compliance_pipeline=success),
        SimpleNamespace(run_provision_pipeline=success),
        pzz,
        SimpleNamespace(
            get_scenario_functional_zone_sources=AsyncMock(
                return_value=[{"name": "PZZ", "year": 2025}]
            )
        ),
        state_store,
        llm_client=fake_llm,
    )


def pipeline(analysis, **overrides):
    kwargs = dict(
        idu_mcp_client=Mock(),
        effects_mcp_client=Mock(),
        dvd_mcp_client=Mock(),
        normgraph_mcp_client=Mock(),
        pzz_mcp_client=Mock(),
        token="t",
        model="m",
        temperature=0.1,
        user_query="Градоанализ",
        scenario_id=17,
    )
    kwargs.update(overrides)
    return analysis.run_urban_analysis_pipeline(**kwargs)


async def collect(analysis, **overrides):
    return [event async for event in pipeline(analysis, **overrides)]


@pytest.mark.asyncio
async def test_checks_start_together_and_stream_in_plan_order(analysis, fake_llm):
    started = [asyncio.Event() for _ in range(3)]
    calls = []

    def check(index):
        async def run(**kwargs):
            calls.append((index, kwargs))
            started[index].set()
            await asyncio.gather(*(event.wait() for event in started))
            yield {
                "type": "pipeline_started",
                "content": {"request_id": kwargs["request_id"]},
            }
            yield chunk(f"Результат {index}")

        return run

    analysis.restriction_service.run_compliance_pipeline = check(0)
    analysis.provision_service.run_provision_pipeline = check(1)
    analysis.pzz_service.run_pzz_pipeline = check(2)
    fake_llm.answer_texts = ["Итоговый отчёт"]
    result = UrbanAnalysisResult()
    emitted = await asyncio.wait_for(collect(analysis, result=result), 3)
    step_numbers = [
        e["content"]["step"]
        for e in emitted
        if e["type"] in {"step_started", "step_event", "step_finished"}
    ]
    assert step_numbers == sorted(step_numbers)
    assert len(calls) == 3 and all(event.is_set() for event in started)
    assert len({kwargs["request_id"] for _, kwargs in calls}) == 3
    for index, kwargs in calls:
        assert kwargs["persist_history"] is False
        assert kwargs["user_query"] == ANALYSIS_STEPS[index][1]
    compliance = next(kwargs for i, kwargs in calls if i == 0)
    assert compliance["conversation_key"] is None and compliance["upload_ids"] is None
    assert result.status == PipelineStatus.DONE
    assert [s["agent"] for s in result.steps] == [a.value for a, _ in ANALYSIS_STEPS]
    assert result.steps[3]["summary"] == "Итоговый отчёт"
    for event in emitted:
        OrchestratorResponse.model_validate(event)


@pytest.mark.asyncio
async def test_token_expired_bypasses_blocked_first_check(analysis):
    release_first = asyncio.Event()
    refresh = asyncio.Event()
    first_finished = asyncio.Event()

    async def first(**kwargs):
        await release_first.wait()
        yield chunk("Нормы")
        first_finished.set()

    async def second(**kwargs):
        yield {"type": "token_expired", "content": {"request_id": kwargs["request_id"]}}
        await refresh.wait()
        yield chunk("Обеспеченность")

    analysis.restriction_service.run_compliance_pipeline = first
    analysis.provision_service.run_provision_pipeline = second

    async def consume():
        emitted = []
        async for event in pipeline(analysis):
            emitted.append(event)
            if (
                event["type"] == "step_event"
                and event["content"]["event"]["type"] == "token_expired"
            ):
                assert event["content"]["step"] == 2
                assert event["content"]["event"]["content"]["request_id"]
                assert not first_finished.is_set()
                refresh.set()
                release_first.set()
        return emitted

    emitted = await asyncio.wait_for(consume(), 3)
    assert len(events_of_type(emitted, "step_finished")) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal,status",
    [
        ("error", "failed"),
        ("exception", "failed"),
        ("clarification", "needs_clarification"),
        ("pipeline_suspended", "suspended"),
        ("suspension_exception", "suspended"),
        ("pipeline_failed", "failed"),
    ],
)
async def test_one_terminal_outcome_does_not_stop_other_checks(
    analysis, fake_llm, terminal, status
):
    closed = asyncio.Event()

    async def broken(**kwargs):
        try:
            yield chunk("НЕПОДТВЕРЖДЁННЫЙ ЧЕРНОВИК")
            if terminal == "exception":
                raise RuntimeError("test")
            if terminal == "suspension_exception":
                raise PipelineSuspendedError("token refresh timed out")
            yield {
                "type": terminal,
                "content": {"text": "Причина", "question": "Причина"},
            }
            pytest.fail("Producer consumed after its terminal event")
        finally:
            closed.set()

    analysis.restriction_service.run_compliance_pipeline = broken
    fake_llm.answer_texts = ["Частичный отчёт"]
    result = UrbanAnalysisResult()
    await collect(analysis, result=result)
    assert [s["status"] for s in result.steps] == [
        status,
        "completed",
        "completed",
        "completed",
    ]
    assert closed.is_set()
    assert result.status == PipelineStatus.DONE
    prompt = next(
        message["content"]
        for c in fake_llm.chat_calls
        if c.request_kind == "urban_analysis.report"
        for message in c.messages
        if message["role"] == "user"
    )
    assert "НЕПОДТВЕРЖДЁННЫЙ ЧЕРНОВИК" not in prompt
    assert status in prompt


@pytest.mark.asyncio
async def test_disconnect_cancels_and_awaits_all_check_tasks(analysis):
    started = [asyncio.Event() for _ in range(3)]
    cancelled = [asyncio.Event() for _ in range(3)]

    def blocked(index):
        async def run(**kwargs):
            started[index].set()
            try:
                await asyncio.gather(*(event.wait() for event in started))
                if index == 0:
                    yield chunk("Начало")
                await asyncio.Event().wait()
            finally:
                cancelled[index].set()

        return run

    analysis.restriction_service.run_compliance_pipeline = blocked(0)
    analysis.provision_service.run_provision_pipeline = blocked(1)
    analysis.pzz_service.run_pzz_pipeline = blocked(2)
    stream = pipeline(analysis)

    async def until_chunk():
        async for event in stream:
            if (
                event["type"] == "step_event"
                and event["content"]["event"]["type"] == "chunk"
            ):
                return

    await asyncio.wait_for(until_chunk(), 3)
    await stream.aclose()
    assert all(event.is_set() for event in cancelled)
    assert not any(
        task.get_name().startswith("urban-analysis:") for task in asyncio.all_tasks()
    )


@pytest.mark.asyncio
async def test_all_failed_skips_llm_and_marks_run_failed(analysis, fake_llm):
    async def failed(**kwargs):
        yield {"type": "error", "content": {"text": "Не выполнено"}}

    analysis.provision_service.run_provision_pipeline = failed
    result = UrbanAnalysisResult()
    emitted = await collect(
        analysis, result=result, normgraph_mcp_client=None, pzz_mcp_client=None
    )
    assert result.status == PipelineStatus.FAILED
    assert fake_llm.chat_calls == []
    assert all(s["status"] == "failed" for s in result.steps[:3])
    assert "NormGraph не настроен" in result.steps[0]["summary"]
    assert "ПЗЗ не настроен" in result.steps[2]["summary"]
    assert result.steps[3]["status"] == "completed"
    assert result.steps[3]["summary"]
    assert events_of_type(emitted, "orchestrator_final")


@pytest.mark.asyncio
async def test_no_zones_skips_pzz_but_other_checks_complete(analysis):
    analysis.urban_api_client.get_scenario_functional_zone_sources.return_value = []
    analysis.pzz_service.run_pzz_pipeline = Mock(
        side_effect=AssertionError("must not run")
    )
    result = UrbanAnalysisResult()
    await collect(analysis, result=result)
    assert [s["status"] for s in result.steps[:3]] == [
        "completed",
        "completed",
        "needs_clarification",
    ]
    analysis.pzz_service.run_pzz_pipeline.assert_not_called()


@pytest.mark.asyncio
async def test_report_receives_longer_evidence_than_history_digest(analysis, fake_llm):
    async def long_result(**kwargs):
        yield chunk("X" * 7000)

    analysis.restriction_service.run_compliance_pipeline = long_result
    result = UrbanAnalysisResult()
    await collect(analysis, result=result)
    assert len(result.steps[0]["summary"]) == 1500
    prompt = next(
        call.messages[1]["content"]
        for call in fake_llm.chat_calls
        if call.request_kind == "urban_analysis.report"
    )
    assert "X" * 5999 in prompt
    assert "X" * 6001 not in prompt
