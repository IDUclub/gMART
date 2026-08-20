"""Integration: pipeline checkpoints and replay against a real Redis."""

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
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *a, **k: fake_llm,
    )
    from src.agents.services.dvd_rag_service import DvdRagService

    svc = DvdRagService("http://ollama", Mock(), FakeUrbanApiClient(), state_store)
    svc.create_chat = AsyncMock(return_value=("chat-1", "Тест"))
    svc.get_chat_messages = AsyncMock(return_value=SimpleNamespace(messages=[]))
    svc._schedule_persist_answer = Mock()
    return svc


async def _clear(redis, request_id=_RID):
    for suffix in ("state", "checkpoint", "events"):
        await redis.delete(f"pipeline:{request_id}:{suffix}")


async def test_reconnect_replays_against_real_redis(require_redis, monkeypatch):
    from src.agents.services.pipeline_state import PipelineStateStore

    store = PipelineStateStore(require_redis)
    await _clear(require_redis)

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
    await asyncio.sleep(0.3)

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

    assert mcp2.search_calls == []
    assert fake_llm2.chat_calls == []
    assert final_chunk(second) is not None
    svc2._schedule_persist_answer.assert_not_called()
    await _clear(require_redis)


async def test_compliance_structured_events_and_checkpoints_survive_redis(
    require_redis,
):
    from src.agents.services.pipeline_state import PipelineStateStore, PipelineStep
    from src.agents.services.restriction_parser_service import RestrictionParserService
    from src.agents.services.service_entities.compliance import (
        ComplianceResult,
        ComplianceSummary,
        VerificationCoverage,
    )

    request_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    store = PipelineStateStore(require_redis)
    await _clear(require_redis, request_id)
    await store.create(
        request_id,
        chat_id="chat-1",
        user_query="check",
        scenario_id=772,
        model="m",
        temperature=0,
    )
    result = ComplianceResult(
        restriction_id="r1",
        template="distance_from_source",
        template_version=1,
        verification_status="complete",
        compliance_status="passed",
        coverage=VerificationCoverage(
            applicable_objects=1,
            checked_objects=1,
            unchecked_objects=0,
            fill_rate=1,
        ),
        summary=ComplianceSummary(violated_objects=0, passed_objects=1),
    )
    service = object.__new__(RestrictionParserService)
    service.state_store = store
    service.compliance_executor = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(
                result=result,
                tool_calls=[],
                timings_ms={"template_execution": 1},
            )
        )
    )
    raw_plan = {
        "schema_version": "1.0",
        "template": "distance_from_source",
        "template_version": 1,
        "params": {},
        "source": {"restriction_id": "r1"},
        "planner_status": "auto",
    }
    first = [
        event
        async for event in service._run_executable_compliance(
            mcp_client=object(),
            request_id=request_id,
            scenario_id=772,
            restrictions=[{"id": "r1", "check_plan": raw_plan}],
            checkpoint={},
        )
    ]
    await asyncio.sleep(0.3)
    assert {event["type"] for event in first} >= {
        "check_plan",
        "requirement_resolution",
        "compliance_result",
        "compliance_summary",
    }
    replayed = await store.get_buffered_events(request_id)
    assert {event["type"] for event in replayed} >= {
        "compliance_result",
        "compliance_summary",
    }
    checkpoints = await store.get_checkpoint(request_id)
    assert {
        PipelineStep.CHECK_PLAN_VALIDATION,
        PipelineStep.REQUIREMENTS_RESOLUTION,
        PipelineStep.TEMPLATE_EXECUTION,
        PipelineStep.VERDICT_AGGREGATION,
    } <= set(checkpoints)

    service.compliance_executor.execute.reset_mock()
    resumed = [
        event
        async for event in service._run_executable_compliance(
            mcp_client=object(),
            request_id=request_id,
            scenario_id=772,
            restrictions=[],
            checkpoint=checkpoints,
        )
    ]
    assert resumed == []
    service.compliance_executor.execute.assert_not_awaited()
    await _clear(require_redis, request_id)
