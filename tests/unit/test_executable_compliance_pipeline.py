import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.schema.restrictions_response import RestrictionsResponse
from src.agents.services.pipeline_state import PipelineStep
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.service_entities.compliance import (
    ComplianceResult,
    ComplianceSummary,
    VerificationCoverage,
)


def _plan():
    return {
        "schema_version": "1.0",
        "template": "distance_from_source",
        "template_version": 1,
        "params": {
            "source_layer": "source",
            "targets": ["targets"],
            "geometry_mode": "buffered",
            "distance_m": 50,
            "predicate": "intersects",
            "violation_when": "matched",
            "result_mode": "both",
        },
        "declared_requirements": {"layers": [], "attributes": []},
        "source": {"restriction_id": "r1"},
        "planner_status": "auto",
    }


async def test_pipeline_emits_and_checkpoints_structured_compliance_events():
    service = object.__new__(RestrictionParserService)
    service.state_store = SimpleNamespace(
        buffer_event=AsyncMock(),
        save_checkpoint=AsyncMock(),
        set_status=AsyncMock(),
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
        source={"restriction_id": "r1"},
    )
    service.compliance_executor = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(
                result=result,
                tool_calls=[
                    {"function": {"name": "CheckDistanceFromSource", "arguments": {}}}
                ],
                timings_ms={"template_execution": 1.0},
            )
        )
    )

    events = [
        event
        async for event in service._run_executable_compliance(
            mcp_client=object(),
            request_id="request-1",
            scenario_id=772,
            restrictions=[{"id": "r1", "check_plan": _plan()}],
            checkpoint={},
        )
    ]
    await asyncio.sleep(0)
    event_types = [event["type"] for event in events]
    for event in events:
        RestrictionsResponse.model_validate(event)
    assert "check_plan" in event_types
    assert "requirement_resolution" in event_types
    assert "compliance_result" in event_types
    assert event_types.count("compliance_progress") == 2
    assert "compliance_summary" in event_types
    progress = [
        event["content"] for event in events if event["type"] == "compliance_progress"
    ]
    assert progress[0]["completed_norms"] == 0
    assert progress[-1]["completed_norms"] == progress[-1]["total_norms"] == 1
    checkpoints = [
        call.args[1] for call in service.state_store.save_checkpoint.await_args_list
    ]
    assert checkpoints == [
        PipelineStep.CHECK_PLAN_VALIDATION,
        PipelineStep.REQUIREMENTS_RESOLUTION,
        PipelineStep.TEMPLATE_EXECUTION,
        PipelineStep.VERDICT_AGGREGATION,
    ]


async def test_completed_reconnect_does_not_reexecute_templates():
    service = object.__new__(RestrictionParserService)
    service.state_store = SimpleNamespace(set_status=AsyncMock())
    service.compliance_executor = SimpleNamespace(execute=AsyncMock())
    events = [
        event
        async for event in service._run_executable_compliance(
            mcp_client=object(),
            request_id="request-1",
            scenario_id=772,
            restrictions=[],
            checkpoint={PipelineStep.VERDICT_AGGREGATION: {}},
        )
    ]
    assert events == []
    service.compliance_executor.execute.assert_not_awaited()


@pytest.mark.parametrize("persist_history", [True, False])
async def test_compliance_history_keeps_tool_calls_and_text_without_large_results(
    persist_history,
):
    service = object.__new__(RestrictionParserService)
    service.resolve_model = AsyncMock(return_value="model")
    service.add_complex_message = AsyncMock()
    oversized_result = {"evidence": "x" * (17 * 1024 * 1024)}
    events = [
        {
            "type": "status",
            "content": {"status": "template_execution", "text": "Проверяю нормы"},
        },
        service._tool_call(
            "normgraph_search",
            [
                {
                    "function": {
                        "name": "search_restrictions",
                        "arguments": {"query": "школы"},
                    }
                }
            ],
            "NORM_GRAPH_MCP_URL",
        ),
        service._tool_call(
            "template_execution",
            [
                {
                    "function": {
                        "name": "CheckDistanceFromSource",
                        "arguments": {"distance_m": 50},
                    }
                }
            ],
            "IDU_MCP_URL",
        ),
        *[
            {"type": event_type, "content": oversized_result}
            for event_type in (
                "check_plan",
                "requirement_resolution",
                "compliance_result",
                "compliance_summary",
            )
        ],
        {"type": "compliance_progress", "content": {"completed_norms": 1}},
        service._chunk("Проверка ", done=False),
        service._chunk("завершена.", done=True),
    ]

    async def run_inner(**kwargs):
        for event in events:
            yield event

    service._run_restriction_execution_pipline = run_inner
    streamed = [
        event
        async for event in service.run_compliance_pipeline(
            mcp_client=object(),
            token="user-token",
            temperature=0,
            model="model",
            user_query="Проверь нормы",
            scenario_id=845,
            normgraph_mcp_client=object(),
            chat_id="chat-1",
            persist_history=persist_history,
        )
    ]
    await asyncio.sleep(0)

    # Full structured results still reach the caller, including oversized evidence.
    assert streamed == [event for event in events if event["type"] != "tool_call"]
    if not persist_history:
        service.add_complex_message.assert_not_awaited()
        return

    service.add_complex_message.assert_awaited_once()
    call = service.add_complex_message.await_args
    assert call.args[:2] == ("user-token", "chat-1")
    assert call.kwargs == {"scenario_id": 845}
    parts = [part.model_dump(mode="json") for part in call.args[3]]
    assert [part["kind"] for part in parts] == ["tool_call", "tool_call", "text"]
    assert [part["mcp_source"] for part in parts[:2]] == [
        "NORM_GRAPH_MCP_URL",
        "IDU_MCP_URL",
    ]
    assert [part["payload"]["calls"][0] for part in parts[:2]] == [
        {
            "step": 1,
            "tool_name": "search_restrictions",
            "arguments": {"query": "школы"},
        },
        {
            "step": 1,
            "tool_name": "CheckDistanceFromSource",
            "arguments": {"distance_m": 50},
        },
    ]
    assert parts[-1]["payload"]["text"] == "Проверка завершена."
    assert len(json.dumps(parts).encode()) < 4096


def test_status_history_is_preserved_outside_compliance():
    event = {"type": "status", "content": {"status": "planning", "text": "План"}}
    part = RestrictionParserService._pipeline_item_to_chat_part(event)
    assert part.kind == "status"
    assert part.payload.text == "План"


def test_compliance_history_keeps_clarification_text():
    part = RestrictionParserService._pipeline_item_to_chat_part(
        {"type": "clarification", "content": {"question": "Какой объект проверить?"}},
        text_only=True,
    )
    assert part.kind == "text"
    assert part.payload.text == "Какой объект проверить?"
