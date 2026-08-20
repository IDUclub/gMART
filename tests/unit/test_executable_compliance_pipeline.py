import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.agents.schema.restrictions_response import RestrictionsResponse
from src.agents.services.pipeline_state import PipelineStep
from src.agents.services.restriction_parser_service import RestrictionParserService
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
    assert "compliance_summary" in event_types
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
