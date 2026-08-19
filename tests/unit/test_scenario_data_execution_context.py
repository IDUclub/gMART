from unittest.mock import AsyncMock

import pytest

from src.agents.mcp_clients.urban_mcp_client import UrbanMcpTool
from src.agents.services.scenario_data_execution_context import (
    ScenarioExecutionContext,
)
from src.agents.services.scenario_data_plan_builder import ScenarioDataPlanBuilder
from src.agents.services.scenario_data_service import _is_transient_tool_error
from src.agents.services.service_entities.scenario_data_plan import (
    AcquisitionPlan,
    DataRequirement,
    MappingDirection,
    MappingNeed,
    PlanStep,
)


def _tool(name: str, *, properties: dict | None = None) -> UrbanMcpTool:
    return UrbanMcpTool(
        group="projects",
        name=name,
        title=name,
        description=name,
        input_schema={
            "type": "object",
            "properties": properties or {"scenario_id": {"type": "integer"}},
            "required": ["scenario_id"],
        },
        tags=("services",),
    )


def _acquisition() -> AcquisitionPlan:
    return AcquisitionPlan(
        objective="Показать все поликлиники выбранного сценария",
        requirements=[
            DataRequirement(
                requirement_id="clinics",
                description="Получить поликлиники сценария",
                mapping_needs=[
                    MappingNeed(
                        domain="service_type",
                        direction=MappingDirection.NAME_TO_ID,
                        values=["Поликлиника"],
                    )
                ],
            )
        ],
        required_output={
            "answer": True,
            "tables": ["clinics"],
            "layers": ["clinics_layer"],
        },
    )


def test_execution_context_keeps_task_mapping_step_and_failure():
    context = ScenarioExecutionContext.from_acquisition(
        _acquisition(),
        user_query="Покажи поликлиники",
        scenario_id=772,
        project_id=604,
    )
    context.update_mappings(
        [
            {
                "domain": "service_type",
                "source_tool": "dictionaries.GetServiceTypes",
                "matches": [{"id": 28, "name": "Поликлиника"}],
            }
        ]
    )
    step = PlanStep(
        step_id="clinics",
        purpose="Получить геометрию поликлиник",
        group="projects",
        tool_name="GetScenarioServicesWithGeometry",
        arguments={"scenario_id": 772, "service_type_id": 28},
        satisfies=["clinics"],
        expected_output="GeoJSON",
    )

    attempt = context.start_step(1, step, step.arguments)
    context.fail_attempt(attempt, TimeoutError("Urban MCP timed out"))
    snapshot = context.planner_snapshot(urban_calls=1, workspace_calls=0, replans=0)

    assert snapshot["task"]["objective"] == _acquisition().objective
    assert snapshot["task"]["scenario_id"] == 772
    assert snapshot["verified_mappings"][0]["matches"] == [
        {"id": 28, "name": "Поликлиника"}
    ]
    assert snapshot["attempts"][0]["tool_name"] == ("GetScenarioServicesWithGeometry")
    assert snapshot["attempts"][0]["arguments"]["service_type_id"] == 28
    assert snapshot["attempts"][0]["error_type"] == "TimeoutError"
    assert snapshot["open_requirements"] == ["clinics"]


def test_recovery_candidates_keep_semantic_alternative_in_catalogue():
    tools = [
        _tool("GetScenarioServicesWithGeometry"),
        _tool("GetScenarioServices"),
        *[_tool(f"GetUnrelatedIndicator{index}") for index in range(20)],
    ]
    execution_context = {
        "attempts": [
            {
                "status": "failed",
                "group": "projects",
                "tool_name": "GetScenarioServicesWithGeometry",
                "arguments": {"scenario_id": 772, "service_type_id": 28},
                "satisfies": ["clinics"],
                "error": "upstream timeout",
            }
        ]
    }

    shortlist = ScenarioDataPlanBuilder._shortlist(
        tools,
        "Получить данные сценария",
        [],
        execution_context=execution_context,
    )

    assert "GetScenarioServices" in {tool.name for tool in shortlist}


def test_deterministic_recovery_preserves_grounded_arguments():
    tools = [
        _tool("GetScenarioServicesWithGeometry"),
        _tool(
            "GetScenarioServices",
            properties={
                "scenario_id": {"type": "integer"},
                "service_type_id": {"type": "integer"},
            },
        ),
    ]
    execution_context = {
        "attempts": [
            {
                "status": "failed",
                "group": "projects",
                "tool_name": "GetScenarioServicesWithGeometry",
                "purpose": "Получить поликлиники",
                "arguments": {"scenario_id": 772, "service_type_id": 28},
                "satisfies": ["clinics"],
            }
        ]
    }

    steps = ScenarioDataPlanBuilder._recovery_seed_steps(
        _acquisition(),
        tools,
        scenario_id=772,
        project_id=604,
        execution_context=execution_context,
    )

    assert len(steps) == 1
    assert steps[0].tool_name == "GetScenarioServices"
    assert steps[0].arguments == {"scenario_id": 772, "service_type_id": 28}
    assert steps[0].satisfies == ["clinics"]


@pytest.mark.asyncio
async def test_recovery_does_not_upgrade_back_to_failed_geometry_tool():
    tools = [
        _tool("GetScenarioServicesWithGeometry"),
        _tool(
            "GetScenarioServices",
            properties={
                "scenario_id": {"type": "integer"},
                "service_type_id": {"type": "integer"},
            },
        ),
    ]
    execution_context = {
        "attempts": [
            {
                "status": "failed",
                "group": "projects",
                "tool_name": "GetScenarioServicesWithGeometry",
                "purpose": "Получить поликлиники",
                "arguments": {"scenario_id": 772, "service_type_id": 28},
                "satisfies": ["clinics"],
            }
        ]
    }
    builder = ScenarioDataPlanBuilder(object())
    builder._structured_plan_call = AsyncMock(side_effect=ValueError("bad replan"))

    plan = await builder.build_execution_plan(
        "model",
        "Покажи поликлиники",
        _acquisition(),
        tools,
        [{"domain": "service_type", "matches": [{"id": 28, "name": "Поликлиника"}]}],
        scenario_id=772,
        project_id=604,
        revision=2,
        execution_context=execution_context,
    )

    assert [step.tool_name for step in plan.steps] == ["GetScenarioServices"]


def test_failure_note_reports_attempts_before_honest_refusal():
    context = ScenarioExecutionContext.from_acquisition(
        _acquisition(),
        user_query="Покажи поликлиники",
        scenario_id=772,
        project_id=604,
    )
    step = PlanStep(
        step_id="clinics",
        purpose="Получить поликлиники",
        group="projects",
        tool_name="GetScenarioServices",
        arguments={"scenario_id": 772, "service_type_id": 28},
        satisfies=["clinics"],
        expected_output="records",
    )
    attempt = context.start_step(1, step, step.arguments)
    context.fail_attempt(attempt, RuntimeError("service unavailable"))

    note = context.failure_note(["не получен требуемый слой"])

    assert "GetScenarioServices" in note
    assert "service unavailable" in note
    assert "clinics" in note


def test_only_transient_tool_errors_are_retried():
    assert _is_transient_tool_error(TimeoutError("timed out")) is True
    assert _is_transient_tool_error(RuntimeError("HTTP 503 unavailable")) is True
    assert _is_transient_tool_error(RuntimeError("unknown argument type_id")) is False


@pytest.mark.asyncio
async def test_read_only_operation_retries_a_transient_failure():
    from src.agents.services.scenario_data_service import ScenarioDataService

    class StateStore:
        async def is_cancelled(self, request_id):
            return False

    class Owner:
        state_store = StateStore()

        @staticmethod
        def _status(status, text):
            return {"type": "status", "content": {"status": status, "text": text}}

    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary timeout")
        return {"ok": True}

    result = []
    events = [
        event
        async for event in ScenarioDataService._retryable_operation(
            Owner(),
            "request-1",
            object(),
            ["token"],
            operation,
            result,
            retry_transient=True,
        )
    ]

    assert calls == 2
    assert result == [{"ok": True}]
    assert events[0]["content"]["status"] == "tool_retry"


@pytest.mark.asyncio
async def test_transient_retry_is_opt_in_for_mutating_operations():
    from src.agents.services.scenario_data_service import ScenarioDataService

    class StateStore:
        async def is_cancelled(self, request_id):
            return False

    class Owner:
        state_store = StateStore()

    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        raise TimeoutError("result unknown after write timeout")

    with pytest.raises(TimeoutError):
        async for _ in ScenarioDataService._retryable_operation(
            Owner(),
            "request-1",
            object(),
            ["token"],
            operation,
            [],
        ):
            pass

    assert calls == 1
