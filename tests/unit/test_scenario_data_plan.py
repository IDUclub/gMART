import json

import pytest
from pydantic import ValidationError

from src.agents.mcp_clients.urban_mcp_client import UrbanMcpTool
from src.agents.services.scenario_data_linear import ScenarioDataLinearWorkflow
from src.agents.services.scenario_data_mapping import (
    UrbanMappingResolver,
    bind_mapping_arguments,
    context_mapping_snapshots,
    enrich_acquisition_mappings,
)
from src.agents.services.scenario_data_plan_builder import ScenarioDataPlanBuilder
from src.agents.services.service_entities.scenario_data_plan import (
    AcquisitionPlan,
    DataRequirement,
    ExecutionPlanRevision,
    MappingDirection,
    MappingNeed,
    PlanStep,
    PlanStepKind,
)


def test_execution_plan_rejects_forward_dependency():
    with pytest.raises(ValidationError):
        ExecutionPlanRevision(
            revision=1,
            reason="initial",
            objective="answer",
            steps=[
                PlanStep(
                    step_id="first",
                    purpose="first",
                    group="projects",
                    tool_name="GetOne",
                    depends_on=["second"],
                    expected_output="records",
                ),
                PlanStep(
                    step_id="second",
                    purpose="second",
                    group="dictionaries",
                    tool_name="GetTypes",
                    expected_output="types",
                ),
            ],
        )


def test_mapping_resolver_never_invents_required_arguments():
    resolver = UrbanMappingResolver()
    plan = AcquisitionPlan(
        objective="resolve types",
        requirements=[
            DataRequirement(
                requirement_id="types",
                description="service type names",
                mapping_needs=[
                    MappingNeed(
                        domain="service types",
                        direction=MappingDirection.ID_TO_NAME,
                        values=[1, 2],
                    )
                ],
            )
        ],
    )
    usable = UrbanMcpTool(
        group="dictionaries",
        name="GetServiceTypes",
        title="Типы сервисов",
        description="Справочник типов сервисов",
        input_schema={
            "type": "object",
            "properties": {"service_type_ids": {"type": "array"}},
            "required": ["service_type_ids"],
        },
        tags=(),
    )
    impossible = UrbanMcpTool(
        group="dictionaries",
        name="GetSecretTypes",
        title="Типы сервисов",
        description="requires an unknown filter",
        input_schema={
            "type": "object",
            "properties": {"unknown_filter": {"type": "string"}},
            "required": ["unknown_filter"],
        },
        tags=(),
    )

    calls = resolver.plan_calls(plan, [impossible, usable], 772)

    assert len(calls) == 1
    assert calls[0].tool.name == "GetServiceTypes"
    assert calls[0].arguments == {"service_type_ids": [1, 2]}


def test_mapping_resolver_injects_project_id_without_using_scenario_id():
    resolver = UrbanMappingResolver()
    plan = AcquisitionPlan(
        objective="resolve project service types",
        requirements=[
            DataRequirement(
                requirement_id="types",
                description="service type names",
                mapping_needs=[
                    MappingNeed(
                        domain="service_type",
                        direction=MappingDirection.NAME_TO_ID,
                        values=[],
                    )
                ],
            )
        ],
    )
    project_types = UrbanMcpTool(
        group="projects",
        name="GetProjectServiceTypes",
        title="Project service types",
        description="Service type dictionary for a project",
        input_schema={
            "type": "object",
            "properties": {"project_id": {"type": "integer"}},
            "required": ["project_id"],
        },
        tags=(),
    )

    calls = resolver.plan_calls(plan, [project_types], 772, project_id=604)

    assert len(calls) == 1
    assert calls[0].arguments == {"project_id": 604}


def test_unproven_named_type_is_resolved_against_both_urban_type_domains():
    plan = AcquisitionPlan(
        objective="Вывести школы на территории проекта",
        requirements=[
            DataRequirement(
                requirement_id="schools",
                description="Получить школы",
                mapping_needs=[
                    MappingNeed(
                        domain="physical_object_type",
                        direction=MappingDirection.NAME_TO_ID,
                        values=["school"],
                    )
                ],
            )
        ],
    )
    physical_types = UrbanMcpTool(
        group="dictionaries",
        name="GetPhysicalObjectTypes",
        title="Physical object types",
        description="Physical object type dictionary",
        input_schema={"type": "object", "properties": {}},
        tags=(),
    )
    service_types = UrbanMcpTool(
        group="dictionaries",
        name="GetServiceTypes",
        title="Service types",
        description="Service type dictionary",
        input_schema={"type": "object", "properties": {}},
        tags=(),
    )

    calls = UrbanMappingResolver().plan_calls(
        plan, [physical_types, service_types], 772, project_id=604
    )

    assert [(call.need.domain, call.tool.name, call.arguments) for call in calls] == [
        ("physical_object_type", "GetPhysicalObjectTypes", {}),
        ("service_type", "GetServiceTypes", {}),
    ]


def test_mapping_resolver_prefers_matching_dictionary_for_empty_mapping_values():
    resolver = UrbanMappingResolver()
    plan = AcquisitionPlan(
        objective="resolve service types",
        requirements=[
            DataRequirement(
                requirement_id="types",
                description="service type names and identifiers",
                mapping_needs=[
                    MappingNeed(
                        domain="service_type",
                        direction=MappingDirection.ID_TO_NAME,
                        values=[],
                    )
                ],
            )
        ],
    )
    service_types = UrbanMcpTool(
        group="dictionaries",
        name="GetServiceTypes",
        title="Типы сервисов",
        description="Справочник типов сервисов",
        input_schema={"type": "object", "properties": {}},
        tags=(),
    )
    unrelated = UrbanMcpTool(
        group="dictionaries",
        name="GetDefaultBufferValues",
        title="Нормативные значения радиусов зон ограничений",
        description="Буферы физических объектов",
        input_schema={"type": "object", "properties": {}},
        tags=(),
    )

    calls = resolver.plan_calls(plan, [unrelated, service_types], 772)

    assert len(calls) == 1
    assert calls[0].tool.name == "GetServiceTypes"


def test_mapping_resolver_never_falls_back_to_unrelated_normative_dictionary():
    resolver = UrbanMappingResolver()
    plan = AcquisitionPlan(
        objective="resolve service types",
        requirements=[
            DataRequirement(
                requirement_id="types",
                description="service type names",
                mapping_needs=[
                    MappingNeed(
                        domain="service_type",
                        direction=MappingDirection.ID_TO_NAME,
                        values=[],
                    )
                ],
            )
        ],
    )
    normative_buffers = UrbanMcpTool(
        group="dictionaries",
        name="GetDefaultBufferValues",
        title="Нормативные значения радиусов зон ограничений",
        description="Буферы для типов физических объектов",
        input_schema={"type": "object", "properties": {}},
        tags=(),
    )

    assert resolver.plan_calls(plan, [normative_buffers], 772) == []


def test_chat_context_mapping_keeps_domain_name_and_id():
    snapshots = context_mapping_snapshots(
        {
            "content": {
                "structured": {
                    "mappings": [
                        {"service_types": {"22": "Школа", "21": "Детский сад"}},
                        {"physical_object_types": {"4": "Жилой дом"}},
                    ]
                }
            }
        }
    )

    assert snapshots == [
        {
            "domain": "physical_object_type",
            "direction": "name_to_id",
            "requested_values": [],
            "source_tool": "chat_context",
            "matches": [{"id": 4, "name": "Жилой дом"}],
        },
        {
            "domain": "service_type",
            "direction": "name_to_id",
            "requested_values": [],
            "source_tool": "chat_context",
            "matches": [
                {"id": 22, "name": "Школа"},
                {"id": 21, "name": "Детский сад"},
            ],
        },
    ]


def test_known_school_mapping_restores_domain_need_and_avoids_dictionary_call():
    snapshots = [
        {
            "domain": "service_type",
            "direction": "name_to_id",
            "matches": [{"id": 22, "name": "Школа"}],
        }
    ]
    plan = AcquisitionPlan(
        objective="Вывести школы на территории проекта",
        requirements=[
            DataRequirement(
                requirement_id="schools",
                description="Показать все школы на территории проекта (type_id=22)",
            )
        ],
    )

    enriched = enrich_acquisition_mappings(
        plan, "Выведи школы на территории проекта", snapshots
    )

    assert enriched.requirements[0].mapping_needs == [
        MappingNeed(
            domain="service_type",
            direction=MappingDirection.NAME_TO_ID,
            values=["Школа"],
        )
    ]
    assert (
        UrbanMappingResolver().plan_calls(
            enriched, [], 772, project_id=604, known_mappings=snapshots
        )
        == []
    )


def test_verified_school_mapping_corrects_an_unproven_physical_type_domain():
    plan = AcquisitionPlan(
        objective="Вывести школы на территории проекта",
        requirements=[
            DataRequirement(
                requirement_id="schools",
                description="Получить школы",
                mapping_needs=[
                    MappingNeed(
                        domain="physical_object_type",
                        direction=MappingDirection.NAME_TO_ID,
                        values=["school"],
                    )
                ],
            )
        ],
    )
    mappings = [
        {
            "domain": "physical_object_type",
            "matches": [{"id": 5, "name": "Нежилое здание"}],
        },
        {
            "domain": "service_type",
            "matches": [{"id": 22, "name": "Школа"}],
        },
    ]

    enriched = enrich_acquisition_mappings(
        plan, "Выведи все школы на территории проекта", mappings
    )

    assert enriched.requirements[0].mapping_needs == [
        MappingNeed(
            domain="service_type",
            direction=MappingDirection.NAME_TO_ID,
            values=["Школа"],
        )
    ]


def test_verified_mapping_binds_only_its_domain_specific_id_argument():
    tool = UrbanMcpTool(
        group="projects",
        name="GetScenarioServices",
        title="Сервисы сценария",
        description="Получить сервисы выбранного типа",
        input_schema={
            "type": "object",
            "properties": {
                "scenario_id": {"type": "integer"},
                "service_type_id": {"type": "integer"},
            },
            "required": ["scenario_id"],
        },
        tags=(),
    )
    snapshots = [
        {
            "domain": "service_type",
            "matches": [{"id": 22, "name": "Школа"}],
        },
        {
            "domain": "physical_object_type",
            "matches": [{"id": 5, "name": "Нежилое здание"}],
        },
    ]

    arguments = bind_mapping_arguments(
        tool, {}, snapshots, "Выведи школы на территории проекта"
    )

    assert arguments == {"service_type_id": 22}
    assert bind_mapping_arguments(
        tool,
        {"service_type_id": 999},
        snapshots,
        "Выведи школы на территории проекта",
    ) == {"service_type_id": 22}


@pytest.mark.asyncio
async def test_acquisition_rejects_unrequested_normative_topic_drift():
    calls = 0

    class DriftingLlm:
        async def chat(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                payload = AcquisitionPlan(
                    objective="Получить нормативные зоны ограничений",
                    requirements=[
                        DataRequirement(
                            requirement_id="buffers",
                            description="Нормативные буферные зоны",
                        )
                    ],
                )
            else:
                assert "without support" in kwargs["messages"][-1]["content"]
                payload = AcquisitionPlan(
                    objective="Посчитать физические объекты и сервисы",
                    requirements=[
                        DataRequirement(
                            requirement_id="entities",
                            description="Физические объекты и сервисы",
                        )
                    ],
                )
            return {"message": {"content": payload.model_dump_json()}}

    result = await ScenarioDataPlanBuilder(DriftingLlm()).build_acquisition_plan(
        "model",
        "Оба набора",
        [
            {"role": "user", "content": "Что посчитать в сценарии?"},
            {
                "role": "assistant",
                "content": (
                    "Ошибочно предлагаю нормативные зоны ограничений. "
                    "Физические объекты, сервисы или оба набора?"
                ),
            },
        ],
        772,
    )

    assert calls == 2
    assert result.objective == "Посчитать физические объекты и сервисы"


@pytest.mark.asyncio
async def test_execution_shortlist_uses_intent_resolved_from_history():
    scenario_tool = UrbanMcpTool(
        group="projects",
        name="GetScenarioServiceTypes",
        title="Типы сервисов сценария",
        description="Типы сервисов, доступные в сценарии",
        input_schema={
            "type": "object",
            "properties": {"scenario_id": {"type": "integer"}},
            "required": ["scenario_id"],
        },
        tags=(),
    )
    fillers = [
        UrbanMcpTool(
            group="projects",
            name=f"GetA{index:02d}",
            title=f"Прочие данные {index}",
            description="Несвязанный набор",
            input_schema={"type": "object", "properties": {}},
            tags=(),
        )
        for index in range(12)
    ]

    class InspectingLlm:
        async def chat(self, **kwargs):
            prompt = kwargs["messages"][0]["content"]
            assert "GetScenarioServiceTypes" in prompt
            assert "Проект сценария: 604" in prompt
            assert "service_type.id" in prompt
            assert "scenario_id" in prompt and "project_id" in prompt
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "revision": 1,
                            "reason": "initial",
                            "objective": "scenario service types",
                            "steps": [
                                {
                                    "step_id": "types",
                                    "kind": "urban_tool",
                                    "purpose": "scenario service types",
                                    "group": "projects",
                                    "tool_name": "GetScenarioServiceTypes",
                                    "arguments": {"scenario_id": 772},
                                    "satisfies": ["types"],
                                    "expected_output": "records",
                                }
                            ],
                        }
                    )
                }
            }

    acquisition = AcquisitionPlan(
        objective="Получить типы сервисов текущего сценария",
        requirements=[
            DataRequirement(
                requirement_id="types",
                description="Названия и ID типов сервисов",
                mapping_needs=[
                    MappingNeed(
                        domain="service_type",
                        direction=MappingDirection.ID_TO_NAME,
                    )
                ],
            )
        ],
    )

    result = await ScenarioDataPlanBuilder(InspectingLlm()).build_execution_plan(
        "model",
        "Да, оба",
        acquisition,
        [*fillers, scenario_tool],
        [],
        scenario_id=772,
        project_id=604,
    )

    assert result.steps[0].tool_name == "GetScenarioServiceTypes"


@pytest.mark.asyncio
async def test_invalid_llm_execution_plan_falls_back_to_scenario_service_query():
    class InvalidLlm:
        async def chat(self, **kwargs):
            return {"message": {"content": "not valid json"}}

    builder = ScenarioDataPlanBuilder(InvalidLlm())
    acquisition = AcquisitionPlan(
        objective="Вывести школы на территории проекта",
        requirements=[
            DataRequirement(
                requirement_id="schools",
                description="Все сервисы типа Школа на территории проекта",
                mapping_needs=[
                    MappingNeed(
                        domain="service_type",
                        direction=MappingDirection.NAME_TO_ID,
                        values=["Школа"],
                    )
                ],
            )
        ],
        required_output={"answer": True, "layers": ["schools_layer"]},
    )
    scenario_services = UrbanMcpTool(
        group="projects",
        name="GetScenarioServicesWithGeometry",
        title="Сервисы сценария с геометрией",
        description="Сервисы в выбранном сценарии с геометрией",
        input_schema={
            "type": "object",
            "properties": {
                "scenario_id": {"type": "integer"},
                "service_type_id": {"type": "integer"},
            },
            "required": ["scenario_id"],
        },
        tags=(),
    )

    result = await builder.build_execution_plan(
        "model",
        "Выведи все школы на территории проекта",
        acquisition,
        [scenario_services],
        [{"domain": "service_type", "matches": [{"id": 22, "name": "Школа"}]}],
        scenario_id=772,
        project_id=604,
    )

    assert result.steps[0].tool_name == "GetScenarioServicesWithGeometry"
    assert result.steps[0].arguments == {"scenario_id": 772}
    assert result.required_output.layers == ["schools_layer"]


def test_required_layer_cannot_validate_without_observed_geometry():
    plan = ExecutionPlanRevision(
        revision=1,
        reason="test",
        objective="show schools",
        steps=[],
        required_output={"answer": True, "layers": ["schools_layer"]},
    )

    assert ScenarioDataLinearWorkflow._required_output_reasons(
        plan, [{"layer_count": 0}]
    ) == [
        "требуется географический слой, но ни один выполненный шаг не вернул геометрию"
    ]
    assert (
        ScenarioDataLinearWorkflow._required_output_reasons(plan, [{"layer_count": 1}])
        == []
    )


@pytest.mark.asyncio
async def test_required_layer_upgrades_a_valid_plan_to_geometry_tool():
    class NonGeometryLlm:
        async def chat(self, **kwargs):
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "revision": 1,
                            "reason": "initial",
                            "objective": "show schools",
                            "steps": [
                                {
                                    "step_id": "schools",
                                    "kind": "urban_tool",
                                    "purpose": "show schools",
                                    "group": "projects",
                                    "tool_name": "GetScenarioServices",
                                    "arguments": {"scenario_id": 772},
                                    "satisfies": ["schools"],
                                    "expected_output": "school features",
                                }
                            ],
                            "required_output": {
                                "answer": True,
                                "layers": ["schools_layer"],
                            },
                        }
                    )
                }
            }

    def scenario_services(name: str) -> UrbanMcpTool:
        return UrbanMcpTool(
            group="projects",
            name=name,
            title=name,
            description=name,
            input_schema={
                "type": "object",
                "properties": {
                    "scenario_id": {"type": "integer"},
                    "service_type_id": {"type": "integer"},
                },
                "required": ["scenario_id"],
            },
            tags=(),
        )

    result = await ScenarioDataPlanBuilder(NonGeometryLlm()).build_execution_plan(
        "model",
        "Покажи школы на карте",
        AcquisitionPlan(
            objective="Показать школы на карте проекта",
            requirements=[
                DataRequirement(
                    requirement_id="schools",
                    description="Школы проекта",
                    mapping_needs=[
                        MappingNeed(
                            domain="service_type",
                            direction=MappingDirection.NAME_TO_ID,
                            values=["Школа"],
                        )
                    ],
                )
            ],
            required_output={"answer": True, "layers": ["schools_layer"]},
        ),
        [
            scenario_services("GetScenarioServices"),
            scenario_services("GetScenarioServicesWithGeometry"),
        ],
        [{"domain": "service_type", "matches": [{"id": 22, "name": "Школа"}]}],
        scenario_id=772,
        project_id=604,
    )

    assert result.steps[0].tool_name == "GetScenarioServicesWithGeometry"


@pytest.mark.asyncio
async def test_initial_scenario_plan_repairs_global_mapping_into_scenario_call():
    payload = {
        "revision": 1,
        "reason": "initial plan",
        "objective": "scenario service types",
        "steps": [
            {
                "step_id": "select",
                "kind": "workspace",
                "purpose": "select mapping rows",
                "tool_name": "WorkspaceSelect",
                "arguments": {
                    "handle": "$artifact:mapping_1",
                    "columns": ["service_type_id", "name"],
                },
                "depends_on": ["mapping_1"],
                "satisfies": ["types"],
                "expected_output": "service types",
            }
        ],
    }

    class WorkspaceOnlyLlm:
        async def chat(self, **kwargs):
            return {"message": {"content": json.dumps(payload)}}

    builder = ScenarioDataPlanBuilder(WorkspaceOnlyLlm())
    acquisition = AcquisitionPlan(
        objective="service types available in scenario",
        requirements=[
            DataRequirement(requirement_id="types", description="scenario types")
        ],
    )
    scenario_tool = UrbanMcpTool(
        group="projects",
        name="GetScenarioServiceTypes",
        title="Типы сервисов сценария",
        description="Типы сервисов, доступные в сценарии",
        input_schema={
            "type": "object",
            "properties": {"scenario_id": {"type": "integer"}},
            "required": ["scenario_id"],
        },
        tags=(),
    )

    repaired = await builder.build_execution_plan(
        "model",
        "Типы сервисов в сценарии",
        acquisition,
        [scenario_tool],
        [],
        scenario_id=772,
        completed_step_ids=["mapping_1"],
        workspace_enabled=True,
    )

    assert [step.tool_name for step in repaired.steps] == ["GetScenarioServiceTypes"]
    assert repaired.steps[0].arguments == {"scenario_id": 772}


def test_workspace_plan_requires_declared_artifact_dependency():
    plan = ExecutionPlanRevision(
        revision=1,
        reason="initial",
        objective="aggregate",
        steps=[
            PlanStep(
                step_id="entities",
                purpose="entities",
                group="projects",
                tool_name="GetEntities",
                expected_output="records",
            ),
            PlanStep(
                step_id="aggregate",
                kind=PlanStepKind.WORKSPACE,
                purpose="count",
                tool_name="WorkspaceAggregate",
                arguments={
                    "handle": "$artifact:entities",
                    "group_by": ["type_id"],
                    "aggregations": [
                        {
                            "column": "id",
                            "function": "nunique",
                            "output_column": "count",
                        }
                    ],
                },
                depends_on=[],
                expected_output="counts",
            ),
        ],
    )
    tool = UrbanMcpTool(
        group="projects",
        name="GetEntities",
        title="Entities",
        description="Entities",
        input_schema={"type": "object", "properties": {}},
        tags=(),
    )

    with pytest.raises(ValueError, match="must depend_on"):
        ScenarioDataPlanBuilder._canonicalize_plan(plan, [tool], workspace_enabled=True)


def test_execution_plan_normalization_drops_group_from_workspace_steps():
    payload = {
        "revision": 1,
        "reason": "initial",
        "objective": "aggregate",
        "steps": [
            {
                "step_id": "aggregate",
                "kind": "workspace",
                "purpose": "count",
                "group": "projects",
                "tool_name": "WorkspaceAggregate",
                "arguments": {},
                "expected_output": "counts",
            }
        ],
    }

    normalized = ScenarioDataPlanBuilder._normalize_execution_plan_payload(payload)

    assert normalized["steps"][0]["group"] is None
    assert payload["steps"][0]["group"] == "projects"


def test_execution_plan_normalization_restores_remote_tool_kind():
    payload = {
        "revision": 1,
        "reason": "initial",
        "objective": "load service types",
        "steps": [
            {
                "step_id": "types",
                "kind": "workspace",
                "purpose": "load types",
                "group": "projects",
                "tool_name": "GetScenarioServiceTypes",
                "arguments": {"scenario_id": 772},
                "expected_output": "types",
            }
        ],
    }

    normalized = ScenarioDataPlanBuilder._normalize_execution_plan_payload(payload)

    assert normalized["steps"][0]["kind"] == "urban_tool"
    assert normalized["steps"][0]["group"] == "projects"


def test_execution_plan_normalization_splits_qualified_remote_tool_name():
    payload = {
        "steps": [
            {
                "step_id": "types",
                "kind": "workspace",
                "group": None,
                "tool_name": "projects.GetScenarioServiceTypes",
            }
        ]
    }

    normalized = ScenarioDataPlanBuilder._normalize_execution_plan_payload(payload)

    assert normalized["steps"][0]["kind"] == "urban_tool"
    assert normalized["steps"][0]["group"] == "projects"
    assert normalized["steps"][0]["tool_name"] == "GetScenarioServiceTypes"


def test_canonical_plan_recovers_unique_runtime_group():
    tool = UrbanMcpTool(
        group="projects",
        name="GetScenarioServiceTypes",
        title="Service types",
        description="Service types",
        input_schema={"type": "object", "properties": {}},
        tags=(),
    )
    plan = ExecutionPlanRevision(
        revision=1,
        reason="initial",
        objective="service types",
        steps=[
            PlanStep(
                step_id="types",
                purpose="load types",
                group="dictionaries",
                tool_name="GetScenarioServiceTypes",
                expected_output="types",
            )
        ],
    )

    canonical = ScenarioDataPlanBuilder._canonicalize_plan(plan, [tool])

    assert canonical.steps[0].group == "projects"


def test_workspace_argument_resolution_reports_missing_artifact():
    with pytest.raises(ValueError, match="artifact шага entities ещё не создан"):
        ScenarioDataLinearWorkflow._resolve_workspace_arguments(
            {"handle": "$artifact:entities"}, {}
        )


def test_canonical_plan_fills_select_arguments_from_dependency_and_output():
    tool = UrbanMcpTool(
        group="projects",
        name="GetScenarioServiceTypes",
        title="Service types",
        description="Service types",
        input_schema={"type": "object", "properties": {}},
        tags=(),
    )
    plan = ExecutionPlanRevision(
        revision=1,
        reason="initial",
        objective="service types",
        required_output={"fields": ["service_type_id", "name"]},
        steps=[
            PlanStep(
                step_id="types",
                purpose="load types",
                group="projects",
                tool_name="GetScenarioServiceTypes",
                expected_output="types",
            ),
            PlanStep(
                step_id="select",
                kind=PlanStepKind.WORKSPACE,
                purpose="select fields",
                tool_name="WorkspaceSelect",
                depends_on=["types"],
                expected_output="selected types",
            ),
        ],
    )

    canonical = ScenarioDataPlanBuilder._canonicalize_plan(
        plan, [tool], workspace_enabled=True
    )

    assert canonical.steps[1].arguments == {
        "handle": "$artifact:types",
        "columns": ["service_type_id", "name"],
    }
