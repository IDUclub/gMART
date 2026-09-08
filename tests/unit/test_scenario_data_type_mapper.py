from __future__ import annotations

import json

import pytest

from src.agents.mcp_clients.urban_mcp_client import UrbanMcpTool
from src.agents.services.scenario_data_mapping import MappingCall, UrbanMappingResolver
from src.agents.services.scenario_data_plan_builder import ScenarioDataPlanBuilder
from src.agents.services.scenario_data_type_mapper import (
    TypeMappingCandidate,
    TypeMappingRequest,
    TypeSearchPattern,
    TypeSearchPlan,
    UrbanTypeMapper,
    apply_verified_type_mappings,
    collect_type_mapping_candidates,
    pending_type_mapping_requests,
    verified_mapping_snapshots,
)
from src.agents.services.service_entities.scenario_data_plan import (
    AcquisitionPlan,
    DataRequirement,
    MappingDirection,
    MappingNeed,
)
from tests.helpers import FakeLlmClient


def _tool(name: str, domain: str, *, scenario_records: bool = False) -> UrbanMcpTool:
    properties = {"scenario_id": {"type": "integer"}} if scenario_records else {}
    if scenario_records:
        properties[f"{domain}_id"] = {"type": "integer"}
    return UrbanMcpTool(
        group="projects" if scenario_records else "dictionaries",
        name=name,
        title=name,
        description=f"Urban {domain} types",
        input_schema={
            "type": "object",
            "properties": properties,
            "required": ["scenario_id"] if scenario_records else [],
        },
        tags=(),
    )


def _acquisition(values: list[str]) -> AcquisitionPlan:
    return AcquisitionPlan(
        objective="Посчитать запрошенные объекты в сценарии",
        requirements=[
            DataRequirement(
                requirement_id="counts",
                description="Количество объектов каждого типа",
                mapping_needs=[
                    MappingNeed(
                        domain="physical_object_type",
                        direction=MappingDirection.NAME_TO_ID,
                        values=values,
                    )
                ],
            )
        ],
    )


@pytest.mark.asyncio
async def test_model_builds_one_safe_search_pattern_for_each_requested_type():
    llm = FakeLlmClient()
    llm.json_responses = [
        json.dumps(
            {
                "patterns": [
                    {
                        "requirement_id": "counts",
                        "requested_value": "школы",
                        "pattern": "школ|общеобразоват",
                    },
                    {
                        "requirement_id": "counts",
                        "requested_value": "детские сады",
                        "pattern": "детск.*сад|дошкольн",
                    },
                ]
            },
            ensure_ascii=False,
        )
    ]
    mapper = UrbanTypeMapper(llm)
    requests = [
        TypeMappingRequest(requirement_id="counts", requested_value="школы"),
        TypeMappingRequest(
            requirement_id="counts", requested_value="детские сады"
        ),
    ]

    plan = await mapper.build_search_plan(
        "model",
        "Сколько школ и детских садов?",
        _acquisition([]),
        requests,
    )

    assert [item.requested_value for item in plan.patterns] == [
        "школы",
        "детские сады",
    ]
    assert len(llm.chat_calls) == 1


def test_generated_patterns_are_applied_locally_to_both_type_domains():
    plan = TypeSearchPlan(
        patterns=[
            TypeSearchPattern(
                requirement_id="counts",
                requested_value="школы",
                pattern="школ|общеобразоват",
            )
        ]
    )
    physical_need = MappingNeed(
        domain="physical_object_type",
        direction=MappingDirection.NAME_TO_ID,
    )
    service_need = physical_need.model_copy(update={"domain": "service_type"})
    results = [
        (
            MappingCall(
                requirement_id="catalog",
                need=physical_need,
                tool=_tool("GetPhysicalObjectTypes", "physical_object_type"),
                arguments={},
                intent_text="",
            ),
            [
                {"physical_object_type_id": 11, "name": "Школа"},
                {"physical_object_type_id": 12, "name": "Больница"},
            ],
        ),
        (
            MappingCall(
                requirement_id="catalog",
                need=service_need,
                tool=_tool("GetServiceTypes", "service_type"),
                arguments={},
                intent_text="",
            ),
            [
                {
                    "service_type_id": 21,
                    "name": "Общеобразовательная школа",
                },
                {"service_type_id": 22, "name": "Аптека"},
            ],
        ),
    ]

    candidates = collect_type_mapping_candidates(plan, results)

    assert [(item.domain, item.type_id, item.name) for item in candidates] == [
        ("physical_object_type", 11, "Школа"),
        ("service_type", 21, "Общеобразовательная школа"),
    ]


@pytest.mark.asyncio
async def test_selection_supports_multiple_types_and_independent_assessment():
    llm = FakeLlmClient()
    llm.json_responses = [
        json.dumps(
            {"candidate_ids": ["candidate_1", "candidate_2"], "reason": "точно"}
        ),
        json.dumps(
            {
                "assessments": [
                    {
                        "candidate_id": "candidate_1",
                        "accepted": True,
                        "reason": "соответствует школе",
                    },
                    {
                        "candidate_id": "candidate_2",
                        "accepted": True,
                        "reason": "соответствует детскому саду",
                    },
                ]
            },
            ensure_ascii=False,
        ),
    ]
    candidates = [
        TypeMappingCandidate(
            candidate_id="candidate_1",
            requirement_id="counts",
            requested_value="школы",
            domain="service_type",
            type_id=21,
            name="Общеобразовательная школа",
            source_tool="dictionaries.GetServiceTypes",
        ),
        TypeMappingCandidate(
            candidate_id="candidate_2",
            requirement_id="counts",
            requested_value="детские сады",
            domain="physical_object_type",
            type_id=12,
            name="Детский сад",
            source_tool="dictionaries.GetPhysicalObjectTypes",
        ),
    ]
    requests = [
        TypeMappingRequest(requirement_id="counts", requested_value="школы"),
        TypeMappingRequest(
            requirement_id="counts", requested_value="детские сады"
        ),
    ]

    resolution = await UrbanTypeMapper(llm).resolve_candidates(
        "model",
        "Сколько школ и детских садов?",
        _acquisition([]),
        requests,
        candidates,
    )

    assert resolution.complete
    assert resolution.accepted == candidates
    assert len(llm.chat_calls) == 2


@pytest.mark.asyncio
async def test_rejected_candidate_is_reported_as_missing():
    llm = FakeLlmClient()
    llm.json_responses = [
        json.dumps({"candidate_ids": ["candidate_1"], "reason": "похож"}),
        json.dumps(
            {
                "assessments": [
                    {
                        "candidate_id": "candidate_1",
                        "accepted": False,
                        "reason": (
                            "это школа искусств, а не "
                            "общеобразовательная школа"
                        ),
                    }
                ]
            },
            ensure_ascii=False,
        ),
    ]
    candidate = TypeMappingCandidate(
        candidate_id="candidate_1",
        requirement_id="counts",
        requested_value="школы",
        domain="service_type",
        type_id=21,
        name="Школа искусств",
        source_tool="dictionaries.GetServiceTypes",
    )

    resolution = await UrbanTypeMapper(llm).resolve_candidates(
        "model",
        "Сколько общеобразовательных школ?",
        _acquisition(["школы"]),
        [TypeMappingRequest(requirement_id="counts", requested_value="школы")],
        [candidate],
    )

    assert not resolution.complete
    assert resolution.accepted == []
    assert resolution.missing_values == ["школы"]


def test_verified_candidates_replace_provisional_domains_and_seed_all_calls():
    acquisition = _acquisition(["школы", "детские сады"])
    accepted = [
        TypeMappingCandidate(
            candidate_id="candidate_1",
            requirement_id="counts",
            requested_value="школы",
            domain="service_type",
            type_id=21,
            name="Общеобразовательная школа",
            source_tool="dictionaries.GetServiceTypes",
        ),
        TypeMappingCandidate(
            candidate_id="candidate_2",
            requirement_id="counts",
            requested_value="детские сады",
            domain="physical_object_type",
            type_id=12,
            name="Детский сад",
            source_tool="dictionaries.GetPhysicalObjectTypes",
        ),
    ]
    acquisition = apply_verified_type_mappings(acquisition, accepted)
    mappings = verified_mapping_snapshots(accepted)
    tools = [
        _tool("GetScenarioServices", "service_type", scenario_records=True),
        _tool(
            "GetScenarioPhysicalObjects",
            "physical_object_type",
            scenario_records=True,
        ),
    ]

    steps = ScenarioDataPlanBuilder._scenario_seed_steps(
        acquisition, tools, 772, mappings
    )

    assert [(step.tool_name, step.arguments) for step in steps] == [
        ("GetScenarioServices", {"scenario_id": 772, "service_type_id": 21}),
        (
            "GetScenarioPhysicalObjects",
            {"scenario_id": 772, "physical_object_type_id": 12},
        ),
    ]


def test_type_catalog_planner_fetches_each_domain_without_name_filter():
    resolver = UrbanMappingResolver()
    tools = [
        _tool("GetPhysicalObjectTypes", "physical_object_type"),
        _tool("GetServiceTypes", "service_type"),
    ]

    calls = resolver.plan_type_catalog_calls(_acquisition(["школы"]), tools, 772)

    assert [(call.need.domain, call.tool.name, call.arguments) for call in calls] == [
        ("physical_object_type", "GetPhysicalObjectTypes", {}),
        ("service_type", "GetServiceTypes", {}),
    ]


def test_pending_requests_include_each_named_type():
    requests = pending_type_mapping_requests(
        _acquisition(["школы", "детские сады"]), []
    )

    assert [request.requested_value for request in requests] == [
        "школы",
        "детские сады",
    ]


def test_unsafe_generated_regex_is_rejected():
    with pytest.raises(ValueError, match="lookarounds"):
        TypeSearchPattern(
            requirement_id="counts",
            requested_value="школы",
            pattern="(?=школа).*",
        )
