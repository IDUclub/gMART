import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.services.orchestrator.orchestrator_catalog import available_agents
from src.agents.services.planning.a2a import PlanningA2AService, PlanningAgentCard
from src.agents.services.planning.artifacts import (
    preview,
    resolve_references,
    result_events,
)
from src.agents.services.planning.planning_service import (
    PlanningAction,
    PlanningService,
)
from src.agents.services.planning.profiles import PROFILES
from src.common.service_auth import internal_user_context_jwt

LAYER = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [28.8, 61.1]},
            "properties": {"residents": 4500},
        }
    ],
}


def test_reference_preserves_full_geometry_and_does_not_alias():
    values = {"a": {"feature_collection": LAYER}}
    result = resolve_references(
        {"blocks": {"$artifact": "a", "path": ["feature_collection"]}}, values
    )
    assert result["blocks"] == LAYER
    result["blocks"]["features"].clear()
    assert len(LAYER["features"]) == 1
    with pytest.raises(ValueError):
        resolve_references({"$artifact": "foreign"}, values)


def test_preview_is_explicitly_partial_but_output_is_complete():
    assert preview(LAYER)["geometry_omitted"]
    events = list(result_events(LAYER, "buildings"))
    assert events[0]["content"]["feature_collection"] == LAYER
    assert events[1]["content"]["rows"][0]["residents"] == 4500


def test_inspection_reaches_catalogue_entries_beyond_preview():
    from src.agents.services.planning.artifacts import inspect_value

    catalogue = [{"id": i, "name": f"zone {i}"} for i in range(20)]
    assert not preview(catalogue)["complete"]
    page = inspect_value(catalogue, offset=6)
    assert [item["id"] for item in page["items"]] == list(range(6, 12))
    assert page["total"] == 20 and not page["complete"]
    assert inspect_value(catalogue, offset=18)["complete"]
    page["items"][0]["id"] = -1
    assert catalogue[6]["id"] == 6


@pytest.mark.parametrize("key", PROFILES)
def test_independent_cards_and_optional_catalogue(key):
    card = PlanningAgentCard(PROFILES[key]).get_agent_card("https://gmart.test")
    assert card["url"] == f"https://gmart.test/{key}/a2a"
    assert card["name"] == key + "-agent"
    config = SimpleNamespace(
        DVD_MCP_URL=None, NORM_GRAPH_MCP_URL=None, URBAN_MCP_URL=None
    )
    assert key not in {a.key for a in available_agents(config, 772)}
    setattr(config, key.upper() + "_MCP_URL", "https://mcp.test/mcp")
    assert key in {a.key for a in available_agents(config, 772)}


@pytest.fixture
def specialist(monkeypatch):
    service = object.__new__(PlanningService)
    service.profile = PROFILES["genbuilder"]
    service.mcp_url = "https://builder.test/mcp"
    service.pzz_api_url = None
    service.service_auth = object()
    service.llm_client = object()
    service.resolve_model = AsyncMock(return_value="model")
    transport = SimpleNamespace(
        load_ollama_tools=AsyncMock(
            return_value=[
                {
                    "function": {
                        "name": "generate_by_territory",
                        "parameters": {
                            "type": "object",
                            "required": ["blocks"],
                            "properties": {
                                "blocks": {"type": "object"},
                                "targets_by_zone": {"type": "object"},
                            },
                        },
                    }
                }
            ]
        ),
        execute_tool=AsyncMock(return_value=LAYER),
    )
    monkeypatch.setattr(
        "src.agents.services.planning.planning_service.service_mcp_client", AsyncMock()
    )
    monkeypatch.setattr(
        "src.agents.services.planning.planning_service.BaseMcpClient",
        lambda _: transport,
    )
    return service, transport


async def test_model_cannot_complete_without_actual_computation(
    specialist, monkeypatch
):
    service, transport = specialist
    actions = AsyncMock(
        side_effect=[
            PlanningAction(
                action="complete", answer="Всё готово", evidence_ids=["input"]
            ),
            PlanningAction(action="blocked", answer="Нужна реальная генерация"),
        ]
    )
    monkeypatch.setattr(
        "src.agents.services.planning.planning_service.run_structured", actions
    )
    events = [
        e
        async for e in service.run(
            token=internal_user_context_jwt("u"),
            user_query="Построй",
            input_artifacts={"input": LAYER},
        )
    ]
    assert events[-1]["type"] == "error"
    transport.execute_tool.assert_not_called()


async def test_generation_uses_lossless_reference_and_explicit_target(
    specialist, monkeypatch
):
    service, transport = specialist
    actions = AsyncMock(
        side_effect=[
            PlanningAction(
                action="call",
                tool="generate_by_territory",
                arguments_json=json.dumps(
                    {
                        "blocks": {"$artifact": "input"},
                        "targets_by_zone": {"residents": {"residential": 4500}},
                    }
                ),
            ),
            PlanningAction(
                action="complete",
                answer="Получена застройка",
                evidence_ids=["run:result1"],
            ),
        ]
    )
    monkeypatch.setattr(
        "src.agents.services.planning.planning_service.run_structured", actions
    )
    events = [
        e
        async for e in service.run(
            token=internal_user_context_jwt("u"),
            user_query="4500 жителей",
            request_id="run",
            input_artifacts={"input": LAYER},
        )
    ]
    assert events[-1]["type"] == "chunk"
    assert transport.execute_tool.call_args.args[1]["blocks"] == LAYER
    assert any(e["type"] == "feature_collection" for e in events)


async def test_a2a_clarification_is_input_required():
    async def run(**kwargs):
        yield {"type": "clarification", "content": {"question": "Сколько жителей?"}}

    service = SimpleNamespace(profile=PROFILES["genbuilder"], run=run)
    a2a = PlanningA2AService(service)
    result = await a2a.handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Построй здания"}],
                }
            },
        },
        None,
        "token",
    )
    assert result["result"]["status"]["state"] == "input-required"


def test_all_features_are_selectable_beyond_preview():
    from src.agents.services.planning.artifacts import (
        layer_values,
        select_layer,
        summarize_layer,
    )

    layer = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": LAYER["features"][0]["geometry"],
                "properties": {
                    "functional_zone_id": i,
                    "functional_zone_type": {"name": "recreation"},
                    "residents_number": i,
                },
            }
            for i in range(68)
        ],
    }
    selected = select_layer(layer, ["functional_zone_type", "name"], ["recreation"])
    ids = layer_values(selected, ["functional_zone_id"])
    assert ids == list(range(68))
    assert resolve_references({"$artifact": "ids"}, {"ids": ids}) == ids
    assert summarize_layer(layer, ["residents_number"])["statistics"][0] == {
        "property": "residents_number",
        "sum": sum(range(68)),
        "known": 68,
        "missing_or_invalid": 0,
    }


def test_building_blocks_preserve_coordinates_and_select_kinds():
    from src.agents.services.planning.artifacts import prepare_building_blocks

    layer = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                },
                "properties": {"territory_zone_name": kind},
            }
            for kind in ["residential", "recreation"]
        ],
    }
    blocks = prepare_building_blocks(layer, ["residential"])
    assert len(blocks["features"]) == 1
    assert blocks["features"][0]["properties"]["zone"] == "residential"
    assert blocks["features"][0]["geometry"] == layer["features"][0]["geometry"]
    assert "zone" not in layer["features"][0]["properties"]


@pytest.mark.parametrize(
    "tool,result",
    [
        ("generate_by_territory", {"type": "FeatureCollection", "features": []}),
        ("generate_by_territory", {"error": "upstream unavailable"}),
        ("run_func_generation", {"zones": LAYER}),
        ("get_task_report", {"ready": False}),
        ("classify_scenario_and_wait", {"timed_out": True}),
        ("get_task_report", {"status": "failed"}),
        ("get_task_report", {"action": "detection_failed"}),
        ("estimate_max_residents_by_blocks", {"1": "unknown"}),
        ("list_zone_types", {"1": "residential"}),
    ],
)
def test_pending_empty_or_failed_results_are_not_success(tool, result):
    from src.agents.services.planning.planning_service import completed_domain_operation

    assert not completed_domain_operation(tool, result)


def test_schema_compaction_preserves_parameters_named_description():
    from src.agents.services.planning.planning_service import compact_schema

    source = {
        "type": "object",
        "description": "long",
        "properties": {
            "description": {"type": "string", "description": "help"},
            "title": {"type": "object", "default": {"description": "value"}},
        },
    }
    result = compact_schema(source)
    assert result["properties"]["description"] == {"type": "string"}
    assert result["properties"]["title"]["default"] == {"description": "value"}


async def test_model_cannot_send_invented_geometry(specialist, monkeypatch):
    service, transport = specialist
    monkeypatch.setattr(
        "src.agents.services.planning.planning_service.run_structured",
        AsyncMock(
            side_effect=[
                PlanningAction(
                    action="call",
                    tool="generate_by_territory",
                    arguments_json=json.dumps(
                        {
                            "blocks": LAYER,
                            "targets_by_zone": {"residents": {"residential": 4500}},
                        }
                    ),
                ),
                PlanningAction(action="blocked", answer="Нет исходной геометрии"),
            ]
        ),
    )
    events = [
        e
        async for e in service.run(
            token=internal_user_context_jwt("u"), user_query="Построй"
        )
    ]
    transport.execute_tool.assert_not_called()
    assert events[-1]["type"] == "error"


async def test_compliance_checks_new_geometry_and_preserves_baseline():
    from src.agents.services.compilance.compliance_executor import (
        ComplianceTemplateExecutor,
    )
    from src.agents.services.planning.variant_compliance import variant_layers
    from tests.unit.test_compliance_executor import FakeMcpClient, _feature, _plan

    class Client(FakeMcpClient):
        async def resolve_urban_entity_types(self, **kwargs):
            result = await super().resolve_urban_entity_types(**kwargs)
            for info in result["physical_object"].values():
                info["type_id"] = 4
            return result

    client = Client()
    generated = {
        "type": "FeatureCollection",
        "features": [
            _feature(30.0002, zone="residential", residents_number=100),
            _feature(30.0003, zone="residential", is_excluded=True),
        ],
    }
    scope = variant_layers.set({"buildings": generated})
    try:
        execution = await ComplianceTemplateExecutor().execute(client, _plan(), 772)
    finally:
        variant_layers.reset(scope)
    assert execution.result.verification_status == "complete"
    assert execution.result.summary.violated_objects == 2
    assert execution.result.summary.passed_objects == 1
    baseline = await ComplianceTemplateExecutor().execute(Client(), _plan(), 772)
    assert baseline.result.summary.violated_objects == 1


def test_candidate_service_is_inside_supplied_site_and_identified_as_proposal():
    from shapely.geometry import shape

    from src.agents.services.planning.artifacts import propose_service

    layer = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                },
                "properties": {"id": 42},
            }
        ],
    }
    result = propose_service(layer, 22, 550)
    f = result["features"][0]
    assert shape(layer["features"][0]["geometry"]).covers(shape(f["geometry"]))
    assert f["properties"]["capacity"] == 550
    assert f["properties"]["design_status"] == "candidate_location"


@pytest.mark.parametrize(
    "tool,result",
    [
        ("get_task_report", {"message": "done"}),
        ("estimate_max_residents_by_blocks", {"1": float("inf")}),
        (
            "CalculateVariantServicesProvision",
            {"services": {"22": {"summary": None, "error": "missing norm"}}},
        ),
    ],
)
def test_unsubstantiated_domain_results_are_not_success(tool, result):
    from src.agents.services.planning.planning_service import completed_domain_operation

    assert not completed_domain_operation(tool, result)


def test_polygon_coverage_detects_lost_area():
    from copy import deepcopy

    from src.agents.services.planning.artifacts import compare_layer_coverage

    before = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [28, 61],
                            [28.001, 61],
                            [28.001, 61.001],
                            [28, 61.001],
                            [28, 61],
                        ]
                    ],
                },
                "properties": {},
            }
        ],
    }
    after = deepcopy(before)
    after["features"][0]["geometry"]["coordinates"] = [
        [[28, 61], [28.0005, 61], [28.0005, 61.001], [28, 61.001], [28, 61]]
    ]
    result = compare_layer_coverage(before, after)
    assert result["features_with_area_loss"] == 1
    assert result["lost_area_m2"] > 1
    assert compare_layer_coverage(before, before)["lost_area_m2"] == 0
