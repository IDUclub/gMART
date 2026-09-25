from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from src.agents.services.compilance.compliance_executor import (
    ComplianceTemplateExecutor,
)
from src.agents.services.service_entities.compliance import DeclaredRequirements
from src.idu_mcp.tools_services.compliance_geometry import ComplianceGeometryTools


class FakeMcpClient:
    def __init__(
        self,
        *,
        unavailable_services=False,
        zero_services=False,
        zero_physical_objects=False,
    ):
        self.calls = []
        self.unavailable_services = unavailable_services
        self.zero_services = zero_services
        self.zero_physical_objects = zero_physical_objects

    async def resolve_urban_entity_types(self, *, service_names, physical_object_names):
        known_services = {"Школа"}
        known_physical_objects = {"Жилой дом"}

        def result(names, known):
            return {
                name: {
                    "found": name in known,
                    "canonical_name": name if name in known else None,
                    "type_id": 1 if name in known else None,
                }
                for name in names
            }

        return {
            "service": result(service_names, known_services),
            "physical_object": result(physical_object_names, known_physical_objects),
        }

    async def get_available_services_prompt(self, scenario_id):
        assert scenario_id == 772
        return "Доступные сервисы: Школа"

    async def get_available_physical_objects_prompt(self, scenario_id):
        assert scenario_id == 772
        return "Доступные физические объекты: Жилой дом"

    async def execute_tool(self, name, arguments, meta=None):
        self.calls.append((name, arguments, meta))
        if name == "GetServices":
            assert arguments["centers_only"] is False
            if self.unavailable_services:
                return None
            if self.zero_services:
                return {"Школа": {"type": "FeatureCollection", "features": []}}
            return {"Школа": _fc(30.0, service_id=1)}
        if name == "GetPhysicalObjects":
            assert arguments["centers_only"] is False
            if self.zero_physical_objects:
                return {"Жилой дом": {"type": "FeatureCollection", "features": []}}
            return {
                "Жилой дом": {
                    "type": "FeatureCollection",
                    "features": [
                        _feature(30.0001, physical_object_id=1),
                        _feature(30.01, physical_object_id=2),
                    ],
                }
            }
        if name == "CheckDistanceFromSource":
            return ComplianceGeometryTools().distance_from_source(**arguments)
        raise AssertionError(name)


def _feature(x, **properties):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [x, 60.0]},
        "properties": properties,
    }


def _fc(x, **properties):
    return {"type": "FeatureCollection", "features": [_feature(x, **properties)]}


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
        "declared_requirements": {
            "layers": [
                {
                    "role": "source",
                    "entity": "Школа",
                    "entity_type": "service",
                    "geometry_types": ["Point"],
                },
                {
                    "role": "targets",
                    "entity": "Жилой дом",
                    "entity_type": "physical_object",
                    "geometry_types": ["Point"],
                },
            ],
            "attributes": [],
        },
        "source": {"restriction_id": "r1", "document_name": "СП"},
        "planner_status": "auto",
    }


async def test_executor_dispatches_valid_plan_without_llm_geometry_planning():
    client = FakeMcpClient()
    execution = await ComplianceTemplateExecutor().execute(client, _plan(), 772)
    assert execution.result.verification_status == "complete"
    assert execution.result.compliance_status == "violated"
    assert execution.result.summary.violated_objects == 1
    assert execution.result.summary.passed_objects == 1
    assert [call[0] for call in client.calls] == [
        "GetServices",
        "GetPhysicalObjects",
        "CheckDistanceFromSource",
    ]


async def test_executor_returns_unsupported_instead_of_silently_skipping():
    plan = _plan()
    plan["template_version"] = 999
    execution = await ComplianceTemplateExecutor().execute(FakeMcpClient(), plan, 772)
    assert execution.result.verification_status == "unsupported"
    assert execution.result.compliance_status == "unknown"
    assert execution.result.missing_requirements
    assert execution.result.source["restriction_id"] == "r1"
    assert execution.result.source["check_plan"]["template_version"] == 999


async def test_executor_rejects_noncanonical_urban_entity_before_retrieval():
    plan = _plan()
    plan["declared_requirements"]["layers"][0]["entity"] = "радиус обслуживания школ"
    client = FakeMcpClient()

    execution = await ComplianceTemplateExecutor().execute(client, plan, 772)

    assert execution.result.verification_status == "unverifiable"
    assert execution.result.compliance_status == "unknown"
    assert execution.result.missing_requirements == ["catalog:source:service:not_found"]
    assert execution.result.resolved_requirements[0].reason == (
        "urban_catalog_entity_not_found"
    )
    assert execution.result.source["document_name"] == "СП"
    assert execution.result.source["check_plan"]["declared_requirements"]
    assert client.calls == []


async def test_executor_treats_empty_mcp_layer_result_as_missing_data():
    execution = await ComplianceTemplateExecutor().execute(
        FakeMcpClient(unavailable_services=True), _plan(), 772
    )

    assert execution.result.verification_status == "unverifiable"
    assert execution.result.compliance_status == "unknown"
    assert execution.result.missing_requirements == ["layer:source"]


async def test_executor_passes_when_complete_applicable_set_is_empty():
    client = FakeMcpClient(zero_physical_objects=True)

    execution = await ComplianceTemplateExecutor().execute(client, _plan(), 772)

    assert execution.result.verification_status == "complete"
    assert execution.result.compliance_status == "passed"
    assert execution.result.coverage.applicable_objects == 0
    assert execution.result.warnings == ["no_applicable_objects"]
    assert [call[0] for call in client.calls] == ["GetServices", "GetPhysicalObjects"]


async def test_executor_passes_prohibition_when_forbidden_source_is_absent():
    client = FakeMcpClient(zero_services=True)

    execution = await ComplianceTemplateExecutor().execute(client, _plan(), 772)

    assert execution.result.verification_status == "complete"
    assert execution.result.compliance_status == "passed"
    assert execution.result.summary.passed_objects == 2


async def test_executor_violates_required_presence_when_source_is_absent():
    plan = _plan()
    plan["params"]["violation_when"] = "not_matched"
    client = FakeMcpClient(zero_services=True)

    execution = await ComplianceTemplateExecutor().execute(client, plan, 772)

    assert execution.result.verification_status == "complete"
    assert execution.result.compliance_status == "violated"
    assert execution.result.summary.violated_objects == 2


@pytest.mark.parametrize(
    "source,target", [("школ", "жилых домов"), ("школы", "жилые дома")]
)
async def test_executor_normalizes_stored_plan_via_catalog_tool(source, target):
    from src.idu_mcp.tools_services.urb_api_tools import UrbanApiTool

    class CatalogMcp(FakeMcpClient):
        async def resolve_urban_entity_types(self, **names):
            api = AsyncMock()
            api.get_type_catalog.side_effect = [{"Школа": 22}, {"Жилой дом": 4}]
            return await UrbanApiTool(api).resolve_entity_types(**names, token="user-1")

    plan = _plan()
    plan["declared_requirements"]["layers"][0]["entity"] = source
    plan["declared_requirements"]["layers"][1]["entity"] = target
    original = deepcopy(plan)
    client = CatalogMcp()
    execution = await ComplianceTemplateExecutor().execute(client, plan, 772)

    assert execution.result.verification_status == "complete"
    assert execution.result.summary.violated_objects == 1
    assert execution.result.summary.passed_objects == 1
    assert client.calls[0][1]["services_names"] == ["Школа"]
    assert client.calls[1][1]["physical_objects_names"] == ["Жилой дом"]
    assert plan == original
    stored_layers = execution.result.source["check_plan"]["declared_requirements"][
        "layers"
    ]
    assert [layer["entity"] for layer in stored_layers] == [source, target]


async def test_history_keeps_only_retrievals_that_found_objects():
    # A client reopening the chat replays these calls into map layers; a call
    # that found nothing would come back as an empty layer.
    client = FakeMcpClient(zero_services=True)
    execution = await ComplianceTemplateExecutor().execute(client, _plan(), 772)

    recorded = [call["function"]["name"] for call in execution.tool_calls]
    assert "GetServices" not in recorded
    assert "GetPhysicalObjects" in recorded
    # The empty layer still reached the check itself.
    assert execution.layers["Школа"]["features"] == []


async def test_empty_functional_zones_are_not_recorded():
    class Tools:
        async def execute_named_tool(self, mcp_client, name, arguments):
            if arguments["zone_type_names"] == ["Жилая зона"]:
                return {"functional_zones": _fc(30.0)}
            return {
                "functional_zones": {
                    "type": "FeatureCollection",
                    "features": [],
                    "meta": {"complete": True, "source": "OSM"},
                }
            }

    executor = ComplianceTemplateExecutor()
    executor.tools = Tools()
    requirements = DeclaredRequirements.model_validate(
        {
            "layers": [
                {
                    "role": "zones",
                    "entity": "Жилая зона",
                    "entity_type": "functional_zone",
                },
                {"role": "other", "entity": "Зона", "entity_type": "functional_zone"},
            ]
        }
    )

    layers, calls = await executor._retrieve_layers(object(), requirements, 772)

    assert set(layers) == {"Жилая зона", "Зона"}
    assert [call["function"]["arguments"]["zone_type_names"] for call in calls] == [
        ["Жилая зона"]
    ]


async def test_an_object_with_many_sources_keeps_its_result():
    class ManySchools(FakeMcpClient):
        async def execute_tool(self, name, arguments, meta=None):
            if name == "GetServices":
                self.calls.append((name, arguments, meta))
                return {
                    "Школа": {
                        "type": "FeatureCollection",
                        "features": [
                            _feature(30.0 + index * 1e-6, service_id=index)
                            for index in range(120)
                        ],
                    }
                }
            return await super().execute_tool(name, arguments, meta)

    execution = await ComplianceTemplateExecutor().execute(ManySchools(), _plan(), 772)

    assert execution.result.compliance_status == "violated"
    crowded = next(item for item in execution.result.evidence if item.violated)
    assert crowded.measured_value == 120
    assert len(crowded.generator_refs) == 120
    assert crowded.warnings == []


def test_geometry_checks_every_object_of_a_large_layer():
    many = {
        "type": "FeatureCollection",
        "features": [_feature(30.0 + index * 1e-3) for index in range(60_000)],
    }
    frame = ComplianceGeometryTools()._layer("objects", {"objects": many})
    assert len(frame) == 60_000


def test_registry_sets_no_object_limit():
    from src.agents.services.compilance.compliance_registry import (
        build_default_registry,
    )

    for entry in build_default_registry()._entries.values():
        assert set(entry.manifest()["limits"]) == {"timeout_seconds"}
