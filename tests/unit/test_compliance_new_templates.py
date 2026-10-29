"""object_attribute_threshold, accessibility_within and service_provision (CheckPlan v1)."""

import pytest
from pydantic import ValidationError
from shapely.geometry import Point, Polygon, mapping

from src.agents.services.compilance.compliance_executor import (
    ComplianceTemplateExecutor,
)
from src.agents.services.compilance.compliance_registry import (
    DEFAULT_COMPLIANCE_REGISTRY,
)
from src.agents.services.compilance.compliance_requirements import ComplianceDataGate
from src.agents.services.service_entities.compliance import (
    AccessibilityWithinParams,
    CheckPlan,
    DeclaredRequirements,
    ServiceProvisionParams,
)
from src.idu_mcp.tools_services.compliance_geometry import ComplianceGeometryTools
from tests.unit.test_compliance_executor import FakeMcpClient


def _building(polygon_side_deg, floors, x=30.0):
    square = Polygon(
        [
            (x, 60.0),
            (x + polygon_side_deg, 60.0),
            (x + polygon_side_deg, 60.0 + polygon_side_deg / 2),
            (x, 60.0 + polygon_side_deg / 2),
        ]
    )
    return {
        "type": "Feature",
        "geometry": mapping(square),
        "properties": {
            "physical_object_id": int(x * 1e5),
            "building": {"floors": floors},
        },
    }


def _houses(*features):
    return {"type": "FeatureCollection", "features": list(features)}


class AttributeClient(FakeMcpClient):
    def __init__(self, houses):
        super().__init__()
        self.houses = houses

    async def execute_tool(self, name, arguments, meta=None):
        self.calls.append((name, arguments, meta))
        if name == "GetPhysicalObjects":
            return {"Жилой дом": self.houses}
        if name == "GetProjectTerritory":
            return {"project_territory": _houses(_building(0.1, 1))}
        if name == "GetServicesByGeometry":
            return {
                "Школа": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": mapping(Point(30.0, 60.0)),
                            "properties": {"service_id": 1},
                        }
                    ],
                }
            }
        if name == "CheckObjectAttributeThreshold":
            return ComplianceGeometryTools().object_attribute_threshold(**arguments)
        if name == "CheckAccessibilityWithin":
            return ComplianceGeometryTools().accessibility_within(**arguments)
        raise AssertionError(name)


def _attribute_plan(accepts, threshold, unit, operator="<="):
    return {
        "schema_version": "1.0",
        "template": "object_attribute_threshold",
        "template_version": 1,
        "params": {
            "objects_layer": "objects",
            "attribute_role": "attribute",
            "operator": operator,
            "threshold": threshold,
            "unit": unit,
            "result_mode": "both",
        },
        "declared_requirements": {
            "layers": [
                {
                    "role": "objects",
                    "entity": "Жилой дом",
                    "entity_type": "physical_object",
                    "geometry_types": ["Polygon", "MultiPolygon"],
                }
            ],
            "attributes": [
                {
                    "role": "attribute",
                    "on": "objects",
                    "min_fill_rate": 0,
                    "accepts": accepts,
                }
            ],
        },
        "source": {"restriction_id": "r-attr"},
        "planner_status": "auto",
    }


HEIGHT = [
    {"field": "building.height", "unit": "m", "quality": "direct"},
    {
        "field": "building.floors",
        "unit": "m",
        "derive": "floors_to_height_v1",
        "quality": "derived",
    },
]


async def test_height_limit_uses_floors_when_no_height_and_leaves_gaps_unchecked():
    client = AttributeClient(
        _houses(
            _building(0.0002, 9, x=30.0),
            _building(0.0002, 12, x=30.01),
            _building(0.0002, None, x=30.02),
        )
    )
    execution = await ComplianceTemplateExecutor().execute(
        client, _attribute_plan(HEIGHT, 28, "m"), 772
    )
    result = execution.result
    assert result.verification_status == "partial"
    assert result.compliance_status == "violated"
    assert (result.summary.violated_objects, result.summary.passed_objects) == (1, 1)
    assert result.coverage.unchecked_objects == 1
    measured = sorted(item.measured_value for item in result.evidence)
    assert measured == [27.0, 36.0]
    derived = next(
        item for item in result.resolved_requirements if item.role == "attribute"
    )
    assert derived.derive == "floors_to_height_v1" and derived.quality == "derived"
    assert [call[0] for call in client.calls][-1] == "CheckObjectAttributeThreshold"


async def test_area_limit_is_derived_from_polygon_geometry():
    client = AttributeClient(
        _houses(_building(0.001, 5, x=30.0), _building(0.0001, 5, x=30.01))
    )
    accepts = [
        {
            "field": "geometry",
            "unit": "m2",
            "derive": "geometry_area_m2_v1",
            "quality": "derived",
        }
    ]
    execution = await ComplianceTemplateExecutor().execute(
        client, _attribute_plan(accepts, 1000, "m2", operator=">="), 772
    )
    result = execution.result
    assert result.verification_status == "complete"
    assert (result.summary.violated_objects, result.summary.passed_objects) == (1, 1)
    assert all(item.unit == "m2" for item in result.evidence)


def test_geometry_area_profile_ignores_points():
    profile = ComplianceDataGate._geometry_area_profile(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": mapping(Point(30, 60)),
                    "properties": {},
                },
                _building(0.001, 1),
            ],
        }
    )
    assert profile["numeric_fill_rate"] == 0.5


def test_accessibility_radius_from_time_and_route_length():
    time = AccessibilityWithinParams.model_validate(
        {
            "objects_layer": "o",
            "required_neighbor_layers": ["n"],
            "limit": {"kind": "time", "minutes": 13},
        }
    )
    assert time.radius_m() == 13 * 80 / 1.3
    route = AccessibilityWithinParams.model_validate(
        {
            "objects_layer": "o",
            "required_neighbor_layers": ["n"],
            "limit": {"kind": "distance", "meters": 650},
            "detour_factor": 1.0,
        }
    )
    assert route.radius_m() == 650


async def test_accessibility_is_checked_with_a_buffer_and_says_so():
    client = AttributeClient(
        _houses(_building(0.0002, 5, x=30.0005), _building(0.0002, 5, x=30.05))
    )
    plan = {
        "schema_version": "1.0",
        "template": "accessibility_within",
        "template_version": 1,
        "params": {
            "objects_layer": "objects",
            "required_neighbor_layers": ["neighbors"],
            "limit": {"kind": "time", "minutes": 10},
            "speed_m_per_min": 80,
            "detour_factor": 1.3,
            "measurement": "buffer_v1",
            "minimum_neighbors": 1,
            "result_mode": "both",
        },
        "declared_requirements": {
            "layers": [
                {
                    "role": "objects",
                    "entity": "Жилой дом",
                    "entity_type": "physical_object",
                },
                {"role": "neighbors", "entity": "Школа", "entity_type": "service"},
            ],
            "attributes": [],
        },
        "source": {"restriction_id": "r-acc"},
        "planner_status": "auto",
    }
    execution = await ComplianceTemplateExecutor().execute(client, plan, 772)
    result = execution.result
    assert (result.summary.violated_objects, result.summary.passed_objects) == (1, 1)
    name, arguments, _ = client.calls[-1]
    assert name == "CheckAccessibilityWithin"
    assert round(arguments["radius_m"]) == round(10 * 80 / 1.3)
    assert all(item.template == "accessibility_within" for item in result.evidence)
    assert all(
        any(w.startswith("accessibility_approximated_by_buffer") for w in item.warnings)
        for item in result.evidence
    )


def _provision_plan(**params):
    return {
        "schema_version": "1.0",
        "template": "service_provision",
        "template_version": 1,
        "params": {
            "services_layer": "services",
            "capacity_per_1000": 124,
            "accessibility": {"kind": "distance", "meters": 500},
            "min_provision": 1.0,
            "result_mode": "both",
            **params,
        },
        "declared_requirements": {
            "layers": [
                {"role": "services", "entity": "Школа", "entity_type": "service"}
            ],
            "attributes": [],
        },
        "source": {"restriction_id": "r-prov", "document_name": "РНГП"},
        "planner_status": "auto",
    }


class FakeEffects:
    def __init__(self):
        self.calls = []

    async def calculate_normative_provision(self, **arguments):
        self.calls.append(arguments)

        def building(building_id, demand, supplied, is_project=True):
            return {
                "type": "Feature",
                "geometry": mapping(Point(30.0, 60.0)),
                "properties": {
                    "building_id": building_id,
                    "demand": demand,
                    "supplied_demands_within": supplied,
                    "is_project": is_project,
                },
            }

        return {
            "normative": {"capacity_per_1000": 124, "accessibility_value": 500},
            "summary": {},
            "buildings": {
                "type": "FeatureCollection",
                "features": [
                    building(1, 10, 10),
                    building(2, 8, 4),
                    building(3, 0, 0),
                    building(4, 5, 0, is_project=False),
                ],
            },
        }


async def test_service_provision_runs_on_object_effects_with_the_norm_values():
    effects = FakeEffects()
    users = []

    async def factory(user_id):
        users.append(user_id)
        return effects

    execution = await ComplianceTemplateExecutor(
        effects_client_factory=factory
    ).execute(FakeMcpClient(), _provision_plan(), 772, user_id="user-1")
    result = execution.result
    assert users == ["user-1"]
    assert effects.calls == [
        {
            "scenario_id": 772,
            "service_type_id": 1,
            "capacity_per_1000": 124,
            "residents_per_service": None,
            "accessibility_type": "dist",
            "accessibility_value": 500,
        }
    ]
    assert result.verification_status == "complete"
    assert result.compliance_status == "violated"
    assert (result.summary.violated_objects, result.summary.passed_objects) == (1, 1)
    violated = next(item for item in result.evidence if item.violated)
    assert violated.measured_value == 0.5 and violated.demand == 8
    assert len(result.violated_features["features"]) == 1
    assert execution.tool_calls == []
    assert (
        execution.effects_tool_calls[0]["function"]["name"]
        == "CalculateNormativeProvision"
    )


async def test_objects_per_residents_norm_is_sent_as_residents_per_service():
    effects = FakeEffects()

    async def factory(user_id):
        return effects

    plan = _provision_plan(capacity_per_1000=None, residents_per_service=10_000)
    await ComplianceTemplateExecutor(effects_client_factory=factory).execute(
        FakeMcpClient(), plan, 772, user_id="user-1"
    )
    assert effects.calls[0]["residents_per_service"] == 10_000
    assert effects.calls[0]["capacity_per_1000"] is None


def test_provision_has_one_capacity_basis():
    with pytest.raises(ValidationError):
        ServiceProvisionParams.model_validate(
            _provision_plan(residents_per_service=10_000)["params"]
        )


async def test_service_provision_without_user_or_effects_is_unverifiable():
    execution = await ComplianceTemplateExecutor(effects_client_factory=None).execute(
        FakeMcpClient(), _provision_plan(), 772, user_id="user-1"
    )
    assert execution.result.verification_status == "unverifiable"
    assert execution.result.missing_requirements == ["effects_mcp:unavailable"]


def test_registry_publishes_the_new_templates():
    templates = {
        item["template"]: item["tool_names"]
        for item in DEFAULT_COMPLIANCE_REGISTRY.public_manifest()["templates"]
    }
    assert templates["object_attribute_threshold"] == ["CheckObjectAttributeThreshold"]
    assert templates["accessibility_within"] == ["CheckAccessibilityWithin"]
    assert templates["service_provision"] == ["CalculateNormativeProvision"]
    plan = CheckPlan.model_validate(_provision_plan())
    requirements = DEFAULT_COMPLIANCE_REGISTRY.effective_requirements(plan)
    assert requirements["missing_registry_roles"] == []
    assert isinstance(
        DeclaredRequirements(layers=requirements["layers"]), DeclaredRequirements
    )


async def test_inventory_draws_accessibility_areas_of_services():
    from src.agents.services.compilance.compliance_inventory import describe_zone
    from tests.unit.test_restriction_inventory import FakeMcp, _builder

    zone = await _builder().build(FakeMcp(), _provision_plan(), 772)
    assert zone.status == "shown" and zone.zone_kind == "required"
    assert zone.description["around"] == "Школа"
    assert zone.description["distance_m"] == 500 / 1.3
    assert zone.description["capacity_per_1000"] == 124
    assert zone.description["whole_territory"] is False
    assert zone.zone_count == 2
    assert all(
        item["properties"]["buffer_size"] == pytest.approx(500 / 1.3)
        for item in zone.zones["features"]
    )
    assert all(
        item["properties"]["status"] == "not_checked"
        and item["properties"]["feature_kind"] == "zone"
        for item in zone.norm_layer["features"]
    )
    assert "124 мест на 1000 жителей" in describe_zone(zone.payload())

    per_residents = _provision_plan(capacity_per_1000=None, residents_per_service=5000)
    zone = await _builder().build(FakeMcp(), per_residents, 772)
    assert "1 объект на 5000 жителей" in describe_zone(zone.payload())

    urban_normative = _provision_plan(accessibility=None)
    zone = await _builder().build(FakeMcp(), urban_normative, 772)
    assert zone.missing_requirements == ["accessibility:urban_api_normative"]


async def test_inventory_pipeline_draws_accessibility_areas_of_services():
    from tests.unit.test_restriction_inventory import FakeMcp, _inventory_service

    mcp = FakeMcp()
    service = _inventory_service(mcp)
    raw = _provision_plan()
    events = [
        event
        async for event in service._run_restriction_inventory(
            mcp_client=mcp,
            request_id="provision-inventory",
            scenario_id=772,
            restrictions=[{"id": raw["source"]["restriction_id"], "check_plan": raw}],
            checkpoint={},
        )
    ]
    summary = next(
        event["content"] for event in events if event["type"] == "restriction_inventory"
    )
    assert summary["shown_norms"] == 1
    assert summary["zones"][0]["status"] == "not_checked"
    layers = [
        event["content"]["feature_collection"]
        for event in events
        if event["type"] == "feature_collection"
    ]
    assert len(layers) == 1 and len(layers[0]["features"]) == 2
    assert all(
        item["properties"]["status"] == "not_checked" for item in layers[0]["features"]
    )
    queries = [args for name, args in mcp.calls if name == "GetServicesByGeometry"]
    assert len(queries) == 1
    assert queries[0]["buffer_m"] == pytest.approx(500 / 1.3)
    assert not any(name.startswith("Check") for name, _ in mcp.calls)


async def test_provision_check_uses_project_territory_not_inventory_buffers():
    from src.agents.services.compilance.compliance_inventory import describe_zone
    from tests.unit.test_restriction_inventory import PROJECT, FakeMcp, _builder, _fc

    zone = await _builder().build(
        FakeMcp(),
        _provision_plan(),
        772,
        project_territory=_fc((PROJECT, {})),
        inventory=False,
    )
    assert zone.status == "shown" and zone.zone_kind == "required"
    assert zone.description["whole_territory"] is True
    assert zone.zone_count == 1
    args = zone.tool_calls[-1]["function"]["arguments"]
    assert args["geometry_mode"] == "geometry"
    assert args["source_layer"] == "project_territory"
    assert "вся территория проекта" in describe_zone(zone.payload())


async def test_service_provision_reports_how_population_was_chosen():
    class PopulationEffects(FakeEffects):
        async def calculate_normative_provision(self, **arguments):
            raw = await super().calculate_normative_provision(**arguments)
            raw["population"] = {
                "scenario": {
                    "source": "housing_stock",
                    "population": 1013,
                    "housing_capacity": 1013,
                    "indicator": 204314,
                }
            }
            return raw

    effects = PopulationEffects()

    async def factory(user_id):
        return effects

    execution = await ComplianceTemplateExecutor(
        effects_client_factory=factory
    ).execute(FakeMcpClient(), _provision_plan(), 772, user_id="user-1")
    warnings = execution.result.warnings
    assert "population:source=housing_stock" in warnings
    assert any(item.startswith("population_indicator_ignored") for item in warnings)
