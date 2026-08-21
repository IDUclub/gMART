from src.agents.services.compliance_requirements import (
    ComplianceDataGate,
    describe_scenario_layer,
)
from src.agents.services.service_entities.compliance import (
    CheckPlan,
    DeclaredRequirements,
)


def _fc(properties):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [30 + i / 100, 60]},
                "properties": value,
            }
            for i, value in enumerate(properties)
        ],
    }


def test_describe_layer_reports_nested_fill_rate_and_revision():
    profile = describe_scenario_layer(
        "objects",
        "houses",
        _fc([{"building": {"floors": 2}}, {"building": {"floors": None}}]),
    )
    floors = next(
        item for item in profile["fields"] if item["name"] == "building.floors"
    )
    assert floors["fill_rate"] == 0.5
    assert profile["complete"] is True
    assert profile["revision"].startswith("sha256:")


def test_describe_layer_reports_numeric_fill_rate_for_string_values():
    profile = describe_scenario_layer(
        "objects",
        "houses",
        _fc([{"floors": "3"}, {"floors": "unknown"}, {"floors": None}]),
    )
    floors = next(item for item in profile["fields"] if item["name"] == "floors")
    assert floors["fill_rate"] == 2 / 3
    assert floors["numeric_fill_rate"] == 1 / 3


def test_gate_selects_candidates_by_priority_and_marks_low_fill_unverifiable():
    requirements = DeclaredRequirements.model_validate(
        {
            "layers": [
                {
                    "role": "objects",
                    "entity": "жилой дом",
                    "entity_type": "physical_object",
                    "geometry_types": ["Point"],
                }
            ],
            "attributes": [
                {
                    "role": "floors",
                    "on": "objects",
                    "accepts": [
                        {"field": "floors", "unit": "floor", "quality": "direct"},
                        {
                            "field": "height_m",
                            "unit": "m",
                            "derive": "height_to_floors_v1",
                            "quality": "derived",
                        },
                    ],
                    "min_fill_rate": 0.6,
                }
            ],
        }
    )
    plan = CheckPlan.model_validate(
        {
            "schema_version": "1.0",
            "template": "distance_table",
            "template_version": 1,
            "params": {},
            "source": {"restriction_id": "r1"},
            "planner_status": "auto",
        }
    )
    result = ComplianceDataGate().resolve(
        plan,
        {"жилой дом": _fc([{"floors": 2}, {"floors": None}])},
        requirements,
    )
    assert result.executable is False
    assert "attribute:floors:fill_rate" in result.missing


def test_gate_uses_registered_height_derivation_when_direct_field_is_absent():
    requirements = DeclaredRequirements.model_validate(
        {
            "layers": [
                {
                    "role": "objects",
                    "entity": "жилой дом",
                    "entity_type": "physical_object",
                    "geometry_types": ["Point"],
                }
            ],
            "attributes": [
                {
                    "role": "floors",
                    "on": "objects",
                    "accepts": [
                        {"field": "floors", "unit": "floor", "quality": "direct"},
                        {
                            "field": "height_m",
                            "unit": "m",
                            "derive": "height_to_floors_v1",
                            "quality": "derived",
                        },
                    ],
                    "min_fill_rate": 1,
                }
            ],
        }
    )
    plan = CheckPlan.model_validate(
        {
            "schema_version": "1.0",
            "template": "distance_table",
            "template_version": 1,
            "params": {},
            "source": {"restriction_id": "r1"},
            "planner_status": "auto",
        }
    )
    result = ComplianceDataGate().resolve(
        plan, {"жилой дом": _fc([{"height_m": 9}])}, requirements
    )
    assert result.executable is True
    assert result.selected_fields["floors"] == "__derived__.floors"
    assert (
        result.layers["жилой дом"]["features"][0]["properties"]["__derived__.floors"]
        == 3
    )


def test_gate_does_not_treat_attribute_schema_of_empty_complete_layer_as_missing():
    requirements = DeclaredRequirements.model_validate(
        {
            "layers": [
                {
                    "role": "objects",
                    "entity": "жилой дом",
                    "entity_type": "physical_object",
                    "geometry_types": ["Point"],
                }
            ],
            "attributes": [
                {
                    "role": "height",
                    "on": "objects",
                    "accepts": [
                        {"field": "height_m", "unit": "m", "quality": "direct"}
                    ],
                }
            ],
        }
    )
    plan = CheckPlan.model_validate(
        {
            "schema_version": "1.0",
            "template": "zonal_attribute_threshold",
            "template_version": 1,
            "params": {},
            "source": {"restriction_id": "r-empty"},
            "planner_status": "reviewed",
        }
    )

    result = ComplianceDataGate().resolve(
        plan,
        {"жилой дом": {"type": "FeatureCollection", "features": []}},
        requirements,
    )

    assert result.executable is True
    assert result.selected_fields["height"] == "height_m"
    assert result.resolved[-1].reason == "empty_layer_no_values"


def test_gate_marks_missing_height_on_existing_objects_unverifiable():
    requirements = DeclaredRequirements.model_validate(
        {
            "layers": [
                {
                    "role": "objects",
                    "entity": "жилой дом",
                    "entity_type": "physical_object",
                    "geometry_types": ["Point"],
                }
            ],
            "attributes": [
                {
                    "role": "height",
                    "on": "objects",
                    "accepts": [
                        {"field": "height_m", "unit": "m", "quality": "direct"}
                    ],
                }
            ],
        }
    )
    plan = CheckPlan.model_validate(
        {
            "schema_version": "1.0",
            "template": "zonal_attribute_threshold",
            "template_version": 1,
            "params": {},
            "source": {"restriction_id": "r-no-height"},
            "planner_status": "reviewed",
        }
    )

    result = ComplianceDataGate().resolve(
        plan, {"жилой дом": _fc([{"name": "Дом без высоты"}])}, requirements
    )

    assert result.executable is False
    assert result.missing == ["attribute:height"]
