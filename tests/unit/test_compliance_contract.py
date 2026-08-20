import pytest
from pydantic import ValidationError

from src.agents.services.compliance_registry import (
    UnsupportedSchemaError,
    UnsupportedTemplateError,
    build_default_registry,
)


def _plan(**updates):
    value = {
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
                    "entity": "школа",
                    "entity_type": "service",
                    "geometry_types": ["Point"],
                    "required": True,
                },
                {
                    "role": "targets",
                    "entity": "жилой дом",
                    "entity_type": "physical_object",
                    "geometry_types": ["Polygon", "MultiPolygon"],
                    "required": True,
                },
            ],
            "attributes": [],
        },
        "source": {"restriction_id": "restriction-1"},
        "planner_status": "auto",
    }
    value.update(updates)
    return value


def test_registry_validates_exact_contract_and_template_version():
    plan, params = build_default_registry().validate_plan(_plan())
    assert plan.schema_version == "1.0"
    assert params.distance_m == 50


def test_unknown_schema_is_not_coerced():
    with pytest.raises(UnsupportedSchemaError):
        build_default_registry().validate_plan(_plan(schema_version="2.0"))


def test_unknown_template_version_is_not_downgraded():
    with pytest.raises(UnsupportedTemplateError):
        build_default_registry().validate_plan(_plan(template_version=2))


def test_source_geometry_forbids_zero_or_nonzero_buffer():
    raw = _plan()
    raw["params"] = {
        **raw["params"],
        "geometry_mode": "source_geometry",
        "distance_m": 1,
    }
    with pytest.raises(ValidationError):
        build_default_registry().validate_plan(raw)


def test_registry_requirement_cannot_be_weakened():
    raw = _plan()
    raw["declared_requirements"]["layers"][0]["required"] = False
    plan, _ = build_default_registry().validate_plan(raw)
    effective = build_default_registry().effective_requirements(plan)
    assert effective["layers"][0].required is True


def test_disabled_template_is_unsupported():
    raw = _plan(template="zonal_ratio")
    raw["params"] = {
        "zones_layer": "zones",
        "numerator": {"layer": "objects", "measure": "area"},
        "denominator": {"measure": "zone_area"},
        "operator": "<=",
        "threshold": 30,
        "unit": "%",
    }
    with pytest.raises(UnsupportedTemplateError):
        build_default_registry({"zonal_ratio"}).validate_plan(raw)
