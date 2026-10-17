"""Plan layers from uploaded files: targets, context, overlay over a scenario."""

from shapely.geometry import box

from src.agents.services.compilance.compliance_entity_mapping import (
    EntityLayers,
    EntityTarget,
)
from src.agents.services.compilance.compliance_registry import (
    DEFAULT_COMPLIANCE_REGISTRY,
)
from src.agents.services.compilance.compliance_user_layers import UserLayerSource
from src.agents.services.service_entities.compliance import DeclaredRequirements

HOUSE = EntityTarget("Жилой дом", "physical_object")
SCHOOL = EntityTarget("Школа", "service")


def square(x, y, size=0.001):
    return {
        "type": "Polygon",
        "coordinates": [
            [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]
        ],
    }


def obj(object_id, x, y):
    return {
        "type": "Feature",
        "id": object_id,
        "geometry": square(x, y),
        "properties": {},
    }


def plan(source_geometry=("Polygon", "MultiPolygon"), template="distance_from_source"):
    raw = {
        "schema_version": "1.0",
        "template": template,
        "template_version": 1,
        "params": {
            "source_layer": "schools",
            "targets": ["houses"],
            "geometry_mode": "buffered",
            "predicate": "intersects",
            "violation_when": "matched",
            "result_mode": "both",
            "distance_m": 100,
        },
        "declared_requirements": {
            "layers": [
                {
                    "role": "schools",
                    "entity": "Школа",
                    "entity_type": "service",
                    "geometry_types": list(source_geometry),
                    "required": True,
                },
                {
                    "role": "houses",
                    "entity": "Жилой дом",
                    "entity_type": "physical_object",
                    "geometry_types": ["Polygon", "MultiPolygon"],
                    "required": True,
                },
            ],
            "attributes": [],
        },
        "source": {"restriction_id": "r1", "document_name": "СП 42.13330.2016"},
        "planner_status": "auto",
    }
    checked, params = DEFAULT_COMPLIANCE_REGISTRY.validate_plan(raw)
    requirements = DeclaredRequirements.model_validate(raw["declared_requirements"])
    return checked, params, requirements


def layers(project=None, context=None, border=None):
    return EntityLayers(project=project or {}, context=context or {}, border=border)


def test_norm_without_design_objects_is_not_run():
    source = UserLayerSource(layers(project={SCHOOL: [obj("s", 37.48, 55.67)]}))
    assert source.missing(*plan()) == ["user_files:no_project_objects:Жилой дом"]


def test_standalone_needs_sources_in_the_files():
    source = UserLayerSource(layers(project={HOUSE: [obj("h", 37.48, 55.67)]}))
    assert source.missing(*plan()) == ["user_files:no_objects:Школа"]
    with_context = UserLayerSource(
        layers(
            project={HOUSE: [obj("h", 37.48, 55.67)]},
            context={SCHOOL: [obj("s", 37.481, 55.67)]},
        )
    )
    assert with_context.missing(*plan()) == []


async def test_targets_get_design_objects_and_sources_get_the_context():
    source = UserLayerSource(
        layers(
            project={
                HOUSE: [obj("h", 37.48, 55.67)],
                SCHOOL: [obj("s1", 37.49, 55.67)],
            },
            context={
                HOUSE: [obj("h-ctx", 37.5, 55.67)],
                SCHOOL: [obj("s2", 37.481, 55.67)],
            },
        )
    )
    result, calls = await source.retrieve(*plan())
    assert calls == []
    assert [item["id"] for item in result["Жилой дом"]["features"]] == ["h"]
    assert [item["id"] for item in result["Школа"]["features"]] == ["s1", "s2"]
    assert result["Школа"]["meta"]["complete"] is True


async def test_point_requirement_gets_a_point_inside_the_contour():
    source = UserLayerSource(
        layers(
            project={HOUSE: [obj("h", 37.48, 55.67)]},
            context={SCHOOL: [obj("s", 37.481, 55.67)]},
        )
    )
    result, _ = await source.retrieve(*plan(source_geometry=("Point", "MultiPoint")))
    (school,) = result["Школа"]["features"]
    assert school["geometry"]["type"] == "Point"
    assert school["properties"]["geometry_fitted"] == "representative_point"


async def test_overlay_replaces_scenario_objects_inside_the_border():
    requested = []

    async def scenario(requirements):
        requested.append([item.entity for item in requirements.layers])
        return (
            {
                "Жилой дом": {
                    "features": [obj("in", 37.4805, 55.6705), obj("out", 37.6, 55.7)]
                },
                "Школа": {"features": [obj("school", 37.4805, 55.6705)]},
            },
            [
                {
                    "function": {
                        "name": "GetServices",
                        "arguments": {"services_names": ["Школа"]},
                    }
                }
            ],
        )

    source = UserLayerSource(
        layers(
            project={HOUSE: [obj("h", 37.48, 55.67)]},
            border=box(37.479, 55.669, 37.483, 55.673),
        ),
        scenario=scenario,
    )
    # Over a scenario Urban API supplies the sources.
    assert source.missing(*plan()) == []
    result, calls = await source.retrieve(*plan())
    # Targets are only the design's objects; the scenario's school stays a source.
    assert [item["id"] for item in result["Жилой дом"]["features"]] == ["h"]
    assert [item["id"] for item in result["Школа"]["features"]] == ["school"]
    assert calls[0]["function"]["name"] == "GetServices"
    await source.retrieve(*plan())
    assert len(requested) == 1  # scenario layers are fetched once per run


async def test_overlay_keeps_scenario_sources_outside_the_border():
    async def scenario(requirements):
        return (
            {
                "Школа": {
                    "features": [obj("in", 37.4805, 55.6705), obj("out", 37.6, 55.7)]
                }
            },
            [],
        )

    source = UserLayerSource(
        layers(
            project={
                HOUSE: [obj("h", 37.48, 55.67)],
                SCHOOL: [obj("new", 37.481, 55.67)],
            },
            border=box(37.479, 55.669, 37.483, 55.673),
        ),
        scenario=scenario,
    )
    result, _ = await source.retrieve(*plan())
    assert [item["id"] for item in result["Школа"]["features"]] == ["new", "out"]


def test_provision_needs_the_scenario():
    checked, params, requirements = plan()
    checked = checked.model_copy(update={"template": "service_provision"})
    source = UserLayerSource(layers(project={HOUSE: [obj("h", 37.48, 55.67)]}))
    assert source.missing(checked, params, requirements) == [
        "user_files:scenario_only_template"
    ]
