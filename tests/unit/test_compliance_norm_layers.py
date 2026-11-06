"""Per-norm layers for every MCP geometry template; no agents imports or services."""

from copy import deepcopy

import pytest
from shapely.geometry import GeometryCollection, LineString, Point, box, mapping, shape

from src.idu_mcp.tools_services.compliance_geometry import ComplianceGeometryTools
from src.idu_mcp.tools_services.norm_layers import (
    GROUP_PROPERTIES,
    build_norm_layer,
    check_type,
    source_reference,
)

SOURCE = {
    "document_name": "СП 42.13330.2016 «Градостроительство»",
    "clause_number": "7.1",
    "extraction_text": "Исходный текст нормы.",
}
COMMON = {"restriction_id": "r1", "provenance": SOURCE, "result_mode": "both"}
TOOLS = ComplianceGeometryTools()


def fc(*items):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": properties,
                "id": str(index),
            }
            for index, (geometry, properties) in enumerate(items)
        ],
    }


def kinds(layer, kind):
    return [
        feature
        for feature in layer["features"]
        if feature["properties"]["feature_kind"] == kind
    ]


def assert_layer(result, status, check_en):
    layer = result["norm_layer"]
    assert layer["meta"]["group_properties"] == GROUP_PROPERTIES
    assert layer["meta"]["group_by"] == "norm_id"
    assert layer["meta"]["norm_status"] == status
    for feature in layer["features"]:
        properties = feature["properties"]
        assert properties["norm_id"] == "r1"
        assert properties["check_type_en"] == check_en
        assert properties["norm_status"] == status
        assert properties["norm_reference"] == "СП 42.13330.2016, п. 7.1"
        assert properties["norm_text"] == SOURCE["extraction_text"]
        assert set(GROUP_PROPERTIES) <= properties.keys()
        assert feature["geometry"]["type"] != "GeometryCollection"
        if properties["feature_kind"] == "zone" and properties["status"] in {
            "partial",
            "violated",
        }:
            assert properties["status_reason"]
        if status in {"partial", "violated"}:
            assert properties["norm_status_reason"]
        if properties["feature_kind"] == "violation":
            assert {
                "object_ref",
                "object_name",
                "measured_value",
                "required_value",
                "deficit",
                "unit",
                "explanation_ru",
                "violation_geometry_mode",
                "zone_ref",
            } <= properties.keys()
            assert properties["explanation_ru"]
    return layer


@pytest.mark.parametrize(
    "violation_when,expected_statuses,mode",
    [
        ("matched", ["violated", "passed"], "intersection"),
        ("not_matched", ["passed", "violated"], "object"),
    ],
)
def test_distance_zones_are_checked_independently(
    violation_when, expected_statuses, mode
):
    layers = {
        "source": fc(
            (Point(30, 60), {"service_id": 1, "name": "Школа"}),
            (Point(30.02, 60), {"service_id": 2}),
        ),
        "targets": fc((box(29.999, 59.999, 30.001, 60.001), {"physical_object_id": 1})),
    }
    result = TOOLS.distance_from_source(
        source_layer="source",
        targets=["targets"],
        geometry_mode="buffered",
        distance_m=100,
        predicate="intersects",
        violation_when=violation_when,
        layers=layers,
        **COMMON,
    )
    layer = assert_layer(result, "violated", "distance")
    zones = kinds(layer, "zone")
    assert [f["properties"]["status"] for f in zones] == expected_statuses
    violation = kinds(layer, "violation")[0]
    assert violation["properties"]["violation_geometry_mode"] == mode
    original = shape(layers["targets"]["features"][0]["geometry"])
    if mode == "intersection":
        assert shape(violation["geometry"]).equals(
            original.intersection(shape(zones[0]["geometry"]))
        )
    else:
        assert shape(violation["geometry"]).equals(original)
    assert violation["properties"]["zone_ref"] in [
        f["properties"]["zone_ref"] for f in zones
    ]
    assert kinds(layer, "object")[0]["properties"]["zone_refs"]


def test_distance_explanation_measures_meters_instead_of_match_count():
    result = TOOLS.distance_from_source(
        source_layer="schools",
        targets=["targets"],
        geometry_mode="buffered",
        distance_m=100,
        predicate="intersects",
        violation_when="matched",
        layers={
            "schools": fc((Point(30, 60), {"service_id": 1, "name": "Школа"})),
            "targets": fc((Point(30.001, 60), {"physical_object_id": 2})),
        },
        **COMMON,
    )
    props = kinds(result["norm_layer"], "violation")[0]["properties"]
    assert 50 < props["measured_value"] < 60
    assert props["required_value"] == 100
    assert props["deficit"] == pytest.approx(100 - props["measured_value"])
    assert "Расстояние до объекта «Школа»" in props["explanation_ru"]
    assert "требуется ≥ 100 м" in props["explanation_ru"]
    assert "не хватает" in props["explanation_ru"]


def test_missing_distance_target_violates_zone_without_inventing_object_geometry():
    result = TOOLS.distance_from_source(
        source_layer="source",
        targets=["targets"],
        geometry_mode="buffered",
        distance_m=100,
        predicate="intersects",
        violation_when="not_matched",
        layers={
            "source": fc((Point(30, 60), {"service_id": 1})),
            "targets": fc(),
        },
        **COMMON,
    )
    layer = assert_layer(result, "violated", "distance")
    assert kinds(layer, "zone")[0]["properties"]["status_reason"] == (
        "Требуемый объект отсутствует в зоне."
    )
    assert kinds(layer, "violation") == []


@pytest.mark.parametrize(
    "predicate,status", [("intersects", "violated"), ("within", "passed")]
)
def test_check_distance_uses_exact_object_relative_predicate(predicate, status):
    result = TOOLS.distance_from_source(
        source_layer="source",
        targets=["targets"],
        geometry_mode="source_geometry",
        predicate=predicate,
        violation_when="matched",
        layers={
            "source": fc((box(30, 60, 30.01, 60.01), {"service_id": 1})),
            "targets": fc(
                (box(30.005, 60.005, 30.02, 60.02), {"physical_object_id": 1})
            ),
        },
        **COMMON,
    )
    assert_layer(result, status, "distance")
    assert result["summary"]["violated_objects"] == int(status == "violated")


def test_distance_table_preserves_individual_radius_and_missing_sources_are_partial():
    result = TOOLS.distance_table(
        source_layer="source",
        targets=["targets"],
        attribute_field="floors",
        bands=[
            {"min": 1, "max": 3, "distance_m": 20},
            {"min": 4, "max": None, "distance_m": 100},
        ],
        predicate="intersects",
        violation_when="matched",
        layers={
            "source": fc(
                (Point(30, 60), {"physical_object_id": 1, "floors": 2}),
                (Point(30.02, 60), {"physical_object_id": 2, "floors": 10}),
                (Point(30.04, 60), {"physical_object_id": 3, "floors": None}),
            ),
            "targets": fc(),
        },
        **COMMON,
    )
    layer = assert_layer(result, "partial", "distance")
    assert [f["properties"]["buffer_size"] for f in kinds(layer, "zone")] == [20, 100]
    assert "источников" in layer["meta"]["norm_status_reason"]
    assert "floors" in layer["meta"]["norm_status_reason"]


@pytest.mark.parametrize(
    "violation_when,x,mode",
    [
        ("matched", 30.0001, "intersection"),
        ("not_matched", 30.01, "object"),
    ],
)
def test_distance_table_violation_modes(violation_when, x, mode):
    result = TOOLS.distance_table(
        source_layer="source",
        targets=["targets"],
        attribute_field="floors",
        bands=[{"min": 1, "max": None, "distance_m": 100}],
        predicate="intersects",
        violation_when=violation_when,
        layers={
            "source": fc((Point(30, 60), {"physical_object_id": 1, "floors": 4})),
            "targets": fc((Point(x, 60), {"physical_object_id": 2})),
        },
        **COMMON,
    )
    layer = assert_layer(result, "violated", "distance")
    props = kinds(layer, "violation")[0]["properties"]
    assert props["violation_geometry_mode"] == mode
    assert props["required_value"] == 100


@pytest.mark.parametrize("template", ["presence_within", "accessibility_within"])
def test_neighbor_templates_include_found_objects_and_missing_explanations(template):
    layers = {
        "houses": fc(
            (Point(30, 60), {"physical_object_id": 1}),
            (Point(30.02, 60), {"physical_object_id": 2}),
        ),
        "theaters": fc((Point(30.0001, 60), {"service_id": 1, "name": "Театр"})),
    }
    kwargs = (
        {"distance_m": 100}
        if template == "presence_within"
        else {"radius_m": 100, "limit": {"kind": "time", "minutes": 10}}
    )
    result = getattr(TOOLS, template)(
        objects_layer="houses",
        required_neighbor_layers=["theaters"],
        minimum_neighbors=1,
        layers=layers,
        **kwargs,
        **COMMON,
    )
    layer = assert_layer(
        result,
        "violated",
        "accessibility" if template == "accessibility_within" else "distance",
    )
    assert [z["properties"]["status"] for z in kinds(layer, "zone")] == [
        "passed",
        "violated",
    ]
    objects = kinds(layer, "object")
    assert any(f["properties"]["object_ref"]["namespace"] == "service" for f in objects)
    props = kinds(layer, "violation")[0]["properties"]
    assert props["violation_geometry_mode"] == "object"
    assert props["measured_value"] == 0
    assert props["deficit"] == 1
    assert "не хватает 1 шт." in props["explanation_ru"]


@pytest.mark.parametrize(
    "predicate,status", [("intersects", "passed"), ("within", "violated")]
)
def test_presence_predicate_applies_to_neighbor_relative_to_buffer(predicate, status):
    result = TOOLS.presence_within(
        objects_layer="objects",
        required_neighbor_layers=["neighbors"],
        distance_m=100,
        minimum_neighbors=1,
        predicate=predicate,
        layers={
            "objects": fc((Point(30, 60), {"physical_object_id": 1})),
            "neighbors": fc((box(29.9, 59.9, 30.1, 60.1), {"service_id": 1})),
        },
        **COMMON,
    )
    assert_layer(result, status, "distance")
    assert result["summary"]["violated_objects"] == int(status == "violated")


def test_zonal_thresholds_use_each_zones_value_and_clip_to_project():
    result = TOOLS.zonal_attribute_threshold(
        objects_layer="objects",
        zones_layer="zones",
        object_attribute="building.floors",
        operator="<=",
        constant_threshold=None,
        zone_threshold_attribute="limit",
        join_predicate="intersects",
        layers={
            "zones": fc(
                (box(30, 60, 30.02, 60.02), {"limit": 10}),
                (box(30, 60, 30.02, 60.02), {"limit": 7}),
            ),
            "objects": fc(
                (
                    box(30.005, 60.005, 30.015, 60.015),
                    {"physical_object_id": 1, "building": {"floors": 8}},
                )
            ),
            "project_territory": fc((box(30, 60, 30.01, 60.01), {})),
        },
        **COMMON,
    )
    layer = assert_layer(result, "violated", "floors")
    zones = kinds(layer, "zone")
    assert [f["properties"]["status"] for f in zones] == ["passed", "violated"]
    assert [f["properties"]["threshold"] for f in zones] == [10, 7]
    assert all(shape(f["geometry"]).equals(box(30, 60, 30.01, 60.01)) for f in zones)
    violation = kinds(layer, "violation")[0]
    assert shape(violation["geometry"]).equals(box(30.005, 60.005, 30.01, 60.01))
    assert violation["properties"]["deficit"] == 1
    assert "превышение 1 эт." in violation["properties"]["explanation_ru"]


@pytest.mark.parametrize(
    "value,status", [(None, "partial"), (4, "passed"), (12, "violated")]
)
def test_object_threshold_project_zone_statuses_and_partial_reason(value, status):
    result = TOOLS.object_attribute_threshold(
        objects_layer="houses",
        object_attribute="floors",
        operator="<=",
        threshold=5,
        unit="floors",
        layers={
            "houses": fc(
                (Point(30.005, 60.005), {"physical_object_id": 1, "floors": value})
            ),
            "project_territory": fc((box(30, 60, 30.01, 60.01), {})),
        },
        **COMMON,
    )
    layer = assert_layer(result, status, "floors")
    zone = kinds(layer, "zone")[0]
    assert zone["properties"]["status"] == status
    if status == "partial":
        assert "Не проверено 1 из 1" in zone["properties"]["status_reason"]
        assert "floors" in zone["properties"]["status_reason"]
    if status == "violated":
        assert (
            kinds(layer, "violation")[0]["properties"]["violation_geometry_mode"]
            == "object"
        )


def test_violation_precedes_partial_and_result_mode_does_not_hide_norm_violations():
    common = {**COMMON, "result_mode": "passed"}
    result = TOOLS.object_attribute_threshold(
        objects_layer="houses",
        object_attribute="height_m",
        operator="<=",
        threshold=10,
        unit="m",
        layers={
            "houses": fc(
                (Point(30.001, 60.001), {"physical_object_id": 1, "height_m": 20}),
                (Point(30.002, 60.002), {"physical_object_id": 2, "height_m": None}),
            ),
            "project_territory": fc((box(30, 60, 30.01, 60.01), {})),
        },
        **common,
    )
    layer = assert_layer(result, "violated", "height")
    assert result["violated_objects"]["features"] == []
    assert len(kinds(layer, "violation")) == 1
    assert (
        "Не проверено объектов: 1"
        in kinds(layer, "zone")[0]["properties"]["status_reason"]
    )


def test_ratio_violates_entire_clipped_zone():
    area = box(30, 60, 30.02, 60.01)
    project = box(30, 60, 30.01, 60.01)
    result = TOOLS.zonal_ratio(
        zones_layer="zones",
        numerator_layer="trees",
        operator=">=",
        threshold=80,
        invalid_geometry_policy="repair",
        layers={
            "zones": fc((area, {})),
            "trees": fc((box(30, 60, 30.002, 60.01), {"physical_object_id": 2})),
            "project_territory": fc((project, {})),
        },
        **COMMON,
    )
    layer = assert_layer(result, "violated", "area_ratio")
    feature = kinds(layer, "violation")[0]
    assert shape(feature["geometry"]).equals(project)
    assert feature["properties"]["violation_geometry_mode"] == "zone"
    assert feature["properties"]["unit"] == "%"


def test_inventory_has_only_not_checked_zones_and_preserves_legacy_collection():
    collection = TOOLS.restriction_zones(
        geometry_mode="buffer",
        source_layer="source",
        distance_m=50,
        layers={"source": fc((Point(30, 60), {"service_id": 1}))},
        properties={
            "restriction_id": "r1",
            "template": "accessibility_within",
            "provenance": SOURCE,
        },
    )
    layer = collection["norm_layer"]
    assert len(collection["features"]) == 1
    assert_layer({"norm_layer": layer}, "not_checked", "accessibility")
    assert {f["properties"]["feature_kind"] for f in layer["features"]} == {"zone"}
    assert all(
        f["properties"]["status_ru"] == "не проверялось" for f in layer["features"]
    )


async def test_public_inventory_adds_top_level_norm_layer():
    from src.idu_mcp.tools_interfaces.geom_interface import create_restriction_zones

    result = await create_restriction_zones(
        layer_name="zones",
        geometry_mode="buffer",
        source_layer="source",
        distance_m=50,
        layers={"source": fc((Point(30, 60), {"service_id": 1}))},
        properties={
            "restriction_id": "r1",
            "template": "distance_from_source",
            "provenance": SOURCE,
        },
        tools=TOOLS,
    )
    assert set(result) == {"zones", "norm_layer"}
    assert "norm_layer" not in result["zones"]
    assert_layer(result, "not_checked", "distance")


def test_object_threshold_without_territory_reports_partial_in_new_layer_only():
    result = TOOLS.object_attribute_threshold(
        objects_layer="houses",
        object_attribute="floors",
        operator="<=",
        threshold=5,
        unit="floors",
        layers={
            "houses": fc((Point(30, 60), {"physical_object_id": 1, "floors": 9})),
        },
        **COMMON,
    )
    layer = assert_layer(result, "partial", "floors")
    assert result["summary"]["violated_objects"] == 1
    assert kinds(layer, "zone") == []
    assert "территория проекта" in layer["meta"]["norm_status_reason"]


def test_norm_violation_precedes_a_different_partial_zone():
    result = TOOLS.zonal_attribute_threshold(
        objects_layer="objects",
        zones_layer="zones",
        object_attribute="floors",
        operator="<=",
        constant_threshold=5,
        zone_threshold_attribute=None,
        join_predicate="within",
        layers={
            "objects": fc(
                (Point(30.005, 60.005), {"physical_object_id": 1, "floors": 9}),
                (Point(30.025, 60.005), {"physical_object_id": 2, "floors": None}),
            ),
            "zones": fc(
                (box(30, 60, 30.01, 60.01), {}), (box(30.02, 60, 30.03, 60.01), {})
            ),
        },
        **COMMON,
    )
    layer = assert_layer(result, "violated", "floors")
    assert [f["properties"]["status"] for f in kinds(layer, "zone")] == [
        "violated",
        "partial",
    ]


def test_ratio_zero_area_is_partial_with_reason():
    result = TOOLS.zonal_ratio(
        zones_layer="zones",
        numerator_layer="trees",
        operator=">=",
        threshold=10,
        invalid_geometry_policy="repair",
        layers={
            "zones": fc((LineString([(30, 60), (30.01, 60)]), {})),
            "trees": fc(),
        },
        **COMMON,
    )
    layer = assert_layer(result, "partial", "area_ratio")
    assert "площади зоны" in kinds(layer, "zone")[0]["properties"]["status_reason"]


def test_reusable_provision_builder_flattens_geometry_collections_without_mutation():
    zone = {
        "type": "Feature",
        "geometry": mapping(
            GeometryCollection(
                [box(30, 60, 30.01, 60.01), LineString([(30, 60), (30.01, 60.01)])]
            )
        ),
        "properties": {
            "feature_kind": "zone",
            "status": "violated",
            "status_reason": "Обеспеченность ниже нормы.",
        },
    }
    original = deepcopy(zone)
    result = build_norm_layer(
        restriction_id="r1",
        template="service_provision",
        provenance=SOURCE,
        zone_features=[zone],
    )
    assert_layer({"norm_layer": result}, "violated", "provision")
    assert len(result["features"]) == 2
    assert zone == original


@pytest.mark.parametrize(
    "template,attribute,expected",
    [
        ("distance_table", "floors", ("distance", "расстояние")),
        ("object_attribute_threshold", "building.floors", ("floors", "этажность")),
        ("zonal_attribute_threshold", "height_m", ("height", "высота")),
        (
            "object_attribute_threshold",
            "capacity",
            ("attribute_threshold", "пороговое значение"),
        ),
    ],
)
def test_check_type_table(template, attribute, expected):
    assert check_type(template, attribute) == expected


def test_source_reference_fallbacks_match_existing_labels():
    assert source_reference({}) == "Источник не указан"
    assert (
        source_reference({"document_name": " ПЗЗ города ", "clause_number": " 2 "})
        == "ПЗЗ города, п. 2"
    )
