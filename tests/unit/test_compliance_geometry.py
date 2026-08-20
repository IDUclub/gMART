from shapely.geometry import Point, Polygon, mapping

from src.idu_mcp.tools_services.compliance_geometry import ComplianceGeometryTools


def _fc(*features):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": str(index),
                "geometry": mapping(geometry),
                "properties": properties,
            }
            for index, (geometry, properties) in enumerate(features)
        ],
    }


def test_distance_from_source_keeps_full_target_set_and_boundary_baskets():
    layers = {
        "source": _fc((Point(0, 0), {"service_id": 1, "name": "source"})),
        "targets": _fc(
            (Point(0.0001, 0), {"physical_object_id": 10}),
            (Point(0.01, 0), {"physical_object_id": 11}),
        ),
    }
    result = ComplianceGeometryTools().distance_from_source(
        source_layer="source",
        targets=["targets"],
        geometry_mode="buffered",
        distance_m=50,
        predicate="intersects",
        violation_when="matched",
        result_mode="both",
        layers=layers,
        restriction_id="r1",
    )
    assert result["coverage"] == {
        "applicable_objects": 2,
        "checked_objects": 2,
        "unchecked_objects": 0,
        "fill_rate": 1.0,
    }
    assert result["summary"] == {"violated_objects": 1, "passed_objects": 1}
    assert len(result["evidence"]) == 2


def test_distance_from_source_includes_touching_boundary_and_all_generators():
    target = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    layers = {
        "source": _fc(
            (Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), {"service_id": 1}),
            (Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]), {"service_id": 2}),
        ),
        "targets": _fc((target, {"physical_object_id": 10})),
    }
    result = ComplianceGeometryTools().distance_from_source(
        source_layer="source",
        targets=["targets"],
        geometry_mode="source_geometry",
        predicate="intersects",
        violation_when="matched",
        result_mode="both",
        layers=layers,
        restriction_id="r-boundary",
        provenance={"document_name": "СП"},
        input_revision="sha256:input",
    )
    assert result["summary"] == {"violated_objects": 1, "passed_objects": 0}
    assert len(result["evidence"][0]["generator_refs"]) == 2
    assert result["evidence"][0]["provenance"] == {"document_name": "СП"}
    assert result["evidence"][0]["input_revision"] == "sha256:input"


def test_presence_within_is_an_anti_join_with_explicit_zero_neighbors():
    layers = {
        "objects": _fc(
            (Point(0, 0), {"physical_object_id": 1}),
            (Point(0.02, 0), {"physical_object_id": 2}),
        ),
        "neighbors": _fc((Point(0.0001, 0), {"service_id": 1})),
    }
    result = ComplianceGeometryTools().presence_within(
        objects_layer="objects",
        required_neighbor_layers=["neighbors"],
        distance_m=100,
        minimum_neighbors=1,
        result_mode="both",
        layers=layers,
        restriction_id="r2",
    )
    assert result["summary"] == {"violated_objects": 1, "passed_objects": 1}
    missing = next(item for item in result["evidence"] if item["violated"])
    assert missing["neighbor_count"] == 0
    assert missing["generator_ref"] is None


def test_distance_table_does_not_guess_missing_source_attribute():
    layers = {
        "source": _fc(
            (Point(0, 0), {"physical_object_id": 1, "floors": 2}),
            (Point(0.01, 0), {"physical_object_id": 2, "floors": None}),
        ),
        "targets": _fc(
            (Point(0.0001, 0), {"physical_object_id": 3}),
            (Point(0.03, 0), {"physical_object_id": 4}),
        ),
    }
    result = ComplianceGeometryTools().distance_table(
        source_layer="source",
        targets=["targets"],
        attribute_field="floors",
        bands=[{"min": 0, "max": 3, "distance_m": 50}],
        predicate="intersects",
        violation_when="matched",
        result_mode="both",
        layers=layers,
        restriction_id="r3",
    )
    assert result["source_coverage"]["checked_objects"] == 1
    assert result["source_coverage"]["unchecked_objects"] == 1
    assert result["coverage"] == {
        "applicable_objects": 2,
        "checked_objects": 1,
        "unchecked_objects": 1,
        "fill_rate": 0.5,
    }
    assert result["evidence"][0]["used_fields"][0]["field"] == "floors"


def test_zonal_attribute_uses_strictest_threshold():
    outer = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    layers = {
        "objects": _fc((Point(0, 0), {"physical_object_id": 1, "floors": 8})),
        "zones": _fc(
            (outer, {"physical_object_id": 10, "limit": 10}),
            (outer, {"physical_object_id": 11, "limit": 7}),
        ),
    }
    result = ComplianceGeometryTools().zonal_attribute_threshold(
        objects_layer="objects",
        zones_layer="zones",
        object_attribute="floors",
        operator="<=",
        constant_threshold=None,
        zone_threshold_attribute="limit",
        join_predicate="within",
        result_mode="both",
        layers=layers,
        restriction_id="r4",
    )
    assert result["summary"]["violated_objects"] == 1
    assert result["evidence"][0]["threshold"] == 7
    assert len(result["evidence"][0]["zone_refs"]) == 2


def test_zonal_attribute_equality_with_conflicting_zone_thresholds_is_unchecked():
    outer = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    layers = {
        "objects": _fc((Point(0, 0), {"physical_object_id": 1, "floors": 8})),
        "zones": _fc(
            (outer, {"physical_object_id": 10, "limit": 8}),
            (outer, {"physical_object_id": 11, "limit": 9}),
        ),
    }
    result = ComplianceGeometryTools().zonal_attribute_threshold(
        objects_layer="objects",
        zones_layer="zones",
        object_attribute="floors",
        operator="==",
        constant_threshold=None,
        zone_threshold_attribute="limit",
        join_predicate="within",
        result_mode="both",
        layers=layers,
        restriction_id="r4-eq",
    )
    assert result["coverage"]["checked_objects"] == 0
    assert result["coverage"]["unchecked_objects"] == 1
    assert result["evidence"] == []


def test_zonal_ratio_unions_overlaps_before_area_sum():
    zone = Polygon([(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)])
    half = Polygon([(0, 0), (0.006, 0), (0.006, 0.01), (0, 0.01)])
    overlap = Polygon([(0.004, 0), (0.006, 0), (0.006, 0.01), (0.004, 0.01)])
    layers = {
        "zones": _fc((zone, {"physical_object_id": 1})),
        "numerator": _fc(
            (half, {"physical_object_id": 2}),
            (overlap, {"physical_object_id": 3}),
        ),
    }
    result = ComplianceGeometryTools().zonal_ratio(
        zones_layer="zones",
        numerator_layer="numerator",
        operator="<=",
        threshold=65,
        result_mode="both",
        invalid_geometry_policy="repair",
        layers=layers,
        restriction_id="r5",
    )
    assert result["summary"] == {"violated_objects": 0, "passed_objects": 1}
    assert 59 < result["evidence"][0]["measured_value"] < 61


def test_zonal_ratio_zero_area_zone_is_unchecked():
    zero_area = Polygon([(0, 0), (0.01, 0), (0.02, 0), (0, 0)])
    layers = {
        "zones": _fc((zero_area, {"physical_object_id": 1})),
        "numerator": _fc(
            (Polygon([(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)]), {"physical_object_id": 2})
        ),
    }
    result = ComplianceGeometryTools().zonal_ratio(
        zones_layer="zones",
        numerator_layer="numerator",
        operator="<=",
        threshold=50,
        result_mode="both",
        invalid_geometry_policy="repair",
        layers=layers,
        restriction_id="r5-zero",
    )
    assert result["coverage"]["checked_objects"] == 0
    assert result["coverage"]["unchecked_objects"] == 1
