"""Checked objects without any violation are merged into one map layer."""

from __future__ import annotations

from src.agents.services.compilance.compliance_layers import passed_objects_layer
from src.agents.services.service_entities.compliance import (
    ComplianceResult,
    ComplianceSummary,
    VerificationCoverage,
)


def _feature(object_id, **properties):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [30.0, 60.0]},
        "properties": {
            "name": f"Дом {object_id}",
            "object_ref": {"id": f"physical_object/{object_id}"},
            "compliance_status": "passed",
            "verification_status": "complete",
            "restriction_id": "hash",
            "compliance_evidence": [{"violated": False}],
            **properties,
        },
    }


def _result(clause, *, passed=(), violated=(), status=None, equivalent=()):
    violated_count = len(violated)
    status = status or ("violated" if violated_count else "passed")
    executed = status != "unknown"
    checked = len(passed) + violated_count
    source = {"document_name": "СП 42.13330.2016", "clause_number": clause}
    return ComplianceResult(
        restriction_id=clause,
        template="distance_from_source",
        template_version=1,
        verification_status="complete" if executed else "unverifiable",
        compliance_status=status,
        coverage=VerificationCoverage(
            applicable_objects=checked,
            checked_objects=checked,
            unchecked_objects=0,
            fill_rate=1,
        ),
        summary=ComplianceSummary(
            violated_objects=violated_count, passed_objects=len(passed)
        ),
        source={
            **source,
            "equivalent_sources": [
                source,
                *({**source, "clause_number": item} for item in equivalent),
            ],
        },
        passed_features={
            "type": "FeatureCollection",
            "features": [_feature(i) for i in passed],
        },
        violated_features={
            "type": "FeatureCollection",
            "features": [_feature(i, compliance_status="violated") for i in violated],
        },
    )


def test_objects_are_merged_once_with_every_norm_they_passed():
    layer = passed_objects_layer(
        [
            _result("7.1", passed=[1, 2], equivalent=["7.3"]),
            _result("8.2", passed=[2, 3], violated=[4]),
        ]
    )

    by_name = {f["properties"]["name"]: f["properties"] for f in layer["features"]}
    assert set(by_name) == {"Дом 1", "Дом 2", "Дом 3"}
    assert by_name["Дом 2"]["passed_norms"] == [
        "СП 42.13330.2016, п. 7.1",
        "СП 42.13330.2016, п. 7.3",
        "СП 42.13330.2016, п. 8.2",
    ]
    assert by_name["Дом 3"]["passed_norms"] == ["СП 42.13330.2016, п. 8.2"]
    for properties in by_name.values():
        assert properties["compliance_status"] == "passed"
        assert "restriction_id" not in properties
        assert "compliance_evidence" not in properties
        assert "verification_status" not in properties


def test_an_object_violating_any_norm_is_left_out():
    layer = passed_objects_layer(
        [_result("7.1", passed=[1, 2]), _result("8.2", passed=[1], violated=[2])]
    )

    assert [f["properties"]["name"] for f in layer["features"]] == ["Дом 1"]


def test_no_layer_without_compliant_objects():
    assert passed_objects_layer([]) is None
    assert passed_objects_layer([_result("7.1", violated=[1])]) is None
    assert passed_objects_layer([_result("7.1", status="unknown")]) is None
