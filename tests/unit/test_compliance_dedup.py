from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.services.compilance.compliance_dedup import group_checks


def plan(rid="a", distance=100):
    return dict(
        schema_version="1.0",
        template="distance_from_source",
        template_version=1,
        planner_status="auto",
        source=dict(restriction_id=rid, document_name=rid),
        params=dict(
            source_layer="source",
            targets=["targets"],
            distance_m=distance,
            geometry_mode="buffered",
            predicate="intersects",
            violation_when="matched",
        ),
        declared_requirements=dict(
            layers=[
                dict(role="source", entity="Школа", entity_type="service"),
                dict(role="targets", entity="Жилой дом", entity_type="physical_object"),
            ],
            attributes=[],
        ),
    )


def client():
    return SimpleNamespace(
        resolve_urban_entity_types=AsyncMock(
            return_value={
                "service": {
                    "Школа": {"found": True, "canonical_name": "Школа"},
                    "школ": {"found": True, "canonical_name": "Школа"},
                },
                "physical_object": {
                    "Жилой дом": {"found": True, "canonical_name": "Жилой дом"}
                },
            }
        )
    )


async def test_canonical_entities_roles_and_default_params_deduplicate():
    a, b = plan(), plan("b")
    b["declared_requirements"]["layers"][0].update(role="school", entity="школ")
    b["params"].update(source_layer="school", result_mode="both")
    b["declared_requirements"]["layers"].reverse()
    groups = await group_checks([a, b], client(), 772)
    assert len(groups) == 1
    assert [s["restriction_id"] for s in groups[0].sources] == ["a", "b"]
    assert b["declared_requirements"]["layers"][1]["entity"] == "школ"


@pytest.mark.parametrize(
    "change", ["distance", "direction", "geometry", "predicate", "result_mode"]
)
async def test_different_semantics_are_not_merged(change):
    a, b = plan(), plan("b")
    if change == "distance":
        b["params"]["distance_m"] = 200
    elif change == "direction":
        b["params"].update(source_layer="targets", targets=["source"])
    elif change == "geometry":
        b["declared_requirements"]["layers"][0]["geometry_types"] = ["Polygon"]
    elif change == "predicate":
        b["params"]["predicate"] = "within"
    else:
        b["params"]["result_mode"] = "violated"
    assert len(await group_checks([a, b], client(), 772)) == 2


async def test_catalog_outage_does_not_drop_checks():
    c = client()
    c.resolve_urban_entity_types.side_effect = RuntimeError("offline")
    assert len(await group_checks([plan(), plan("b")], c, 772)) == 2


async def test_unknown_entity_does_not_deduplicate():
    c = client()
    c.resolve_urban_entity_types.return_value = {}
    assert len(await group_checks([plan(), plan("b")], c, 772)) == 2


async def test_pipeline_executes_duplicates_once_and_preserves_sources():
    from src.agents.services.restriction.restriction_parser_service import (
        RestrictionParserService,
    )
    from src.agents.services.service_entities.compliance import ComplianceResult

    service = object.__new__(RestrictionParserService)
    service._buf = AsyncMock(side_effect=lambda rid, event: event)
    service.state_store = SimpleNamespace(
        save_checkpoint=AsyncMock(), set_status=AsyncMock()
    )
    result = ComplianceResult(
        restriction_id="a",
        template="distance_from_source",
        template_version=1,
        verification_status="complete",
        compliance_status="passed",
        coverage=dict(
            applicable_objects=1, checked_objects=1, unchecked_objects=0, fill_rate=1
        ),
        summary=dict(violated_objects=0, passed_objects=1),
        source=dict(restriction_id="a"),
    )
    service.compliance_executor = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(result=result, tool_calls=[], timings_ms={})
        )
    )
    events = [
        event
        async for event in service._run_executable_compliance(
            mcp_client=client(),
            request_id="request",
            scenario_id=772,
            restrictions=[{"check_plan": plan()}, {"check_plan": plan("b")}],
            checkpoint={},
        )
    ]
    service.compliance_executor.execute.assert_awaited_once()
    assert sum(e["type"] == "check_plan" for e in events) == 1
    assert not any(e["type"] == "feature_collection" for e in events)
    summary = next(e["content"] for e in events if e["type"] == "compliance_summary")
    assert summary["total_norms"] == 1 and summary["duplicate_checks"] == 1
    assert [s["restriction_id"] for s in summary["equivalent_sources"]["a"]] == [
        "a",
        "b",
    ]


@pytest.mark.parametrize("change", ["field", "fill_rate", "candidate_order"])
async def test_attribute_selection_and_coverage_policies_remain_distinct(change):
    a = plan()
    a["template"] = "zonal_attribute_threshold"
    a["params"] = dict(
        objects_layer="targets",
        zones_layer="zones",
        attribute_role="floors",
        operator="<=",
        threshold_source=dict(kind="constant", value=5, unit="floors"),
    )
    a["declared_requirements"]["layers"][0] = dict(
        role="zones", entity="functional_zones", entity_type="functional_zone"
    )
    a["declared_requirements"]["attributes"] = [
        dict(
            role="floors",
            on="targets",
            min_fill_rate=0,
            accepts=[
                dict(field="building.floors", unit="floors", quality="direct"),
                dict(field="levels", unit="floors", quality="direct"),
            ],
        )
    ]
    b = deepcopy(a)
    b["source"]["restriction_id"] = "b"
    attr = b["declared_requirements"]["attributes"][0]
    if change == "field":
        attr["accepts"][0]["field"] = "floors"
    elif change == "fill_rate":
        attr["min_fill_rate"] = 1
    else:
        attr["accepts"].reverse()
    assert len(await group_checks([a, b], client(), 772)) == 2
