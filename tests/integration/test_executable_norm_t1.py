"""Contract integration: NormGraph-shaped CheckPlan → gMART → IDU geometry."""

from src.agents.schema.restrictions_response import RestrictionsResponse
from src.agents.services.compliance_executor import ComplianceTemplateExecutor
from src.idu_mcp.tools_services.compliance_geometry import ComplianceGeometryTools


class InMemoryIduMcp:
    async def execute_tool(self, name, arguments, meta=None):
        if name == "GetServices":
            return {"Школа": _fc(30.0, service_id=1)}
        if name == "GetPhysicalObjects":
            return {
                "Жилой дом": {
                    "type": "FeatureCollection",
                    "features": [
                        _feature(30.0001, physical_object_id=1),
                        _feature(30.01, physical_object_id=2),
                    ],
                    "meta": {
                        "complete": True,
                        "truncated": False,
                        "revision": "scenario:772:physical_object:4",
                    },
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
    return {
        "type": "FeatureCollection",
        "features": [_feature(x, **properties)],
        "meta": {
            "complete": True,
            "truncated": False,
            "revision": "scenario:772:service:1",
        },
    }


async def test_normgraph_plan_executes_as_structured_t1_result():
    normgraph_hit = {
        "id": "norm-r1",
        "check_plan": {
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
            "source": {
                "restriction_id": "norm-r1",
                "document_name": "СП test",
                "clause_number": "5.5",
            },
            "planner_status": "reviewed",
        },
    }
    execution = await ComplianceTemplateExecutor().execute(
        InMemoryIduMcp(), normgraph_hit["check_plan"], 772
    )
    result = execution.result
    assert result.verification_status == "complete"
    assert result.compliance_status == "violated"
    assert result.summary.model_dump() == {
        "violated_objects": 1,
        "passed_objects": 1,
    }
    assert result.evidence[0].input_revision.startswith("sha256:")
    RestrictionsResponse.model_validate(
        {"type": "compliance_result", "content": result.model_dump(mode="json")}
    )
