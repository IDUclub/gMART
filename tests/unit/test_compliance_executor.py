from src.agents.services.compliance_executor import ComplianceTemplateExecutor
from src.idu_mcp.tools_services.compliance_geometry import ComplianceGeometryTools


class FakeMcpClient:
    def __init__(self):
        self.calls = []

    async def execute_tool(self, name, arguments, meta=None):
        self.calls.append((name, arguments, meta))
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
