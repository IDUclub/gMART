from src.agents.services.compliance_metrics import ComplianceMetrics
from src.agents.services.service_entities.compliance import (
    ComplianceResult,
    ComplianceSummary,
    ResolvedRequirement,
    VerificationCoverage,
)


def test_compliance_metrics_aggregate_without_request_identity():
    metrics = ComplianceMetrics()
    result = ComplianceResult(
        restriction_id="r-1",
        template="distance_table",
        template_version=1,
        verification_status="partial",
        compliance_status="passed",
        coverage=VerificationCoverage(
            applicable_objects=2,
            checked_objects=1,
            unchecked_objects=1,
            fill_rate=0.5,
        ),
        summary=ComplianceSummary(violated_objects=0, passed_objects=1),
        resolved_requirements=[
            ResolvedRequirement(
                role="floors",
                requirement_type="attribute",
                resolved=True,
                layer="houses",
                field="floors",
                unit="floor",
                quality="direct",
                fill_rate=0.5,
            )
        ],
    )

    metrics.observe(
        result,
        planner_status="reviewed",
        timings_ms={"requirements_resolution": 10, "template_execution": 20},
    )
    metrics.observe_downstream_error("idu_mcp")

    snapshot = metrics.snapshot()
    assert snapshot["norms_by_template"] == {"distance_table@1": 1}
    assert snapshot["verification_status"] == {"partial": 1}
    assert snapshot["planner_status"] == {"reviewed": 1}
    assert snapshot["duration_ms"]["template_execution"]["avg"] == 20
    assert snapshot["fill_rate_by_role"]["floors"]["avg"] == 0.5
    assert snapshot["downstream_errors"] == {"urban_api": 0, "idu_mcp": 1}
    assert "restriction_id" not in snapshot
