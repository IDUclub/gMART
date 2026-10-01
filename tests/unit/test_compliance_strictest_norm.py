"""Norms with conditions: the strictest value is applied and the report says so."""

from src.agents.services.compilance.compliance_executor import (
    ComplianceTemplateExecutor,
)
from src.agents.services.compilance.compliance_report import build_compliance_report
from src.agents.services.compilance.compliance_result_harness import (
    ComplianceResultHarness,
)
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.service_entities.compliance import STRICTEST_NORM_WARNING
from tests.unit.test_compliance_executor import FakeMcpClient
from tests.unit.test_compliance_executor import _plan as _executor_plan
from tests.unit.test_compliance_report import _result, _summary

APPLICABILITY = {
    "mode": "strictest_variant",
    "conditions": ["для городских поселений", "для сельских поселений"],
    "variants": ["50 м — городские поселения", "25 м — сельские поселения"],
    "applied": "50 м",
}


def _strictest(result):
    result["source"]["check_plan"]["applicability"] = APPLICABILITY
    return result


async def test_executor_marks_a_strictest_norm():
    plan = {**_executor_plan(), "applicability": APPLICABILITY}
    execution = await ComplianceTemplateExecutor().execute(FakeMcpClient(), plan, 772)
    assert execution.result.compliance_status == "violated"
    assert STRICTEST_NORM_WARNING in execution.result.warnings
    assert execution.result.source["check_plan"]["applicability"] == APPLICABILITY

    plain = await ComplianceTemplateExecutor().execute(
        FakeMcpClient(), _executor_plan(), 772
    )
    assert STRICTEST_NORM_WARNING not in plain.result.warnings


def test_report_states_the_strictest_norm_and_its_conditions():
    report = build_compliance_report(
        _summary([_strictest(_result("r1", "violated", violated=2))])
    )
    assert "| Проверены по самой строгой норме (проверьте условия) | 1 |" in report
    assert "необходимо проверить на дополнительные условия" in report
    assert "- **Применена самая строгая норма:** 50 м." in report
    assert "Условия пункта: для городских поселений; для сельских поселений" in report
    assert "Варианты нормы: 50 м — городские поселения" in report


def test_report_without_strictest_norms_has_no_notice():
    report = build_compliance_report(_summary([_result("r1", "violated", violated=2)]))
    assert "самой строгой" not in report and "самая строгая" not in report


def test_summary_text_and_answers_name_the_strictest_norm():
    summary = {
        "total_norms": 1,
        "violated_norms": 1,
        "passed_norms": 0,
        "unverifiable_norms": 0,
        "unsupported_norms": 0,
        "partial_norms": 0,
        "results": [_strictest(_result("r1", "violated", violated=3))],
    }
    text = RestrictionParserService._compliance_summary_text(summary)
    assert "По самой строгой норме проверено: 1." in text
    assert (
        "Применена самая строгая норма (50 м); проверьте дополнительные условия" in text
    )

    compact = ComplianceResultHarness._compact_result(summary["results"][0])
    assert compact["applicability"] == APPLICABILITY
    fallback = ComplianceResultHarness.fallback_answer(summary)
    assert "применена самая строгая норма (50 м)" in fallback


def test_report_names_the_population_source_of_a_provision_check():
    result = _result("r1", "violated", violated=2)
    result["warnings"] = [
        "population:source=housing_stock",
        "population:residents=1013",
        "population_indicator_ignored: Urban API population 204314 ...",
    ]
    report = build_compliance_report(_summary([result]))
    assert "- **Население сценария:** 1013 чел. — ёмкость жилого фонда" in report
    assert "индикатор Urban API не совпадает с жилым фондом" in report
