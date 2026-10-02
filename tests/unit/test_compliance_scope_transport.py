"""Norms stated for one building type, and transport accessibility estimates."""

import pytest

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
from src.agents.services.service_entities.compliance import (
    SCOPE_WARNING,
    TRANSPORT_WARNING,
    CheckPlan,
)
from src.idu_mcp.tools_services.compliance_geometry import ComplianceGeometryTools
from tests.unit.test_compliance_executor import FakeMcpClient, _feature
from tests.unit.test_compliance_executor import _plan as _executor_plan
from tests.unit.test_compliance_report import _result, _summary


class FloorsMcpClient(FakeMcpClient):
    """Houses of 9 and 2 floors (the first violates), plus one without floors."""

    def __init__(self, *, unknown_floors=False, **kwargs):
        super().__init__(**kwargs)
        self.unknown_floors = unknown_floors

    async def execute_tool(self, name, arguments, meta=None):
        if name != "GetPhysicalObjects":
            return await super().execute_tool(name, arguments, meta)
        self.calls.append((name, arguments, meta))
        features = [
            _feature(30.0001, physical_object_id=1, building={"floors": 9}),
            _feature(30.01, physical_object_id=2, building={"floors": "2"}),
        ]
        if self.unknown_floors:
            features.append(_feature(30.0002, physical_object_id=3, building={}))
        return {"Жилой дом": {"type": "FeatureCollection", "features": features}}


def _scoped_plan(minimum=None, maximum=None):
    plan = _executor_plan()
    plan["declared_requirements"]["attributes"] = [
        {
            "role": "scope_floors",
            "on": "targets",
            "required": True,
            "min_fill_rate": 0,
            "accepts": [
                {"field": "building.floors", "unit": "floors", "quality": "direct"}
            ],
        }
    ]
    plan["scope"] = {
        "layer": "targets",
        "attribute": "scope_floors",
        "min": minimum,
        "max": maximum,
        "condition": "многоэтажная жилая застройка",
    }
    return plan


async def test_only_buildings_of_the_condition_are_checked():
    execution = await ComplianceTemplateExecutor().execute(
        FloorsMcpClient(), _scoped_plan(minimum=5), 772
    )
    result = execution.result
    assert result.compliance_status == "violated"
    assert result.coverage.applicable_objects == 1
    assert result.summary.violated_objects == 1
    assert SCOPE_WARNING in result.warnings
    assert "scope:outside_condition=1" in result.warnings

    execution = await ComplianceTemplateExecutor().execute(
        FloorsMcpClient(), _scoped_plan(maximum=4), 772
    )
    assert execution.result.compliance_status == "passed"
    assert execution.result.coverage.checked_objects == 1


async def test_buildings_without_floors_stay_unchecked():
    execution = await ComplianceTemplateExecutor().execute(
        FloorsMcpClient(unknown_floors=True), _scoped_plan(minimum=5), 772
    )
    result = execution.result
    assert result.verification_status == "partial"
    assert result.coverage.unchecked_objects == 1
    assert result.coverage.checked_objects == 1
    assert "scope:without_value=1" in result.warnings


def test_scope_must_name_a_declared_layer_and_attribute():
    plan = _scoped_plan(minimum=5)
    plan["scope"]["attribute"] = "unknown"
    with pytest.raises(ValueError):
        CheckPlan.model_validate(plan)
    plan = _scoped_plan()
    with pytest.raises(ValueError):
        CheckPlan.model_validate(plan)


def _scoped_result():
    result = _result("r1", "violated", violated=2)
    result["source"]["check_plan"]["scope"] = {
        "layer": "objects",
        "attribute": "scope_floors",
        "min": 5,
        "max": None,
        "condition": "многоэтажная, среднеэтажная жилая застройка",
    }
    result["warnings"] = [
        SCOPE_WARNING,
        "scope:outside_condition=4",
        "scope:without_value=1",
    ]
    return result


def _transport_result():
    result = _result("r2", "violated", violated=1)
    result["template"] = "accessibility_within"
    result["source"]["check_plan"]["params"] = {
        "objects_layer": "objects",
        "required_neighbor_layers": ["neighbors"],
        "limit": {"kind": "time", "minutes": 30},
        "speed_m_per_min": 25_000 / 60,
        "detour_factor": 1.3,
        "measurement": "buffer_v1",
        "mode": "transport",
        "minimum_neighbors": 1,
        "result_mode": "both",
    }
    result["warnings"] = [TRANSPORT_WARNING]
    return result


def test_report_names_the_condition_and_the_transport_estimate():
    report = build_compliance_report(_summary([_scoped_result(), _transport_result()]))
    assert (
        "| Проверены только на объектах, подходящих под условие пункта | 1 |" in report
    )
    assert "| Транспортная доступность оценена приближённо | 1 |" in report
    assert (
        "- **Проверены только объекты, к которым относится значение:** "
        "многоэтажная, среднеэтажная жилая застройка — объекты этажностью от 5 и "
        "выше; вне условия — 4, без этажности (не проверены) — 1"
    ) in report
    assert (
        "- **Транспортная доступность оценена приближённо:** радиус 9615 м по прямой "
        "при средней скорости транспорта 25 км/ч"
    ) in report
    assert "Вид доступности: транспортная (приближённая оценка)" in report


def test_summary_text_and_answers_mention_condition_and_estimate():
    summary = {
        "total_norms": 2,
        "violated_norms": 2,
        "passed_norms": 0,
        "unverifiable_norms": 0,
        "unsupported_norms": 0,
        "partial_norms": 0,
        "results": [_scoped_result(), _transport_result()],
    }
    text = RestrictionParserService._compliance_summary_text(summary)
    assert "Транспортная доступность оценена приближённо: радиусом" in text
    assert "Проверены только многоэтажная, среднеэтажная жилая застройка" in text
    compact = ComplianceResultHarness._compact_result(summary["results"][0])
    assert compact["scope"].startswith("многоэтажная, среднеэтажная")
    fallback = ComplianceResultHarness.fallback_answer(summary)
    assert "транспортная доступность оценена приближённо" in fallback


class AccessibilityMcpClient(FakeMcpClient):
    async def execute_tool(self, name, arguments, meta=None):
        if name == "CheckAccessibilityWithin":
            self.calls.append((name, arguments, meta))
            return ComplianceGeometryTools().accessibility_within(**arguments)
        return await super().execute_tool(name, arguments, meta)


def _accessibility_plan(mode):
    plan = _executor_plan()
    plan["template"] = "accessibility_within"
    plan["params"] = {
        "objects_layer": "objects",
        "required_neighbor_layers": ["neighbors"],
        "limit": {"kind": "time", "minutes": 1},
        "speed_m_per_min": 25_000 / 60 if mode == "transport" else 80.0,
        "detour_factor": 1.3,
        "measurement": "buffer_v1",
        "mode": mode,
        "minimum_neighbors": 1,
        "result_mode": "both",
    }
    layers = plan["declared_requirements"]["layers"]
    layers[0]["role"], layers[1]["role"] = "neighbors", "objects"
    return plan


async def test_executor_flags_a_transport_estimate():
    client = AccessibilityMcpClient()
    execution = await ComplianceTemplateExecutor().execute(
        client, _accessibility_plan("transport"), 772
    )
    assert execution.result.compliance_status in {"violated", "passed"}
    assert TRANSPORT_WARNING in execution.result.warnings
    radius = next(args for name, args, _ in client.calls if name == "CheckAccessibilityWithin")[
        "radius_m"
    ]
    assert radius == pytest.approx(25_000 / 60 / 1.3)

    walking = await ComplianceTemplateExecutor().execute(
        AccessibilityMcpClient(), _accessibility_plan("walk"), 772
    )
    assert TRANSPORT_WARNING not in walking.result.warnings
