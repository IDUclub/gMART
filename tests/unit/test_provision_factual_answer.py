"""Numerical interpretation regressions found in the live 200+100 evaluation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.agents.services.pipeline_state import PipelineStatus
from src.agents.services.provision.provision_context import ProvisionContextBuilder
from src.agents.services.provision.provsion_service import ProvisionService
from src.agents.services.service_entities.provision_plan import ProvisionPlan


def test_answer_separates_population_coverage_from_mean_building_index():
    summary = dict(
        total_capacity=3060,
        total_demand=3637,
        satisfied_demand_within=3060,
        unsatisfied_demand=577,
        deficit=577,
        surplus=0,
        balance=-577,
        services_count=4,
        average_provision_value=0.874,
        median_provision_value=1.0,
    )
    answer = ProvisionContextBuilder().build_provision_answer(summary, "Школа")
    assert "84.1%" in answer
    assert "Средняя обеспеченность по зданиям: 0.874" in answer
    assert "Медианная обеспеченность по зданиям: 1.0" in answer
    assert "половина школ" not in answer
    assert "577" in answer


def test_missing_or_zero_demand_does_not_invent_percentage():
    builder = ProvisionContextBuilder()
    for summary in ({}, {"total_demand": 0, "satisfied_demand_within": 0}):
        answer = builder.build_provision_answer(summary, "Школа")
        assert "%" not in answer


def test_failed_service_is_reported_without_zero_deficit():
    answer = ProvisionContextBuilder().build_summary_answer(
        {
            "services": {
                "1": {"name": "Школа", "error": "Недоступен расчёт"},
                "2": {
                    "name": "Детский сад",
                    "summary": {"total_demand": 100, "deficit": 20},
                },
            }
        }
    )
    assert "Школа: расчёт не выполнен" in answer
    assert "Детский сад" in answer and "20" in answer
    assert "Дефицит (чел): 0" not in answer


async def test_missing_calculation_emits_error_instead_of_completed_answer():
    service = object.__new__(ProvisionService)
    service.state_store = SimpleNamespace(set_status=AsyncMock())
    service._buf = AsyncMock(side_effect=lambda _id, event: event)

    async def calculation(*args):
        args[-1].append(
            SimpleNamespace(
                data={
                    "services": {
                        "22": {"name": "Школа", "error": "Нет расчётных данных"}
                    }
                },
                tool_calls=[],
            )
        )
        if False:
            yield {}

    service._checkpointed_provision_step = calculation
    events = [
        e
        async for e in service._run_single_provision(
            "r",
            object(),
            object(),
            ["token"],
            {},
            ProvisionPlan(mode="provision", service_name="Школа"),
            {"Школа": 22},
            "m",
            0,
            "Обеспеченность школами",
            772,
            [],
        )
    ]
    assert any(e["type"] == "error" for e in events)
    assert not any(e["type"] == "chunk" and e["content"]["done"] for e in events)
    service.state_store.set_status.assert_awaited_with("r", PipelineStatus.FAILED)
