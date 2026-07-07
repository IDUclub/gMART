"""Unit tests for ProvisionContextBuilder strict tables and LLM contexts."""

from __future__ import annotations

from src.agents.services.provision_context import ProvisionContextBuilder

SUMMARY_SCHOOLS = {
    "services_count": 3,
    "total_capacity": 1200,
    "total_demand": 1450,
    "satisfied_demand_within": 900,
    "satisfied_demand_without": 300,
    "unsatisfied_demand": 250,
    "balance": -250,
    "deficit": 250,
    "surplus": 0,
    "average_provision_value": 0.7123,
    "median_provision_value": 0.8,
}

SUMMARY_KINDERGARTENS = {
    "services_count": 5,
    "total_capacity": 800,
    "total_demand": 620,
    "satisfied_demand_within": 600,
    "satisfied_demand_without": 20,
    "unsatisfied_demand": 0,
    "balance": 180,
    "deficit": 0,
    "surplus": 180,
    "average_provision_value": 0.95,
    "median_provision_value": 1.0,
}

MULTI_RESULT = {
    "services": {
        "21": {"name": "Детские сады", "summary": SUMMARY_KINDERGARTENS},
        "22": {"name": "Школы", "summary": SUMMARY_SCHOOLS},
        "35": {"name": "Аптеки", "summary": None, "error": "No services found"},
    }
}


def test_summary_table_sorted_by_deficit_with_strict_columns():
    table = ProvisionContextBuilder().build_summary_table(MULTI_RESULT)
    assert table["name"] == "provision_summary"
    assert [column["key"] for column in table["columns"]] == [
        "service",
        "capacity",
        "demand",
        "deficit",
        "surplus",
        "balance",
    ]
    assert [row["service"] for row in table["rows"]] == ["Школы", "Детские сады"]
    schools = table["rows"][0]
    assert schools["deficit"] == 250
    assert schools["balance"] == -250


def test_provision_metrics_table_uses_fixed_labels():
    table = ProvisionContextBuilder().build_provision_metrics_table(
        SUMMARY_SCHOOLS, "Школы"
    )
    assert table["name"] == "provision_metrics"
    assert [column["key"] for column in table["columns"]] == ["metric", "value"]
    metrics = {row["metric"]: row["value"] for row in table["rows"]}
    assert metrics["Дефицит (чел)"] == 250
    assert metrics["Вместимость (чел)"] == 1200
    assert metrics["Средняя обеспеченность (0–1)"] == 0.712


def test_effects_pivot_table_from_pivot():
    effects_result = {
        "pivot": {
            "sum_absolute_total": 100,
            "average_index_total": 0.51234,
            "unknown_key": 1,
        }
    }
    table = ProvisionContextBuilder().build_effects_pivot_table(effects_result, "Школы")
    assert table["name"] == "effects_pivot"
    metrics = {row["metric"]: row["value"] for row in table["rows"]}
    assert metrics["Суммарное абсолютное покрытие (всего)"] == 100
    assert metrics["Средний индекс покрытия (всего)"] == 0.512
    assert len(metrics) == 2  # unknown keys are not rendered


def test_effects_pivot_table_none_without_pivot():
    builder = ProvisionContextBuilder()
    assert builder.build_effects_pivot_table({}, "Школы") is None
    assert builder.build_effects_pivot_table({"pivot": {}}, "Школы") is None


def test_summary_context_mentions_values_and_failures():
    context = ProvisionContextBuilder().build_summary_context(MULTI_RESULT)
    assert "Школы" in context
    assert "Детские сады" in context
    assert "Аптеки" in context  # failed service is reported
    assert "250" in context


def test_provision_context_lists_metrics():
    context = ProvisionContextBuilder().build_provision_context(
        SUMMARY_SCHOOLS, "Школы"
    )
    assert "«Школы»" in context
    assert "Дефицит (чел): 250" in context
