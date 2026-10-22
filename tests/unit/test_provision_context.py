"""Unit tests for ProvisionContextBuilder strict tables and LLM contexts."""

from __future__ import annotations

import json

import pytest

from src.agents.services.provision.provision_context import ProvisionContextBuilder

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
        "status",
    ]
    assert [row["service"] for row in table["rows"]] == [
        "Школы",
        "Детские сады",
        "Аптеки",
    ]
    assert table["rows"][-1]["status"] == "Нет данных"
    assert table["rows"][-1]["capacity"] is None
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


def effects(delta=0.2):
    metrics = {
        "provision_before": 0.4,
        "provision_after": 0.4 + delta,
        "provision_delta": delta,
        "deficit_before": 100,
        "deficit_after": 50,
        "deficit_delta": -50,
        "unmet_demand_after": 60,
        "total_capacity_after": 200,
        "unused_metric": "large unwanted context",
    }
    fc = {"type": "FeatureCollection", "features": []}
    return {
        "text_pivot": json.dumps(
            {"all": metrics, "project": metrics, "context": {"secret": 123}}
        ),
        "pivot": {"large": "raw pivot"},
        "before_prove_data": {"buildings": fc},
        "after_prove_data": {"services": fc},
        "effects": fc,
    }


def test_table_columns_sort_rounding_and_no_data():
    builder = ProvisionContextBuilder()
    data = {
        "services": {
            "1": {"name": "worse", **effects(-0.123456)},
            "2": {"name": "missing", "text_pivot": "not JSON"},
            "3": {"name": "better", **effects(0.234567)},
            "4": {"name": "unchanged", **effects(0)},
        }
    }
    table = builder.build_effects_summary_table(data)
    assert table["name"] == "provision_effects_summary"
    assert table["title"] == "Сравнение эффектов обеспеченности по сервисам"
    assert [col["label"] for col in table["columns"]] == [
        "Сервис",
        "Обеспеченность до",
        "Обеспеченность после",
        "Изменение обеспеченности",
        "Дефицит до (чел)",
        "Дефицит после (чел)",
        "Изменение дефицита (чел)",
        "Статус",
    ]
    rows = table["rows"]
    assert [row["service"] for row in rows] == [
        "better",
        "unchanged",
        "worse",
        "missing",
    ]
    assert rows[0]["provision_delta"] == 0.235
    assert rows[1]["provision_delta"] == 0
    assert rows[2]["provision_delta"] == -0.123
    assert rows[0]["status"] == "Рассчитано"
    assert rows[-1]["status"] == "Нет данных"
    assert all(rows[-1][col["key"]] is None for col in table["columns"][1:-1])
    del data["services"]["2"]
    assert all(
        col["key"] != "status"
        for col in builder.build_effects_summary_table(data)["columns"]
    )


@pytest.mark.parametrize(
    "pivot",
    [
        None,
        "broken",
        "[]",
        "null",
        "{}",
        '{"all":{}}',
        '{"all":{"provision_delta":"bad"}}',
    ],
)
def test_missing_or_unparseable_metrics_are_not_zero(pivot):
    data = {"services": {"1": {"name": "empty", "text_pivot": pivot}}}
    builder = ProvisionContextBuilder()
    row = builder.build_effects_summary_table(data)["rows"][0]
    assert row["status"] == "Нет данных"
    assert row["provision_delta"] is None
    assert (
        json.loads(builder.build_effects_summary_context(data))[0]["metrics"]
        == "нет данных"
    )


def test_compact_context_has_only_key_all_project_metrics_and_no_errors():
    data = {
        "services": {
            "1": {"name": "Школы", **effects()},
            "2": {"name": "Сады", "error": "PRIVATE_INTERNAL_ERROR"},
        }
    }
    context = ProvisionContextBuilder().build_effects_summary_context(data)
    values = json.loads(context)
    assert set(values[0]["metrics"]) == {"all", "project"}
    assert values[0]["metrics"]["all"]["provision_delta"] == 0.2
    assert values[1]["metrics"] == "нет данных"
    for excluded in (
        "PRIVATE_INTERNAL_ERROR",
        "secret",
        "unused_metric",
        "raw pivot",
        "features",
        '"1"',
    ):
        assert excluded not in context
    assert len(context) < 700
