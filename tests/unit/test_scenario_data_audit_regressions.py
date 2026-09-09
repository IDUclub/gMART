"""Regressions from real Urban MCP queries; no live boundaries."""

import json

import pytest

from agents.services.scenario_data.scenario_data_evaluator import wants_layers
from agents.services.scenario_data.scenario_data_indicators import (
    IndicatorRequest,
    validate_request,
)
from agents.services.scenario_data.scenario_data_selection import may_select_entities
from agents.services.scenario_data.scenario_data_service import ScenarioDataService


def test_literal_indicator_pair_cannot_be_substituted():
    facts = [
        {"name": name}
        for name in (
            "Население",
            "Численность населения",
            "Срок рекультивации территории",
        )
    ]
    request = IndicatorRequest(
        operation="values",
        names=["Численность населения", "Срок рекультивации территории"],
        missing=[],
    )
    with pytest.raises(ValueError):
        validate_request(
            request,
            facts,
            "Покажи «Население» и «Срок рекультивации территории» сценария 772",
        )


def test_count_request_stays_on_verified_entity_path():
    assert may_select_entities(
        "Посчитай сервисов типа «Библиотека» в выбранном сценарии"
    )


@pytest.mark.parametrize(
    "query",
    [
        "Покажи все сервисы территории 1955",
        "Покажи все типы физических объектов в общем справочнике",
        "Покажи карточку физического объекта 586200",
    ],
)
def test_non_scenario_type_queries_do_not_select_one_type(query):
    assert not may_select_entities(query)


def test_map_request_preserves_output():
    assert wants_layers("Нужна карта сервисов типа «Библиотека»")


def test_root_card_preserves_zero_values():
    table = ScenarioDataService._table_from_result(
        {"project_id": 604, "preparation": 0, "implementation": 0},
        name="project",
        title="Проект",
    )
    assert table is not None
    assert table["rows"] == [{"project_id": 604, "preparation": 0, "implementation": 0}]


def test_geometry_bearing_card_is_not_mistaken_for_geojson_feature():
    row = {
        "territory_id": 17,
        "name": "Город",
        "properties": {},
        "geometry": {"type": "Point", "coordinates": [30, 60]},
    }
    table = ScenarioDataService._table_from_result(row, name="card", title="Город")
    assert table["rows"][0]["territory_id"] == 17
    assert table["rows"][0]["name"] == "Город"


def test_nested_source_fields_are_not_cut_into_invalid_json():
    value = {"id": 17, "name": "a" * 1200, "capacity": 0}
    assert json.loads(ScenarioDataService._table_value(value)) == value
