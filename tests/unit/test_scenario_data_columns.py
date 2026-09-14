"""Tables reach users with Russian labels, readable cells and no identifiers."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from src.agents.services.scenario_data.scenario_data_columns import (
    COLUMN_LABELS,
    DROPPED_FIELDS,
    ID_LABELS,
    ids_requested,
    valid_label,
)
from src.agents.services.scenario_data.scenario_data_service import ScenarioDataService

SNAPSHOT = (
    Path(__file__).parents[1]
    / "fixtures"
    / "scenario_data"
    / "urban_mcp_output_fields.json"
)


def table(rows, **kwargs):
    return ScenarioDataService._table_from_result(rows, name="t", title="T", **kwargs)


def test_every_urban_mcp_output_field_has_a_label_or_is_dropped():
    fields = set(json.loads(SNAPSHOT.read_text(encoding="utf-8")))

    assert fields - (set(COLUMN_LABELS) | set(ID_LABELS) | DROPPED_FIELDS) == set()


def test_every_dictionary_label_is_short_russian_text():
    for label in [*COLUMN_LABELS.values(), *ID_LABELS.values()]:
        assert valid_label(label) == label


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Покажи ID сервисов", True),
        ("какие айди у объектов", True),
        ("Выведи идентификаторы школ", True),
        ("покажи service_id", True),
        ("Покажи школы сценария 772", False),
        ("Покажи объекты по видам", False),
        ("Provide the list", False),
    ],
)
def test_ids_are_shown_only_when_asked_for(query, expected):
    assert ids_requested(query) is expected


def test_ids_are_hidden_and_labels_come_from_the_dictionary():
    result = table([{"id": 1, "service_id": 5, "name": "Школа", "capacity": 600}])

    assert result["columns"] == [
        {"key": "name", "label": "Название"},
        {"key": "capacity", "label": "Вместимость"},
    ]
    assert result["rows"] == [{"name": "Школа", "capacity": 600}]


def test_an_explicit_id_request_keeps_the_column_with_a_russian_label():
    result = table([{"service_id": 5, "name": "Школа"}], show_ids=True)

    assert result["columns"][0] == {"key": "service_id", "label": "ID сервиса"}
    assert result["rows"] == [{"service_id": 5, "name": "Школа"}]


def test_nested_objects_and_lists_are_shown_by_name():
    result = table(
        [
            {
                "name": "Школа № 1",
                "service_type": {"id": 7, "name": "Школа"},
                "territories": [{"id": 1, "name": "Центр"}, {"id": 2, "name": "Север"}],
                "possible_vri_list": ["2.1", "3.5"],
            }
        ]
    )

    assert result["rows"] == [
        {
            "name": "Школа № 1",
            "service_type": "Школа",
            "territories": "Центр, Север",
            "possible_vri_list": "2.1, 3.5",
        }
    ]
    assert [column["label"] for column in result["columns"]] == [
        "Название",
        "Тип сервиса",
        "Территории",
        "Возможные ВРИ",
    ]


def test_a_long_nested_name_is_kept_whole():
    result = table([{"territory": {"id": 17, "name": "а" * 1200}}])

    assert result["rows"] == [{"territory": "а" * 1200}]


def test_geometry_envelope_and_unnamed_nested_values_are_dropped():
    result = table(
        [
            {
                "name": "Школа",
                "geometry": {"type": "Point", "coordinates": [30, 60]},
                "centre_point": {"type": "Point", "coordinates": [30, 60]},
                "properties": {"weight": 1},
                "building": {"floors": 3},
            }
        ]
    )

    assert [column["key"] for column in result["columns"]] == ["name"]
    assert result["rows"] == [{"name": "Школа"}]


def test_booleans_read_as_yes_and_no():
    result = table(
        [
            {"name": "A", "is_based": True},
            {"name": "B", "is_based": False},
            {"name": "C", "is_based": None},
        ]
    )

    assert [row["is_based"] for row in result["rows"]] == ["да", "нет", None]


def test_colliding_labels_stay_distinct():
    result = table(
        [{"address": "ул. Ленина, 1", "readable_address": "Город, ул. Ленина, 1"}]
    )

    assert [column["label"] for column in result["columns"]] == [
        "Адрес",
        "Адрес (полный)",
    ]


def test_a_table_of_identifiers_only_is_not_shown():
    assert table([{"id": 1, "scenario_id": 2}]) is None


@pytest.fixture
def service(monkeypatch, fake_llm, fake_urban, state_store):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *args, **kwargs: fake_llm,
    )
    return ScenarioDataService("http://llm", AsyncMock(), fake_urban, state_store)


async def emit(service, content):
    if not await service.state_store.exists("req"):
        await service.state_store.create(
            "req",
            chat_id=None,
            user_query="q",
            scenario_id=None,
            model="model",
            temperature=0,
        )
    await service._buf("req", {"type": "table", "content": content})
    return [column["label"] for column in content["columns"]]


async def test_an_unknown_field_is_translated_once_from_its_description(
    service, fake_llm
):
    service._column_descriptions["heating_type"] = "Type of the heating system"
    fake_llm.json_responses = [json.dumps({"heating_type": "Тип отопления"})]

    first = await emit(service, table([{"name": "Дом", "heating_type": "газ"}]))
    second = await emit(service, table([{"name": "Баня", "heating_type": "дрова"}]))

    assert first == second == ["Название", "Тип отопления"]
    assert len(fake_llm.chat_calls) == 1
    call = fake_llm.chat_calls[0]
    assert call.model == "model" and call.options["temperature"] == 0
    assert "Type of the heating system" in call.messages[-1]["content"]


@pytest.mark.parametrize(
    "answer",
    [
        '{"heating_type": "heating_type"}',
        '{"heating_type": "Подпись колонки, которая длиннее сорока символов"}',
        "не JSON",
    ],
)
async def test_an_unusable_translation_keeps_the_key_and_warns(
    service, fake_llm, answer
):
    fake_llm.json_responses = [answer]
    warnings: list[str] = []
    handler = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        labels = await emit(service, table([{"name": "Дом", "heating_type": "газ"}]))
    finally:
        logger.remove(handler)

    assert labels == ["Название", "heating_type"]
    assert any("keeps its source key" in message for message in warnings)


async def test_a_failed_translation_call_still_sends_the_table(service, fake_llm):
    fake_llm.chat = AsyncMock(side_effect=RuntimeError("model is down"))

    labels = await emit(service, table([{"name": "Дом", "heating_type": "газ"}]))

    assert labels == ["Название", "heating_type"]


async def test_a_table_with_known_labels_makes_no_model_call(service, fake_llm):
    await emit(service, table([{"name": "Дом", "floors": 5}]))

    assert fake_llm.chat_calls == []
