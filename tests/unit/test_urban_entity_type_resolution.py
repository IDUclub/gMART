from unittest.mock import AsyncMock, call

import pytest

from src.idu_mcp.api_clients.urban_api_client import UrbanApiClient
from src.idu_mcp.tools_services.entity_names import resolve_catalog_names
from src.idu_mcp.tools_services.urb_api_tools import UrbanApiTool


async def test_global_type_resolution_is_independent_of_scenario_instances():
    client = AsyncMock()
    client.get_type_catalog.side_effect = [
        {"Школа": 11, "Спортивная площадка": 66},
        {"Жилой дом": 22},
    ]
    result = await UrbanApiTool(client).resolve_entity_types(
        service_names=["школ", "спортивных площадок", "Несуществующая услуга"],
        physical_object_names=["жилых домов"],
        token="user-1",
    )
    assert result["service"]["Школ"] == {
        "found": True,
        "canonical_name": "Школа",
        "type_id": 11,
    }
    assert (
        result["service"]["Спортивных площадок"]["canonical_name"]
        == "Спортивная площадка"
    )
    assert result["service"]["Несуществующая услуга"]["found"] is False
    assert result["physical_object"]["Жилых домов"]["canonical_name"] == "Жилой дом"
    client.get_type_catalog.assert_has_awaits(
        [call("service", "user-1"), call("physical_object", "user-1")]
    )


async def test_global_type_resolution_skips_unused_dictionary_requests():
    client = AsyncMock()
    client.get_type_catalog.return_value = {"Школа": 11}
    result = await UrbanApiTool(client).resolve_entity_types(
        service_names=["Школа"], physical_object_names=[], token="user-1"
    )
    assert result["physical_object"] == {}
    client.get_type_catalog.assert_awaited_once_with("service", "user-1")


@pytest.mark.parametrize(
    "name,canonical",
    [
        ("спортивных площадок", "Спортивная площадка"),
        ("СПОРТИВНЫМИ  ПЛОЩАДКАМИ", "Спортивная площадка"),
        ("школ", "Школа"),
        ("школы", "Школа"),
        ("школой", "Школа"),
        ("жилых домов", "Жилой дом"),
        ("жилые дома", "Жилой дом"),
        ("детских садов", "Детский сад"),
        ("трёхэтажных жилых домов", "Трехэтажный жилой дом"),
    ],
)
def test_inflected_phrase_matches_real_catalog_name(name, canonical):
    result = resolve_catalog_names([name], {canonical: 42})[name]
    assert result == {"found": True, "canonical_name": canonical, "type_id": 42}


@pytest.mark.parametrize(
    "name",
    [
        "радиус обслуживания школ",
        "школьные спортивные площадки",
        "трёхэтажных жилых домов",
        "нежилых домов",
        "жилой застройки",
        "школы и детские сады",
        "школ 12",
        "",
    ],
)
def test_normalization_does_not_drop_qualifiers_or_invent_synonyms(name):
    catalog = {"Школа": 22, "Жилой дом": 4, "Спортивная площадка": 66}
    assert resolve_catalog_names([name], catalog)[name]["found"] is False


def test_ambiguous_morphology_stays_unresolved_but_exact_name_wins():
    catalog = {"Школа": 22, "Школы": 23}
    result = resolve_catalog_names(["школ", "ШКОЛА"], catalog)
    assert result["школ"]["found"] is False
    assert result["ШКОЛА"]["type_id"] == 22


@pytest.mark.parametrize(
    "entity_type,endpoint,id_field",
    [
        ("service", "v1/service_types", "service_type_id"),
        ("physical_object", "v1/physical_object_types", "physical_object_type_id"),
    ],
)
async def test_catalog_client_fetches_all_global_names(entity_type, endpoint, id_field):
    handler = AsyncMock()
    handler.get.return_value = [
        {"name": "Тип А", id_field: 1},
        {"name": "Тип Б", id_field: 2},
    ]
    result = await UrbanApiClient(handler).get_type_catalog(entity_type, "user-1")
    assert result == {"Тип А": 1, "Тип Б": 2}
    handler.get.assert_awaited_once_with(endpoint, auth_token="user-1")


async def test_catalog_failure_is_not_reported_as_empty_dictionary():
    handler = AsyncMock()
    handler.get.side_effect = RuntimeError("catalog unavailable")
    with pytest.raises(RuntimeError, match="catalog unavailable"):
        await UrbanApiClient(handler).get_type_catalog("service", "user-1")
