from unittest.mock import AsyncMock

from src.idu_mcp.tools_services.urb_api_tools import UrbanApiTool


async def test_global_type_resolution_is_independent_of_scenario_instances():
    client = AsyncMock()
    client.get_service_name_id.return_value = {"Школа": 11}
    client.get_physical_objects_name_id.return_value = {"Жилой дом": 22}
    tool = UrbanApiTool(client)

    result = await tool.resolve_entity_types(
        service_names=["школа", "Несуществующая услуга"],
        physical_object_names=["жилой дом"],
        token="user-1",
    )

    assert result["service"]["Школа"] == {
        "found": True,
        "canonical_name": "Школа",
        "type_id": 11,
    }
    assert result["service"]["Несуществующая услуга"]["found"] is False
    assert result["physical_object"]["Жилой дом"]["type_id"] == 22
    client.get_service_name_id.assert_awaited_once_with(
        ["Школа", "Несуществующая услуга"], "user-1"
    )


async def test_global_type_resolution_skips_unused_dictionary_requests():
    client = AsyncMock()
    client.get_service_name_id.return_value = {"Школа": 11}
    tool = UrbanApiTool(client)

    result = await tool.resolve_entity_types(
        service_names=["Школа"], physical_object_names=[], token="user-1"
    )

    assert result["physical_object"] == {}
    client.get_physical_objects_name_id.assert_not_awaited()
