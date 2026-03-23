from typing import Annotated

from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from fastmcp.server.dependencies import get_access_token
from geojson_pydantic import FeatureCollection

from src.idu_mcp.common.auth.token_verifier import AnyTokenVerifier
from src.idu_mcp.dependencies.dependencies import get_scenario_id, get_urban_api_tools
from src.idu_mcp.tools_services.entites.object_type_enum import ObjectTypeEnum
from src.idu_mcp.tools_services.urb_api_tools import UrbanApiTool

urban_api_mcp = FastMCP("URBAN API MCP", auth=AnyTokenVerifier())
tools_tags = {"data", "urban_api"}


@urban_api_mcp.tool(
    name="Получить сервисы на проектной территории",
    description="""Получить сервисы на проектной территории по ID сценария.
    
    Сервисами являются объекты инфраструктуры, такие как школы, поликлиники, детские сады, рестораны и др. 
    Названия сервисов передаются списком строк в единственном числе, именительном падеже. 
    В качестве ответа возвращается JSON‑словарь, где ключом является название сервиса, 
    а значением FeatureCollection с пространственной репрезентацией объектов в системе координат WGS84 (EPSG:4326).
    
    Структура ответа: dict[str (название сервиса), FeatureCollection (слой с сервисами в формате GeoJSON)].""",
    tags=tools_tags,
    annotations={"title": "GET physical objects for scenario", "readOnlyHint": True},
    meta={"author": "LeonDeTur"},
)
async def get_services_by_name(
    services_names: Annotated[
        list[str],
        "Название сервиса на русском языке в единственном числе и именительном падеже",
    ],
    scenario_id=Depends(get_scenario_id),
    urban_api_tools: UrbanApiTool = Depends(get_urban_api_tools),
) -> dict[str, FeatureCollection]:
    """
    Urban API tools interface method to retrieve services from scenario.
    Args:
        services_names (list[str]): Services names from db.
        scenario_id (int): Scenario id from Urban API. Extracts from requests headers.
        urban_api_tools (UrbanApiTool): Urban API tools instance.
    Returns:
        dict[str | FeatureCollection]: dict with service name as key and FeatureCollection as value.
    """

    token = get_access_token()
    return await urban_api_tools.get_entity_by_names(
        scenario_id, services_names, ObjectTypeEnum.SERVICE, token.token
    )


@urban_api_mcp.tool(
    name="Получить физические объекты на проектной территории по ID сценария.",
    description="""Получить физические объекты на проектной территории по ID сценария.
    
    Физическими объектами являются объекты капитального строительства, такие как жилые здания, промышленные объекты и др.
    Названия физических объектов передаются списком строк в единственном числе, именительном падеже.
    В качестве ответа возвращается json словарь, где ключом является название физического объекта, который есть на проектной 
    территории для сценария, а значением FeatureCollection с пространственной репрезентацией объектов в WGS 84 (4326).
    
    Структуры ответа: dict[str (название сервиса), FeatureCollection (слой с физическими объектами в формате geojson)]""",
    tags=tools_tags,
    annotations={"title": "GET physical objects for scenario", "readOnlyHint": True},
    meta={"author": "LeonDeTur"},
)
async def get_physical_objects_by_name(
    physical_objects_names: Annotated[
        list[str], "Physical object names as list from db"
    ],
    scenario_id=Depends(get_scenario_id),
    urban_api_tools: UrbanApiTool = Depends(get_urban_api_tools),
) -> dict[str, FeatureCollection]:
    """
    Urban API tools interface method to retrieve physical objects from scenario.
    Args:
        scenario_id (int): scenario id from Urban API
        physical_objects_names (list[str]): physical object names from db
        urban_api_tools (UrbanApiTool): Urban API tools instance
    Returns:
        dict[str | FeatureCollection]: dict with physical object name as key and FeatureCollection as value
    """

    token = get_access_token()
    return await urban_api_tools.get_entity_by_names(
        scenario_id, physical_objects_names, ObjectTypeEnum.PHYSICAL_OBJECT, token.token
    )
