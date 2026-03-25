from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from fastmcp.prompts import Message
from fastmcp.server.dependencies import get_access_token
from pydantic import Field

from src.idu_mcp.api_clients.urban_api_client import UrbanApiClient
from src.idu_mcp.dependencies.dependencies import get_urban_api_client

mcp = FastMCP(name="RestrictionsPromptService")

@mcp.prompt(name="NoGetServicesExample", tags={"services"})
async def get_no_services_example_prompt() -> list[Message]:
    """
    Function forms example prompt for no services in user request.
    Returns:
        list[dict[str, str]]: example response for model.
    """

    return [
        Message(
            "Нельзя размещать свалки в непосредственной близости от промышленных объектов. "
            "Какие промышленный объекты попадают в радиус действия свалок в пределах 500 метров?",
            role="user",
        ),
        Message(
            "Исходя из запроса информация о сервисах не требуется, т.к. в запросе пользователя "
            "не упоминаются инфраструктурные сервисы.",
            role="assistant",
        ),
    ]

@mcp.prompt(name="GetServicesExample", tags={"services"})
async def get_services_example_prompt() -> list[Message]:
    """
    Function forms example prompt for services in user request.
    Returns:
        list[Message]: example response for a model with tool call.
    """

    return [
        Message(
            "Нельзя размещать магазины алкогольной продукции в непосредственной близости от школ. "
            "Какие магазины попадают в радиус действия школ в пределах 200 метров?",
            role="user",
        ),
        Message(
            {
                "tool_calls": [
                    {
                        "name": "GetServices",
                        "arguments": {
                            "services_names": ["школа", "магазин у дома"]
                        }
                    }
                ]
            },
            role="assistant",
        ),
    ]

@mcp.prompt(name="NoGetPhysicalObjectsExample", tags={"physical_objects"})
async def get_no_physical_objects_example_prompt() -> list[Message]:
    """
    Function forms example prompt for no physical objects in user request.
    Returns:
        list[Message]: example response for model.
    """

    return [
        Message(
            "Нельзя размещать свалки в непосредственной близости от промышленных объектов. "
            "Какие промышленный объекты попадают в радиус действия свалок в пределах 500 метров?",
            role="user",
        ),
        Message(
            "Исходя из запроса информация о сервисах не требуется, т.к. в запросе пользователя "
            "не упоминаются инфраструктурные сервисы.",
            role="assistant",
        ),
    ]

@mcp.prompt(name="GetPhysicalObjectsExample", tags={"physical_objects"})
async def get_physical_objects_example_prompt() -> list[Message]:
    """
    Function forms example prompt for no physical objects in user request.
    Returns:
        list[Message]: example response for a model with tool call.
    """

    return [
        Message(
            "Нельзя размещать свалки в непосредственной близости от промышленных объектов. "
            "Какие промышленный объекты попадают в радиус действия свалок в пределах 500 метров?",
            role="user",
        ),
        Message(
            {
                "tool_calls": [
                    {
                        "name": "GetPhysicalObjects",
                        "arguments": {
                            "physical_objects_names": [
                                "промышленная территория",
                                "полигон тбо",
                            ]
                        },
                    }
                ]
            },
            role="assistant",
        ),
    ]

@mcp.prompt(name="GetAvailableServices", tags={"services"})
async def get_available_services(
        scenario_id: int = Field(description="Scenario ID from Urban API"),
        urban_api_client: UrbanApiClient = Depends(get_urban_api_client),
) -> str:
    """
    Function retrieves prompt for data retrieval with available services names.
    Args:
        scenario_id (int): Scenario ID from Urban API.
        urban_api_client (UrbanApiClient): UrbanApiClient instance from dependencies.
    """

    token = get_access_token()
    available_names = await urban_api_client.get_available_scenario_services(scenario_id, token)
    return f"Выбирай из списка сервисов: {', '.join([name.lower() for name in available_names])}"

@mcp.prompt(name="GetAvailablePhysicalObjects", tags={"physical_objects"})
async def get_available_physical_objects(
        scenario_id: int = Field(description="Scenario ID from Urban API"),
        urban_api_client = Depends(get_urban_api_client)
):
    """
    Function retrieves prompt for data retrieval with available services names.
    Args:
        scenario_id (int): Scenario ID from Urban API.
        urban_api_client (UrbanApiClient): UrbanApiClient instance from dependencies.
    """

    token = get_access_token()
    available_names = await urban_api_client.get_available_physical_objects(scenario_id, token)
    return f"Выбирай из списка физических объектов: {', '.join([name.lower() for name in available_names])}"
