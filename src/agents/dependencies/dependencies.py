from fastmcp import Client

from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.services.base_service import BaseService
from src.agents.services.simple_llm_service import SimpleLlmService


from .init_dependencies import init_dependencies
from ..mcp_clients.idu_mcp_client import IduMcpClient

app_deps: dict[
    str, AgentsAppConfig | BaseService | SimpleLlmService
] = init_dependencies()


def get_simple_llm_service() -> SimpleLlmService:
    """
    Function returns initialized SimpleLlmService object from dependencies.
    Returns:
         SimpleLlmService: simple_llm_service object initialized on startup.
    """

    simple_llm_service: SimpleLlmService = app_deps["simple_llm_service"]
    if not isinstance(simple_llm_service, SimpleLlmService):
        raise TypeError(f"Expected SimpleLlmService, got {type(simple_llm_service)}")
    return simple_llm_service


async def get_idu_mcp_client(token: str) -> IduMcpClient:
    """
    Function returns IduMcpClient instance with provided authorization.
    Args:
        token (str): Bearer token for auth.
    Returns:
        IduMcpClient: IduMcpClient instance for IDU MCP Server.
    """

    return IduMcpClient(Client(app_deps["app_config"].IDU_MCP_URL, auth=token))
