from fastmcp import Client
from fastapi import Depends

from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.common.auth.auth import verify_bearer_token
from src.agents.services.a2a_service import A2AService
from src.agents.services.simple_llm_service import SimpleLlmService


from .init_dependencies import init_dependencies
from ..mcp_clients.idu_mcp_client import IduMcpClient
from ..services.restriction_parser_service import RestrictionParserService
from ..services.system_service import SystemService

app_deps: dict[str, object] = init_dependencies()


def get_simple_llm_service() -> SimpleLlmService:
    """
    Function returns initialized SimpleLlmService object from dependencies.
    Returns:
         SimpleLlmService: simple_llm_service object initialized on startup.
    """

    simple_llm_service: SimpleLlmService = app_deps["simple_llm_service"]
    if not isinstance(simple_llm_service, SimpleLlmService):
        raise TypeError("Expected SimpleLlmService, got {}".format(type(simple_llm_service)))
    return simple_llm_service


async def get_idu_mcp_client(token: str = Depends(verify_bearer_token)) -> IduMcpClient:
    """
    Function returns IduMcpClient instance with provided authorization.
    Args:
        token (str): Bearer token for auth.
    Returns:
        IduMcpClient: IduMcpClient instance for IDU MCP Server.
    """

    return IduMcpClient(Client(app_deps["app_config"].IDU_MCP_URL, auth=token))

async def get_restriction_parser_service() -> RestrictionParserService:
    """
    Function returns RestrictionParserService instance.
    Returns:
        RestrictionParserService: RestrictionParserService instance.
    """

    restriction_parser_service: RestrictionParserService = app_deps["restriction_parser_service"]
    if not isinstance(restriction_parser_service, RestrictionParserService):
        raise TypeError("Expected SimpleLlmService, got {}".format(type(restriction_parser_service)))
    return app_deps["restriction_parser_service"]


async def get_a2a_service() -> A2AService:
    """
    Function returns A2A service for restriction generation agent.
    Returns:
        A2AService: A2A service instance.
    """

    a2a_service = app_deps["a2a_service"]
    if not isinstance(a2a_service, A2AService):
        raise TypeError("Expected A2AService, got {}".format(type(a2a_service)))
    return a2a_service


async def get_system_service() -> SystemService:

    return app_deps["system_service"]
