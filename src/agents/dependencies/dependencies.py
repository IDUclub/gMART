from fastapi import Depends
from fastmcp import Client

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.dependencies.init_dependencies import init_dependencies
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.services.a2a_service import A2AService
from src.agents.services.pipeline_state import PipelineStateStore
from src.agents.services.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.simple_llm_service import SimpleLlmService
from src.agents.services.system_service import SystemService

app_deps: dict[str, object] = init_dependencies()


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


def get_pipeline_state_store() -> PipelineStateStore:
    """Returns the shared PipelineStateStore (Redis-backed)."""
    store: PipelineStateStore = app_deps["pipeline_state_store"]
    if not isinstance(store, PipelineStateStore):
        raise TypeError(f"Expected PipelineStateStore, got {type(store)}")
    return store


async def get_idu_mcp_client(
    token: str = Depends(verify_bearer_token),
) -> IduMcpClient:
    """
    Function returns IduMcpClient instance with provided authorization.
    Args:
        token (str): Bearer token for auth.
    Returns:
        IduMcpClient: IduMcpClient instance for IDU MCP Server.
    """

    mcp_url: str = app_deps["app_config"].IDU_MCP_URL
    return IduMcpClient(Client(mcp_url, auth=token), mcp_url=mcp_url)


async def get_restriction_parser_service() -> RestrictionParserService:
    """
    Function returns RestrictionParserService instance.
    Returns:
        RestrictionParserService: RestrictionParserService instance.
    """

    restriction_parser_service: RestrictionParserService = app_deps[
        "restriction_parser_service"
    ]
    if not isinstance(restriction_parser_service, RestrictionParserService):
        raise TypeError(
            f"Expected SimpleLlmService, got {type(restriction_parser_service)}"
        )
    return app_deps["restriction_parser_service"]


async def get_a2a_service() -> A2AService:
    """
    Function returns A2A service for restriction generation agent.
    Returns:
        A2AService: A2A service instance.
    """

    a2a_service = app_deps["a2a_service"]
    if not isinstance(a2a_service, A2AService):
        raise TypeError(f"Expected A2AService, got {type(a2a_service)}")
    return a2a_service


async def get_system_service() -> SystemService:
    """
    Function returns SystemService instance.
    Returns:
        SystemService: SystemService instance for current app.
    """

    return app_deps["system_service"]
