from fastapi import Depends
from fastmcp import Client

from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.common.auth.auth import verify_bearer_token
from src.agents.dependencies.init_dependencies import init_dependencies
from src.agents.mcp_clients.effects_mcp_client import EffectsMcpClient
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.services.a2a_service import A2AService
from src.agents.services.pipeline_state import PipelineStateStore
from src.agents.services.provision_a2a_service import ProvisionA2AService
from src.agents.services.provsion_service import ProvisionService
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


def get_urban_api_client() -> UrbanApiClient:
    """
    Function returns initialized UrbanApiClient object from dependencies.
    Returns:
         UrbanApiClient: urban_api_client object initialized on startup.
    """

    urban_api_client: UrbanApiClient = app_deps["urban_api_client"]
    if not isinstance(urban_api_client, UrbanApiClient):
        raise TypeError(f"Expected UrbanApiClient, got {type(urban_api_client)}")
    return urban_api_client


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


async def get_effects_mcp_client(
    token: str = Depends(verify_bearer_token),
) -> EffectsMcpClient:
    """
    Function returns EffectsMcpClient instance with provided authorization.
    Args:
        token (str): Bearer token for auth.
    Returns:
        EffectsMcpClient: EffectsMcpClient instance for the Object Effects MCP Server.
    """

    mcp_url: str = app_deps["app_config"].EFFECTS_MCP_URL
    return EffectsMcpClient(Client(mcp_url, auth=token), mcp_url=mcp_url)


def get_provision_service() -> ProvisionService:
    """
    Function returns initialized ProvisionService object from dependencies.
    Returns:
        ProvisionService: ProvisionService instance.
    """

    service: ProvisionService = app_deps["provision_service"]
    if not isinstance(service, ProvisionService):
        raise TypeError(f"Expected ProvisionService, got {type(service)}")
    return service


async def get_provision_a2a_service() -> ProvisionA2AService:
    """
    Function returns ProvisionA2AService instance.
    Returns:
        ProvisionA2AService: ProvisionA2AService instance.
    """

    service = app_deps["provision_a2a_service"]
    if not isinstance(service, ProvisionA2AService):
        raise TypeError(f"Expected ProvisionA2AService, got {type(service)}")
    return service


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
