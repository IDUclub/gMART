from fastapi import Depends
from fastmcp import Client

from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.common.auth.auth import verify_bearer_token
from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.dependencies.init_dependencies import init_dependencies
from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient
from src.agents.mcp_clients.effects_mcp_client import EffectsMcpClient
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient
from src.agents.mcp_clients.urban_data_mcp_client import UrbanDataMcpClient
from src.agents.services.a2a_service import A2AService
from src.agents.services.dvd_a2a_service import DocumentQaA2AService
from src.agents.services.dvd_rag_service import DvdRagService
from src.agents.services.normgraph_a2a_service import NormGraphA2AService
from src.agents.services.normgraph_rag_service import NormGraphRagService
from src.agents.services.orchestrator_service import OrchestratorService
from src.agents.services.pipeline_state import PipelineStateStore
from src.agents.services.provision_a2a_service import ProvisionA2AService
from src.agents.services.provsion_service import ProvisionService
from src.agents.services.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.simple_llm_service import SimpleLlmService
from src.agents.services.system_service import SystemService
from src.agents.services.urban_data_a2a_service import UrbanDataA2AService
from src.agents.services.urban_data_qa_service import UrbanDataQaService

app_deps: dict[str, object] = init_dependencies()


def get_app_config() -> AgentsAppConfig:
    """
    Function returns the AgentsAppConfig loaded on startup.
    Returns:
         AgentsAppConfig: app_config object initialized on startup.
    """

    app_config: AgentsAppConfig = app_deps["app_config"]
    if not isinstance(app_config, AgentsAppConfig):
        raise TypeError(f"Expected AgentsAppConfig, got {type(app_config)}")
    return app_config


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


async def get_dvd_mcp_client() -> DvdMcpClient:
    """
    Function returns a DvdMcpClient for the IDU_DVD document vector-DB MCP server.

    The IDU_DVD MCP server is unauthenticated, so no bearer token is attached.
    Returns:
        DvdMcpClient: Client for the IDU_DVD MCP server.
    Raises:
        ValueError: If DVD_MCP_SERVER is not configured.
    """

    mcp_url: str | None = app_deps["app_config"].DVD_MCP_URL
    if not mcp_url:
        raise ValueError(
            "DVD_MCP_SERVER is not configured — set it to enable the /documents agent"
        )
    return DvdMcpClient(Client(mcp_url), mcp_url=mcp_url)


def get_dvd_rag_service() -> DvdRagService:
    """
    Function returns initialized DvdRagService object from dependencies.
    Returns:
        DvdRagService: DvdRagService instance.
    """

    service: DvdRagService = app_deps["dvd_rag_service"]
    if not isinstance(service, DvdRagService):
        raise TypeError(f"Expected DvdRagService, got {type(service)}")
    return service


async def get_dvd_a2a_service() -> DocumentQaA2AService:
    """
    Function returns DocumentQaA2AService instance.
    Returns:
        DocumentQaA2AService: DocumentQaA2AService instance.
    """

    service = app_deps["dvd_a2a_service"]
    if not isinstance(service, DocumentQaA2AService):
        raise TypeError(f"Expected DocumentQaA2AService, got {type(service)}")
    return service


async def get_normgraph_mcp_client() -> NormGraphMcpClient:
    """
    Function returns a NormGraphMcpClient for the NormGraph restriction-graph MCP server.

    The NormGraph MCP server is unauthenticated, so no bearer token is attached.
    Returns:
        NormGraphMcpClient: Client for the NormGraph MCP server.
    Raises:
        ValueError: If NORM_GRAPH_MCP_SERVER is not configured.
    """

    mcp_url: str | None = app_deps["app_config"].NORM_GRAPH_MCP_URL
    if not mcp_url:
        raise ValueError(
            "NORM_GRAPH_MCP_SERVER is not configured — set it to enable the /norms agent"
        )
    return NormGraphMcpClient(Client(mcp_url), mcp_url=mcp_url)


async def get_optional_dvd_mcp_client() -> DvdMcpClient | None:
    """
    Function returns a DvdMcpClient when DVD_MCP_SERVER is configured, else None.

    Used by the orchestrator: the documents agent is simply excluded from the
    planner catalogue when the URL is unset, so the endpoint must not fail.
    Returns:
        DvdMcpClient | None: Client for the IDU_DVD MCP server or None.
    """

    mcp_url: str | None = app_deps["app_config"].DVD_MCP_URL
    if not mcp_url:
        return None
    return DvdMcpClient(Client(mcp_url), mcp_url=mcp_url)


async def get_optional_normgraph_mcp_client() -> NormGraphMcpClient | None:
    """
    Function returns a NormGraphMcpClient when NORM_GRAPH_MCP_SERVER is configured, else None.

    Used by the orchestrator: the norms agent is simply excluded from the
    planner catalogue when the URL is unset, so the endpoint must not fail.
    Returns:
        NormGraphMcpClient | None: Client for the NormGraph MCP server or None.
    """

    mcp_url: str | None = app_deps["app_config"].NORM_GRAPH_MCP_URL
    if not mcp_url:
        return None
    return NormGraphMcpClient(Client(mcp_url), mcp_url=mcp_url)


async def get_urban_data_mcp_client(
    token: str = Depends(verify_bearer_token),
) -> UrbanDataMcpClient:
    """
    Function returns UrbanDataMcpClient instance with provided authorization.
    Args:
        token (str): Bearer token for auth.
    Returns:
        UrbanDataMcpClient: Client for the external, grouped Urban MCP server.
    Raises:
        ValueError: If URBAN_DATA_MCP_SERVER is not configured.
    """

    base_url: str | None = app_deps["app_config"].URBAN_DATA_MCP_URL
    if not base_url:
        raise ValueError(
            "URBAN_DATA_MCP_SERVER is not configured — set it to enable the "
            "/urban-data agent"
        )
    return UrbanDataMcpClient(base_url=base_url, token=token)


async def get_optional_urban_data_mcp_client(
    token: str = Depends(verify_bearer_token),
) -> UrbanDataMcpClient | None:
    """
    Function returns an UrbanDataMcpClient when URBAN_DATA_MCP_SERVER is configured,
    else None.

    Used by the orchestrator: the urban-data agent is simply excluded from the planner
    catalogue when the URL is unset, so the endpoint must not fail. Unlike the DVD/NormGraph
    optional clients, this one still needs the caller's token — only its ``projects`` group
    is authenticated — so it is created eagerly whenever the URL is configured, regardless
    of whether the planner ends up routing to it.
    Returns:
        UrbanDataMcpClient | None: Client for the Urban MCP server or None.
    """

    base_url: str | None = app_deps["app_config"].URBAN_DATA_MCP_URL
    if not base_url:
        return None
    return UrbanDataMcpClient(base_url=base_url, token=token)


def get_orchestrator_service() -> OrchestratorService:
    """
    Function returns initialized OrchestratorService object from dependencies.
    Returns:
        OrchestratorService: OrchestratorService instance.
    """

    service: OrchestratorService = app_deps["orchestrator_service"]
    if not isinstance(service, OrchestratorService):
        raise TypeError(f"Expected OrchestratorService, got {type(service)}")
    return service


def get_normgraph_rag_service() -> NormGraphRagService:
    """
    Function returns initialized NormGraphRagService object from dependencies.
    Returns:
        NormGraphRagService: NormGraphRagService instance.
    """

    service: NormGraphRagService = app_deps["normgraph_rag_service"]
    if not isinstance(service, NormGraphRagService):
        raise TypeError(f"Expected NormGraphRagService, got {type(service)}")
    return service


async def get_normgraph_a2a_service() -> NormGraphA2AService:
    """
    Function returns NormGraphA2AService instance.
    Returns:
        NormGraphA2AService: NormGraphA2AService instance.
    """

    service = app_deps["normgraph_a2a_service"]
    if not isinstance(service, NormGraphA2AService):
        raise TypeError(f"Expected NormGraphA2AService, got {type(service)}")
    return service


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


def get_urban_data_qa_service() -> UrbanDataQaService:
    """
    Function returns initialized UrbanDataQaService object from dependencies.
    Returns:
        UrbanDataQaService: UrbanDataQaService instance.
    """

    service: UrbanDataQaService = app_deps["urban_data_qa_service"]
    if not isinstance(service, UrbanDataQaService):
        raise TypeError(f"Expected UrbanDataQaService, got {type(service)}")
    return service


async def get_urban_data_a2a_service() -> UrbanDataA2AService:
    """
    Function returns UrbanDataA2AService instance.
    Returns:
        UrbanDataA2AService: UrbanDataA2AService instance.
    """

    service = app_deps["urban_data_a2a_service"]
    if not isinstance(service, UrbanDataA2AService):
        raise TypeError(f"Expected UrbanDataA2AService, got {type(service)}")
    return service


async def get_system_service() -> SystemService:
    """
    Function returns SystemService instance.
    Returns:
        SystemService: SystemService instance for current app.
    """

    return app_deps["system_service"]
