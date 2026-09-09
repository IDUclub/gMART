from dataclasses import dataclass
from typing import Any

from fastapi import Depends
from idu_service_auth import KeycloakTokenClient

from src.agents.api_clients.dvd_api_client import DvdApiClient
from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.common.auth.auth import optional_bearer_token, verify_bearer_token
from src.agents.common.auth.synapse_auth import SynapseCallerVerifier
from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.common.exceptions.base_exceptions import AgentsNotFound
from src.agents.dependencies.init_dependencies import init_dependencies
from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient
from src.agents.mcp_clients.effects_mcp_client import EffectsMcpClient
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient
from src.agents.mcp_clients.urban_mcp_client import UrbanMcpClient
from src.agents.services.a2a_service import A2AService
from src.agents.services.dvd_a2a_service import DocumentQaA2AService
from src.agents.services.dvd_rag_service import DvdRagService
from src.agents.services.mcp_diagnostics_service import McpDiagnosticsService
from src.agents.services.normgraph_a2a_service import NormGraphA2AService
from src.agents.services.normgraph_rag_service import NormGraphRagService
from src.agents.services.orchestrator_service import OrchestratorService
from src.agents.services.pipeline_state import PipelineStateStore
from src.agents.services.provision_a2a_service import ProvisionA2AService
from src.agents.services.provsion_service import ProvisionService
from src.agents.services.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.scenario_data_a2a_service import ScenarioDataA2AService
from src.agents.services.scenario_data_service import ScenarioDataService
from src.agents.services.simple_llm_service import SimpleLlmService
from src.agents.services.synapse_gateway_service import SynapseGatewayService
from src.agents.services.synapse_run_store import SynapseRunStore
from src.agents.services.system_service import SystemService
from src.common.service_auth import (
    ANONYMOUS_USER_ID,
    ServiceTokenAuth,
    internal_user_context_jwt,
    service_mcp_client,
    user_id_from_jwt,
)

app_deps: dict[str, object] = init_dependencies()


@dataclass(frozen=True)
class A2ACallerContext:
    user_id: str
    pipeline_token: str
    is_synapse: bool


async def resolve_a2a_caller(payload: Any, token: str) -> A2ACallerContext:
    """Verify Synapse service identity and recover the original user mapping."""

    config = get_app_config()
    if not config.SYNAPSE_ENABLED:
        return A2ACallerContext("", token, False)

    claims = await get_synapse_caller_verifier().verify_claims(token)
    client_id = claims.get("azp") or claims.get("client_id")
    if client_id != config.SYNAPSE_A2A_CLIENT_ID:
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            from src.agents.common.exceptions.base_exceptions import (
                AgentsUnauthorizedException,
            )

            raise AgentsUnauthorizedException("JWT subject is missing")
        return A2ACallerContext(subject, token, False)

    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json", exclude_none=True)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    params = payload.get("params") if isinstance(payload, dict) else {}
    params = params if isinstance(params, dict) else {}
    message = params.get("message")
    message = message if isinstance(message, dict) else {}
    metadata = params.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    project_id = metadata.get("project_id")
    run_id = (
        params.get("contextId")
        or params.get("context_id")
        or message.get("contextId")
        or message.get("context_id")
    )
    user_id = await get_synapse_run_store().resolve_a2a_user(
        project_id=str(project_id) if project_id else None,
        run_id=str(run_id) if run_id else None,
    )
    if not user_id:
        from src.agents.common.exceptions.base_exceptions import (
            AgentsUnauthorizedException,
        )

        raise AgentsUnauthorizedException("Unknown Synapse project/run correlation")
    return A2ACallerContext(user_id, internal_user_context_jwt(user_id), True)


async def a2a_idu_mcp_client(user_id: str) -> IduMcpClient:
    client = await service_mcp_client(
        get_app_config().IDU_MCP_URL, get_idu_mcp_service_auth(), user_id
    )
    return IduMcpClient(client)


async def a2a_effects_mcp_client(user_id: str) -> EffectsMcpClient:
    client = await service_mcp_client(
        get_app_config().EFFECTS_MCP_URL, get_service_auth(), user_id
    )
    return EffectsMcpClient(client)


async def a2a_dvd_mcp_client(user_id: str) -> DvdMcpClient:
    mcp_url = get_app_config().DVD_MCP_URL
    if not mcp_url:
        raise ValueError("DVD_MCP_SERVER is not configured")
    client = await service_mcp_client(mcp_url, get_service_auth(), user_id)
    return DvdMcpClient(client, mcp_url=mcp_url)


async def a2a_normgraph_mcp_client(user_id: str) -> NormGraphMcpClient:
    mcp_url = get_app_config().NORM_GRAPH_MCP_URL
    if not mcp_url:
        raise ValueError("NORM_GRAPH_MCP_SERVER is not configured")
    client = await service_mcp_client(mcp_url, get_service_auth(), user_id)
    return NormGraphMcpClient(client, mcp_url=mcp_url)


async def a2a_urban_mcp_client(user_id: str) -> UrbanMcpClient:
    mcp_url = get_app_config().URBAN_MCP_URL
    if not mcp_url:
        raise ValueError("URBAN_MCP_SERVER is not configured")
    return UrbanMcpClient(mcp_url, ServiceTokenAuth(get_service_auth(), user_id))


async def get_mcp_diagnostics_service(
    token: str = Depends(verify_bearer_token),
) -> McpDiagnosticsService:
    """Return the request-scoped, allowlisted MCP console service."""

    return McpDiagnosticsService(get_app_config(), token, get_service_auth())


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


def get_service_auth() -> KeycloakTokenClient:
    auth = app_deps["service_auth"]
    if not isinstance(auth, KeycloakTokenClient):
        raise TypeError(f"Expected KeycloakTokenClient, got {type(auth)}")
    return auth


def get_idu_mcp_service_auth() -> KeycloakTokenClient:
    """Return the identity dedicated to the internal gMART IDU MCP boundary."""

    auth = app_deps["idu_mcp_service_auth"]
    if not isinstance(auth, KeycloakTokenClient):
        raise TypeError(f"Expected KeycloakTokenClient, got {type(auth)}")
    return auth


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


def get_synapse_run_store() -> SynapseRunStore:
    store = app_deps["synapse_run_store"]
    if not isinstance(store, SynapseRunStore):
        raise TypeError(f"Expected SynapseRunStore, got {type(store)}")
    return store


def get_synapse_gateway_service() -> SynapseGatewayService:
    service = app_deps.get("synapse_gateway_service")
    if not isinstance(service, SynapseGatewayService):
        raise AgentsNotFound("Synapse integration is disabled")
    return service


def get_synapse_caller_verifier() -> SynapseCallerVerifier:
    verifier = app_deps.get("synapse_caller_verifier")
    if not isinstance(verifier, SynapseCallerVerifier):
        raise AgentsNotFound("Synapse integration is disabled")
    return verifier


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
    client = await service_mcp_client(
        mcp_url, get_idu_mcp_service_auth(), user_id_from_jwt(token)
    )
    return IduMcpClient(client)


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
    client = await service_mcp_client(
        mcp_url, get_service_auth(), user_id_from_jwt(token)
    )
    return EffectsMcpClient(client)


async def get_dvd_mcp_client(
    token: str | None = Depends(optional_bearer_token),
) -> DvdMcpClient:
    """
    Function returns a DvdMcpClient for the IDU_DVD document vector-DB MCP server.

    The IDU_DVD MCP server receives the process-wide service token and user id.
    The bearer token is optional: anonymous callers of the public document-QA stream
    have no Keycloak subject, so they are announced as ANONYMOUS_USER_ID. IDU_DVD
    requires the header but honours the value only together with project_id /
    scenario_id, which an anonymous request is never allowed to send — so the search
    stays on the shared index. Endpoints that must stay authorized keep their own
    verify_bearer_token.
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
    client = await service_mcp_client(
        mcp_url,
        get_service_auth(),
        user_id_from_jwt(token) if token else ANONYMOUS_USER_ID,
    )
    return DvdMcpClient(client, mcp_url=mcp_url)


async def get_dvd_api_client(
    token: str = Depends(verify_bearer_token),
) -> DvdApiClient:
    """Return a user-scoped REST client for IDU_DVD uploads and listings."""

    api_url: str | None = app_deps["app_config"].DVD_API_URL
    if not api_url:
        raise ValueError(
            "DVD_API_URL is not configured and could not be derived from DVD_MCP_SERVER"
        )
    return DvdApiClient(api_url, get_service_auth(), user_id_from_jwt(token))


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


async def get_normgraph_mcp_client(
    token: str = Depends(verify_bearer_token),
) -> NormGraphMcpClient:
    """
    Function returns a NormGraphMcpClient for the NormGraph restriction-graph MCP server.

    The NormGraph MCP server receives the process-wide service token and user id.
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
    client = await service_mcp_client(
        mcp_url, get_service_auth(), user_id_from_jwt(token)
    )
    return NormGraphMcpClient(client, mcp_url=mcp_url)


async def get_optional_dvd_mcp_client(
    token: str = Depends(verify_bearer_token),
) -> DvdMcpClient | None:
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
    client = await service_mcp_client(
        mcp_url, get_service_auth(), user_id_from_jwt(token)
    )
    return DvdMcpClient(client, mcp_url=mcp_url)


async def get_optional_normgraph_mcp_client(
    token: str = Depends(verify_bearer_token),
) -> NormGraphMcpClient | None:
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
    client = await service_mcp_client(
        mcp_url, get_service_auth(), user_id_from_jwt(token)
    )
    return NormGraphMcpClient(client, mcp_url=mcp_url)


async def get_urban_mcp_client(
    token: str = Depends(verify_bearer_token),
) -> UrbanMcpClient:
    """Return an authenticated aggregate client for all Urban MCP groups."""

    base_url: str | None = app_deps["app_config"].URBAN_MCP_URL
    if not base_url:
        raise ValueError(
            "URBAN_MCP_SERVER is not configured — set it to enable the "
            "/scenario-data agent"
        )
    return UrbanMcpClient(
        base_url,
        ServiceTokenAuth(get_service_auth(), user_id_from_jwt(token)),
    )


async def get_optional_urban_mcp_client(
    token: str = Depends(verify_bearer_token),
) -> UrbanMcpClient | None:
    """Return Urban MCP client for the orchestrator when configured."""

    base_url: str | None = app_deps["app_config"].URBAN_MCP_URL
    if not base_url:
        return None
    return UrbanMcpClient(
        base_url,
        ServiceTokenAuth(get_service_auth(), user_id_from_jwt(token)),
    )


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


def get_scenario_data_service() -> ScenarioDataService:
    """Return the initialized scenario-data agent service."""

    service = app_deps["scenario_data_service"]
    if not isinstance(service, ScenarioDataService):
        raise TypeError(f"Expected ScenarioDataService, got {type(service)}")
    return service


async def get_scenario_data_a2a_service() -> ScenarioDataA2AService:
    """Return the initialized scenario-data A2A JSON-RPC service."""

    service = app_deps["scenario_data_a2a_service"]
    if not isinstance(service, ScenarioDataA2AService):
        raise TypeError(f"Expected ScenarioDataA2AService, got {type(service)}")
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


async def get_system_service() -> SystemService:
    """
    Function returns SystemService instance.
    Returns:
        SystemService: SystemService instance for current app.
    """

    return app_deps["system_service"]
