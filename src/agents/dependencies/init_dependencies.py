import redis.asyncio as aioredis
from idu_service_auth import KeycloakTokenClient

from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)
from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.common.api_handlers.json_api_handler import JsonApiHandler
from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.common.config.app_config_loader import load_config
from src.agents.common.logging.log_config import config_logger
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
from src.agents.services.scenario_data_a2a_service import ScenarioDataA2AService
from src.agents.services.scenario_data_service import ScenarioDataService
from src.agents.services.simple_llm_service import SimpleLlmService
from src.agents.services.system_service import SystemService
from src.common.service_auth import build_service_auth


def init_dependencies() -> dict[
    str,
    AgentsAppConfig
    | SystemService
    | SimpleLlmService
    | RestrictionParserService
    | ProvisionService
    | DvdRagService
    | NormGraphRagService
    | OrchestratorService
    | A2AService
    | ProvisionA2AService
    | DocumentQaA2AService
    | NormGraphA2AService
    | ScenarioDataA2AService
    | JsonApiHandler
    | ChatStorageApiClient
    | UrbanApiClient
    | PipelineStateStore
    | KeycloakTokenClient,
]:

    logs_path = config_logger()
    app_config: AgentsAppConfig = load_config()
    service_auth = build_service_auth()
    chat_storage_json_handler = JsonApiHandler(
        app_config.CHAT_STORAGE_URL, service_auth=service_auth
    )
    chat_storage_client = ChatStorageApiClient(chat_storage_json_handler)
    urban_api_json_handler = JsonApiHandler(
        app_config.URBAN_API_URL, service_auth=service_auth
    )
    urban_api_client = UrbanApiClient(urban_api_json_handler)
    redis_client = aioredis.from_url(app_config.REDIS_URL, decode_responses=True)
    pipeline_state_store = PipelineStateStore(redis_client)
    restriction_parser_service = RestrictionParserService(
        app_config.OLLAMA_URL,
        chat_storage_client,
        urban_api_client,
        pipeline_state_store,
    )
    provision_service = ProvisionService(
        app_config.OLLAMA_URL,
        chat_storage_client,
        urban_api_client,
        pipeline_state_store,
    )
    scenario_data_service = ScenarioDataService(
        app_config.OLLAMA_URL,
        chat_storage_client,
        urban_api_client,
        pipeline_state_store,
        linear_workflow_enabled=app_config.SCENARIO_DATA_LINEAR_WORKFLOW_ENABLED,
        workspace_enabled=app_config.SCENARIO_DATA_WORKSPACE_ENABLED,
        idu_mcp_url=app_config.IDU_MCP_URL,
        service_auth=service_auth,
    )
    dvd_rag_service = DvdRagService(
        app_config.OLLAMA_URL,
        chat_storage_client,
        urban_api_client,
        pipeline_state_store,
    )
    normgraph_rag_service = NormGraphRagService(
        app_config.OLLAMA_URL,
        chat_storage_client,
        urban_api_client,
        pipeline_state_store,
    )
    orchestrator_service = OrchestratorService(
        app_config.OLLAMA_URL,
        chat_storage_client,
        urban_api_client,
        pipeline_state_store,
        restriction_parser_service,
        provision_service,
        dvd_rag_service,
        normgraph_rag_service,
        app_config,
        scenario_data_service=scenario_data_service,
    )
    return {
        "app_config": app_config,
        "service_auth": service_auth,
        "system_service": SystemService(logs_path, app_config),
        "simple_llm_service": SimpleLlmService(
            app_config.OLLAMA_URL, chat_storage_client, urban_api_client
        ),
        "restriction_parser_service": restriction_parser_service,
        "provision_service": provision_service,
        "scenario_data_service": scenario_data_service,
        "dvd_rag_service": dvd_rag_service,
        "normgraph_rag_service": normgraph_rag_service,
        "orchestrator_service": orchestrator_service,
        "a2a_service": A2AService(restriction_parser_service),
        "provision_a2a_service": ProvisionA2AService(provision_service),
        "dvd_a2a_service": DocumentQaA2AService(dvd_rag_service),
        "normgraph_a2a_service": NormGraphA2AService(normgraph_rag_service),
        "scenario_data_a2a_service": ScenarioDataA2AService(scenario_data_service),
        "chat_storage_json_handler": chat_storage_json_handler,
        "chat_storage_client": chat_storage_client,
        "urban_api_json_handler": urban_api_json_handler,
        "urban_api_client": urban_api_client,
        "pipeline_state_store": pipeline_state_store,
    }
