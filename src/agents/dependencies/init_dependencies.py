from typing import Any

import redis.asyncio as aioredis
from fastmcp import Client

from src.agents.a2a.executor import RestrictionAgentExecutor
from src.agents.a2a.orchestrator_executor import OrchestratorAgentExecutor
from src.agents.a2a.provision_executor import ProvisionAgentExecutor
from src.agents.a2a.task_store import A2ATaskStore
from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)
from src.agents.common.api_handlers.json_api_handler import JsonApiHandler
from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.common.config.app_config_loader import load_config
from src.agents.common.logging.log_config import config_logger
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.services.a2a_service import A2AService
from src.agents.services.critic_service import CriticService
from src.agents.services.orchestrator_a2a_service import OrchestratorA2AService
from src.agents.services.orchestrator_pipeline_service import (
    OrchestratorPipelineService,
)
from src.agents.services.orchestrator_service import OrchestratorService
from src.agents.services.pipeline_state import PipelineStateStore
from src.agents.services.provision_a2a_service import ProvisionA2AService
from src.agents.services.provsion_service import ProvisionService
from src.agents.services.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.simple_llm_service import SimpleLlmService
from src.agents.services.system_service import SystemService


def init_dependencies() -> dict[
    str,
    AgentsAppConfig
    | SystemService
    | Client[Any]
    | IduMcpClient
    | SimpleLlmService
    | RestrictionParserService
    | ProvisionService
    | A2AService
    | ProvisionA2AService
    | OrchestratorA2AService
    | OrchestratorPipelineService
    | JsonApiHandler
    | ChatStorageApiClient
    | PipelineStateStore,
]:

    logs_path = config_logger()
    app_config: AgentsAppConfig = load_config()
    idu_fastmcp_client = Client(app_config.IDU_MCP_URL)
    chat_storage_json_handler = JsonApiHandler(app_config.CHAT_STORAGE_URL)
    chat_storage_client = ChatStorageApiClient(chat_storage_json_handler)
    redis_client = aioredis.from_url(app_config.REDIS_URL, decode_responses=True)
    pipeline_state_store = PipelineStateStore(redis_client)
    restriction_parser_service = RestrictionParserService(
        app_config.OLLAMA_URL, chat_storage_client, pipeline_state_store
    )
    provision_service = ProvisionService(
        app_config.OLLAMA_URL, chat_storage_client, pipeline_state_store
    )

    # Orchestrator: dedicated executor instances with isolated task stores
    # so orchestrator sub-calls don't pollute the standalone agent task stores.
    orchestrator_service = OrchestratorService(host=app_config.OLLAMA_URL)
    orchestrator_executor = OrchestratorAgentExecutor(
        orchestrator_service=orchestrator_service,
        restriction_executor=RestrictionAgentExecutor(
            restriction_parser_service, A2ATaskStore()
        ),
        provision_executor=ProvisionAgentExecutor(provision_service, A2ATaskStore()),
        task_store=A2ATaskStore(),
    )
    orchestrator_a2a_service = OrchestratorA2AService(orchestrator_executor)
    critic_service = CriticService(host=app_config.OLLAMA_URL)
    orchestrator_pipeline_service = OrchestratorPipelineService(
        host=app_config.OLLAMA_URL,
        orchestrator_service=orchestrator_service,
        restriction_service=restriction_parser_service,
        provision_service=provision_service,
        critic_service=critic_service,
        state_store=pipeline_state_store,
    )

    return {
        "app_config": app_config,
        "system_service": SystemService(logs_path),
        "idu_fastmcp_client": idu_fastmcp_client,
        "idu_mcp_client": IduMcpClient(idu_fastmcp_client),
        "simple_llm_service": SimpleLlmService(
            app_config.OLLAMA_URL, chat_storage_client
        ),
        "restriction_parser_service": restriction_parser_service,
        "provision_service": provision_service,
        "a2a_service": A2AService(restriction_parser_service),
        "provision_a2a_service": ProvisionA2AService(provision_service),
        "orchestrator_a2a_service": orchestrator_a2a_service,
        "orchestrator_pipeline_service": orchestrator_pipeline_service,
        "chat_storage_json_handler": chat_storage_json_handler,
        "chat_storage_client": chat_storage_client,
        "pipeline_state_store": pipeline_state_store,
    }
