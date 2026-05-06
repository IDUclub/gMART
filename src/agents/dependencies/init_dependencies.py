from typing import Any

from fastmcp import Client

from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)
from src.agents.common.api_handlers.json_api_handler import JsonApiHandler
from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.common.config.app_config_loader import load_config
from src.agents.common.logging.log_config import config_logger
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.services.a2a_service import A2AService
from src.agents.services.restriction_parser_service import RestrictionParserService
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
    | A2AService
    | JsonApiHandler
    | ChatStorageApiClient,
]:

    logs_path = config_logger()
    app_config: AgentsAppConfig = load_config()
    idu_fastmcp_client = Client(app_config.IDU_MCP_URL)
    chat_storage_json_handler = JsonApiHandler(app_config.CHAT_STORAGE_URL)
    chat_storage_client = ChatStorageApiClient(chat_storage_json_handler)
    restriction_parser_service = RestrictionParserService(
        app_config.OLLAMA_URL, chat_storage_client
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
        "a2a_service": A2AService(restriction_parser_service),
        "chat_storage_json_handler": chat_storage_json_handler,
        "chat_storage_client": chat_storage_client,
    }
