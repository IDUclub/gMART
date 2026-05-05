from typing import Any

from fastmcp import Client

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
    | A2AService,
]:

    logs_path = config_logger()
    app_config: AgentsAppConfig = load_config()
    idu_fastmcp_client = Client(app_config.IDU_MCP_URL)
    restriction_parser_service = RestrictionParserService(app_config.OLLAMA_URL)
    return {
        "app_config": app_config,
        "system_service": SystemService(logs_path),
        "idu_fastmcp_client": idu_fastmcp_client,
        "idu_mcp_client": IduMcpClient(idu_fastmcp_client),
        "simple_llm_service": SimpleLlmService(app_config.OLLAMA_URL),
        "restriction_parser_service": restriction_parser_service,
        "a2a_service": A2AService(restriction_parser_service),
    }
