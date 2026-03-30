from fastmcp import Client

from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.common.config.app_config_loader import load_config
from src.agents.services.simple_llm_service import SimpleLlmService
from src.agents.services.restriction_parser_service import RestrictionParserService
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient


def init_dependencies() -> dict[str, SimpleLlmService]:

    app_config: AgentsAppConfig = load_config()
    idu_fastmcp_client = Client(app_config.IDU_MCP_URL)
    return {
        "app_config": app_config,
        "idu_fastmcp_client": idu_fastmcp_client,
        "idu_mcp_client": IduMcpClient(idu_fastmcp_client),
        "simple_llm_service": SimpleLlmService(app_config.OLLAMA_URL),
        "restriction_parser_service": RestrictionParserService(app_config.OLLAMA_URL)
    }
