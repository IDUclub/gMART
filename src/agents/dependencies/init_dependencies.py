from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.common.config.app_config_loader import load_config
from src.agents.model_clients.base_client import BaseClient
from src.agents.services.base_service import BaseService
from src.agents.services.simple_llm_service import SimpleLlmService


def init_dependencies() -> dict[str, BaseService]:

    app_config: AgentsAppConfig = load_config()
    base_client: BaseClient = BaseClient(app_config.OLLAMA_URL)
    return {
        "app_config": app_config,
        "simple_llm_service": SimpleLlmService(base_client),
    }
