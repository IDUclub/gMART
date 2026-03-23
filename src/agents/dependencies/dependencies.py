from src.agents.services.base_service import BaseService
from src.agents.services.simple_llm_service import SimpleLlmService

from .init_dependencies import init_dependencies

app_deps: dict[str, BaseService | SimpleLlmService] = init_dependencies()


def get_simple_llm_service() -> SimpleLlmService:
    simple_llm_service: SimpleLlmService = app_deps["simple_llm_service"]
    if not isinstance(simple_llm_service, SimpleLlmService):
        raise TypeError(f"Expected SimpleLlmService, got {type(simple_llm_service)}")
    return simple_llm_service
