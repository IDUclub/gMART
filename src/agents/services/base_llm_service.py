from src.agents.common.exceptions.ollama_exceptions import ModelNotFound
from src.agents.model_clients.base_client import BaseClient

from .base_service import BaseService


class BaseLlmService(BaseService):

    def __init__(self, llm_client: BaseClient):
        """
        Initialization function for BaseLlmService. Inherits from BaseService.
        Args:
            llm_client (BaseClient): BaseClient for communicating with LLM.
        """

        super().__init__()
        self.llm_client = llm_client

    async def get_models(self, only_running: bool = False):
        """
        Get list of available models.
        Args:
            only_running (bool, optional): If True, get only running models. Defaults to False.
        """

        client = await self.llm_client.get_client()
        models = await client.ps() if only_running else await client.list()
        return [model["model"] for model in models["models"]]

    async def validate_model(self, model_name: str):
        """
        Function validates model requested by user.
        Args:
            model_name (str): Model name to validate.
        Raises:
            ModelNotFound (Exception): Exception raised if model not found.
        """

        available_models = await self.get_models()
        if model_name not in available_models:
            raise ModelNotFound(model_name, available_models)
