from src.agents.common.exceptions.ollama_exceptions import ModelNotFound
from src.agents.model_clients.base_client import BaseLlmClient


class BaseLlmService(BaseLlmClient):

    def __init__(self, llm_host: str):
        """
        Initialization function for BaseLlmService. Inherits from BaseLlmClient.
        Args:
            llm_host (str): Ollama host.
        """

        super().__init__(host=llm_host)

    async def get_models(self, only_running: bool = False):
        """
        Get list of available models.
        Args:
            only_running (bool, optional): If True, get only running models. Defaults to False.
        """

        models = await self.llm_client.ps() if only_running else await self.llm_client.list()
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
