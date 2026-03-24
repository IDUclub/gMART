from typing import AsyncGenerator

from loguru import logger
from ollama import AsyncClient

from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.model_clients.base_client import BaseLlmClient

from .base_llm_service import BaseLlmService


#TODO add full streaming for all llm responses
class RestrictionParserService(BaseLlmService):

    def __init__(self, ollama_host: str):
        """
        Initialization function for SimpleLlmService. Inherits from BaseService.
        Args:
            ollama_host (str): Ollama host.
        """

        super().__init__(ollama_host)

    async def run_services_retirement(
            self,
            mcp_client: IduMcpClient,
            user_query: str,
            model: str,
            scenario_id: int,
    ):

        instructions = "Исходя из запроса пользователя получи данные по сервисам для формирования ограничений\n" + "\nВыбирай из списка сервисов: школа, поликлиника. Используй именно предложенное сочетание как название сервиса. Если исходя из запроса пользователя получение сервисов не требуется, то не вызывай инструмент, а просто верни сообщение."
        services_prompts = await mcp_client.get_services_example_prompts()


        try:
            response = await self.llm_client.chat(
                model = model,
                messages=[
                    {"role": "system", "content": instructions},
                    *services_prompts,
                    {"role": "user", "content": user_query}
                ],

            )
        except Exception as e:
            logger.exception(e)
            raise

    async def run_restriction_execution_pipline(
        self,
        mcp_client: IduMcpClient,
        model: str,
        user_query: str,
        scenario_id: int,
    ) -> AsyncGenerator:
        """
        Run pipline fo forming restrictions
        Returns:
            AsyncGenerator
        """

        data_tools = await self.mcp_client.get_urban_api_tools()
        geometry_tools = await self.mcp_client.get_geometry_tools()

