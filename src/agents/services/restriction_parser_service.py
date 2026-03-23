from typing import AsyncGenerator

from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.model_clients.base_client import BaseClient

from .base_llm_service import BaseLlmService


class RestrictionParserService(BaseLlmService):

    def __init__(self, llm_client: BaseClient, mcp_client: IduMcpClient):
        """
        Initialization function for SimpleLlmService. Inherits from BaseService.
        Args:
            llm_client (BaseClient): BaseClient for communicating with LLM.
        """
        super().__init__(llm_client)
        self.llm_client: BaseClient = llm_client
        self.mcp_client: IduMcpClient = mcp_client

    async def run_restriction_execution_pipline(
        self, model: str, user_query: str
    ) -> AsyncGenerator:
        """
        Run pipline fo forming restrictions
        Returns:
            AsyncGenerator
        """

        pass
