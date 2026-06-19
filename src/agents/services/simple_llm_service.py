from collections.abc import AsyncGenerator
from typing import Any

from ollama import ChatResponse

from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)
from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.services.base_llm_service import BaseLlmService


class SimpleLlmService(BaseLlmService):
    """
    Class for handling simple LLM messages and chats. Inherits from BaseLlmService.
    Attributes:
        llm_client (BaseLlmClient):BaseClient for communicating with LLM.
        chat_storage_client (ChatStorageApiClient): Instance of ChatStorageApiClient for current app.
    """

    def __init__(
        self,
        llm_host: str,
        chat_storage_client: ChatStorageApiClient,
        urban_api_client: UrbanApiClient,
    ):
        """
        Initialization function for SimpleLlmService. Inherits from BaseService.
        Args:
            llm_host (str): Ollama host.
            chat_storage_client (ChatStorageApiClient): Instance of ChatStorageApiClient.
            urban_api_client (UrbanApiClient): Instance of UrbanApiClient.
        """
        super().__init__(
            llm_host=llm_host,
            chat_storage_client=chat_storage_client,
            urban_api_client=urban_api_client,
        )

    async def generate_message(self, user_request: str, model: str) -> dict[str, Any]:
        """
        Generate a message from a user request.
        Args:
            user_request (str): User request.
            model (str): Model name gto generate response on.
        Returns:
            dict[str, Any]: Response message.
        """

        await self.validate_model(model)
        messages = [{"role": "user", "content": user_request}]
        return await self.llm_client.chat(model, messages, stream=False)

    async def generate_stream_message(
        self, user_request: str, model: str
    ) -> AsyncGenerator[dict[str, str], None]:
        """
        Generate a message from a user request.
        Args:
            user_request (str): User request.
            model (str): Model name to generate response on.
        Returns:
            AsyncGenerator[dict[str, Any], None]: generator for chunks from ollama api.
        """

        messages = [{"role": "user", "content": user_request}]
        async for part in await self.llm_client.chat(model, messages, stream=True):
            part: ChatResponse
            if part.done:
                yield {"type": "Text", "content": part.message.content}
                return
            yield {"type": "Text", "content": part.message.content}
