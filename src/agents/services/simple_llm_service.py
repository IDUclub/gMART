from typing import Any, AsyncGenerator

from ollama_chat import ChatResponse

from src.agents.model_clients.base_client import BaseClient

from .base_llm_service import BaseLlmService


class SimpleLlmService(BaseLlmService):
    """
    Class for handling simple LLM messages and chats. Inherits from BaseLlmService.
    Attributes:
        llm_client (BaseClient):BaseClient for communicating with LLM.
    """

    def __init__(self, llm_client: BaseClient):
        """
        Initialization function for SimpleLlmService. Inherits from BaseService.
        Args:
            llm_client (BaseClient): BaseClient for communicating with LLM.
        """
        super().__init__(llm_client)

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
        client = await self.llm_client.get_client()
        messages = [{"role": "user", "content": user_request}]
        return await client.chat(model, messages, stream=False)

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

        client = await self.llm_client.get_client()
        messages = [{"role": "user", "content": user_request}]
        async for part in await client.chat(model, messages, stream=True):
            part: ChatResponse
            if part.done:
                yield {"type": "Text", "content": part.message.content}
                return
            yield {"type": "Text", "content": part.message.content}
