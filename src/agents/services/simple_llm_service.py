from collections.abc import AsyncGenerator

from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)
from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.llm_agents.stage import LlmStage
from src.agents.model_clients.llm_base import LlmChatResponse, LlmMessage
from src.agents.services.base_llm_service import BaseLlmService

# A bare chat turn: one user message, plain text back.
_SIMPLE_STAGE = LlmStage("simple_llm")


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

    async def generate_message(
        self, user_request: str, model: str | None
    ) -> LlmChatResponse:
        """
        Generate a message from a user request.
        Args:
            user_request (str): User request.
            model (str | None): Model name; None resolves to the provider's default.
        Returns:
            LlmChatResponse: Response message.
        """

        model = await self.resolve_model(model)
        await self.validate_model(model)
        messages = [{"role": "user", "content": user_request}]
        result = await _SIMPLE_STAGE.run(self.llm_client, model, messages)
        # The REST response keeps the chat-response shape it always had.
        return LlmChatResponse(
            model=model,
            message=LlmMessage(content=result.text, thinking=result.thinking),
            done=True,
            done_reason=result.done_reason,
        )

    async def generate_stream_message(
        self, user_request: str, model: str | None
    ) -> AsyncGenerator[dict[str, str], None]:
        """
        Generate a message from a user request.
        Args:
            user_request (str): User request.
            model (str | None): Model name; None resolves to the provider's default.
        Returns:
            AsyncGenerator[dict[str, Any], None]: generator of chunks from the LLM backend.
        """

        model = await self.resolve_model(model)
        messages = [{"role": "user", "content": user_request}]
        async for delta in _SIMPLE_STAGE.stream(self.llm_client, model, messages):
            yield {"type": "Text", "content": delta}
        # The closing chunk, as the backend's final ``done`` part was relayed.
        yield {"type": "Text", "content": ""}
