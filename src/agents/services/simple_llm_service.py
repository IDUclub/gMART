from collections.abc import AsyncGenerator
from typing import Any

from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)
from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.model_clients.llm_base import LlmChatResponse
from src.agents.runtime.runner import run_completion
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

    async def generate_message(
        self, user_request: str, model: str | None
    ) -> dict[str, Any]:
        """
        Generate a message from a user request.
        Args:
            user_request (str): User request.
            model (str | None): Model name; None resolves to the provider's default.
        Returns:
            dict[str, Any]: Response message.
        """

        model = await self.resolve_model(model)
        await self.validate_model(model)
        messages = [{"role": "user", "content": user_request}]
        response = await run_completion(
            self.llm_client,
            model,
            messages,
            stream=False,
            agent_name="simple_llm_service",
        )
        # REST exposes plain data, never provider-owned Pydantic serializers.
        # Some lazily constructed OpenAI usage-detail models cannot serialize
        # under Pydantic 2.13 even though the completion itself is valid.
        message = response["message"]
        usage = response.get("usage")
        counts = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = (
                usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
            )
            if value is not None:
                counts[key] = value
        return {
            "model": response.get("model", model),
            "message": {
                key: message.get(key) for key in ("role", "content", "thinking")
            },
            "done": response.get("done", True),
            "done_reason": response.get("done_reason"),
            "usage": counts or None,
        }

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
        async for part in await run_completion(
            self.llm_client,
            model,
            messages,
            stream=True,
            agent_name="simple_llm_service",
        ):
            part: LlmChatResponse
            if part.done:
                yield {"type": "Text", "content": part.message.content}
                return
            yield {"type": "Text", "content": part.message.content}
