from ollama import AsyncClient as AsyncOllamaClient
from ollama import ChatResponse


class BaseLlmClient:
    """
    Base class for agent clients.
    Attributes:
        host (str): The host of the agent.
        client (AsyncOllamaClient | AsyncClient): asynchronous ollama client.
    """

    def __init__(self, host: str):
        """
        Base client initialization function.
        Args:
            host (str): The host of the agent.
        """

        self.host = host
        self.llm_client = AsyncOllamaClient(host=self.host)

    async def execute_request(self, model: str, messages: list[dict]):

        async for part in await self.llm_client.chat(model, messages, stream=True):
            part: ChatResponse
            yield {
                "type": "chunk",
                "content": {"text": part.message.content, "done": part.done},
            }
