from src.agents.model_clients.factory import build_llm_adapter
from src.agents.model_clients.llm_base import BaseLlmAdapter


class BaseLlmClient:
    """
    Base class for agent clients.
    Attributes:
        host (str): The host of the agent.
        llm_client (BaseLlmAdapter): backend-neutral LLM adapter (Ollama by
            default, an OpenAI-compatible server when LLM_BACKEND=openai).
    """

    def __init__(self, host: str):
        """
        Base client initialization function.
        Args:
            host (str): The host of the agent.
        """

        self.host = host
        self.llm_client: BaseLlmAdapter = build_llm_adapter(host)

    async def execute_request(self, model: str, messages: list[dict]):

        async for part in await self.llm_client.chat(model, messages, stream=True):
            yield {
                "type": "chunk",
                "content": {"text": part.message.content, "done": part.done},
            }
