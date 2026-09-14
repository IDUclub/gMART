from src.agents.model_clients.factory import build_llm_adapter
from src.agents.model_clients.llm_base import BaseLlmAdapter
from src.agents.runtime.runner import run_completion


class BaseLlmClient:
    """
    Base class for agent clients.
    Attributes:
        host (str): The host of the agent.
        llm_client (BaseLlmAdapter): backend-neutral LLM adapter (an
            OpenAI-compatible server by default, the native Ollama client when
            LLM_BACKEND=ollama).
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

        async for part in await run_completion(
            self.llm_client, model, messages, stream=True, agent_name="base_client"
        ):
            yield {
                "type": "chunk",
                "content": {"text": part.message.content, "done": part.done},
            }
