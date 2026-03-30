from ollama import AsyncClient as AsyncOllamaClient


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
