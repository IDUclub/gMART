from ollama import AsyncClient as AsyncOllamaClient


class BaseClient:
    """
    Base class for agent clients.
    Attributes:
        host (str): The host of the agent.
        client_cert (str): The client certificate for the agent.
        client_key (str): The client key for the agent.
    """

    def __init__(
        self, host: str, client_cert: str | None = None, client_key: str | None = None
    ):
        """
        Base client initialization function.
        Args:
            host (str): The host of the agent.
            client_cert (str): The client certificate for the agent to connect to gpu server.
            client_key (str): The client key for the agent to connect to gpu server.
        """

        self.host = host
        self.client_cert = client_cert
        self.client_key = client_key

    async def get_client(self):
        """
        Get a client for the agent.
        """

        return AsyncOllamaClient(host=self.host)
