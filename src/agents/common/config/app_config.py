class AgentsAppConfig:
    """
    Fast API rest agents service configuration class.
    Attributes:
        OLLAMA_URL (str): Ollama URL
    """

    OLLAMA_URL: str

    def __init__(self, ollama_api_url: str) -> None:
        """
        Initialization function for AgentsAppConfig class.
        Args:
            ollama_api_url (str): Ollama URL
        """

        if not ollama_api_url:
            raise ValueError("OLLAMA_API_URL must be set")
        self.OLLAMA_URL = ollama_api_url
