class AgentsAppConfig:
    """
    Fast API rest agents service configuration class.
    Attributes:
        OLLAMA_URL (str): Ollama URL.
        IDU_MCP_URL (str): IDU MCP URL.
        CHAT_STORAGE_URL:
    """

    OLLAMA_URL: str
    IDU_MCP_URL: str
    CHAT_STORAGE_URL: str

    def __init__(
        self, ollama_api_url: str, idu_mcp_url: str, chat_storage_url: str
    ) -> None:
        """
        Initialization function for AgentsAppConfig class.
        Args:
            ollama_api_url (str): Ollama URL.
            idu_mcp_url (str): IDU MCP URL.
            chat_storage_url (str): Chat Storage service URL.
        Raises:
            ValueError: if provided value is not valid.
        """

        if not ollama_api_url:
            raise ValueError("OLLAMA_API_URL must be set")
        self.OLLAMA_URL = ollama_api_url
        if not idu_mcp_url:
            raise ValueError("IDU_MCP_URL must be set")
        self.IDU_MCP_URL = idu_mcp_url
        if not chat_storage_url:
            raise ValueError("CHAT_STORAGE_URL must be set")
        self.CHAT_STORAGE_URL = chat_storage_url
