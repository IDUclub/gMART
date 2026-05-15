class AgentsAppConfig:
    """
    Fast API rest agents service configuration class.
    Attributes:
        OLLAMA_URL (str): Ollama URL.
        IDU_MCP_URL (str): IDU MCP URL.
        CHAT_STORAGE_URL (str): Chat Storage service URL.
        REDIS_URL (str): Redis URL (used for pipeline state and pub/sub).
    """

    OLLAMA_URL: str
    IDU_MCP_URL: str
    CHAT_STORAGE_URL: str
    REDIS_URL: str

    def __init__(
        self,
        ollama_api_url: str,
        idu_mcp_url: str,
        chat_storage_url: str,
        redis_url: str = "redis://localhost:6379",
    ) -> None:
        """
        Initialization function for AgentsAppConfig class.
        Args:
            ollama_api_url (str): Ollama URL.
            idu_mcp_url (str): IDU MCP URL.
            chat_storage_url (str): Chat Storage service URL.
            redis_url (str): Redis URL. Defaults to redis://localhost:6379.
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
        self.REDIS_URL = redis_url
