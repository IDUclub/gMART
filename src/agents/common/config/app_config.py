class AgentsAppConfig:
    """
    Fast API rest agents service configuration class.
    Attributes:
        OLLAMA_URL (str): Ollama URL.
        IDU_MCP_URL (str): IDU MCP URL.
        EFFECTS_MCP_URL (str): Object Effects MCP URL.
        DVD_MCP_URL (str | None): IDU_DVD document vector-DB MCP URL (optional).
        CHAT_STORAGE_URL (str): Chat Storage service URL.
        URBAN_API_URL (str): Urban API URL.
        REDIS_URL (str): Redis URL (used for pipeline state and pub/sub).
        SYSTEM_PASSWORD (str | None): Optional password guarding system config retrieval.
    """

    OLLAMA_URL: str
    IDU_MCP_URL: str
    EFFECTS_MCP_URL: str
    DVD_MCP_URL: str | None
    CHAT_STORAGE_URL: str
    URBAN_API_URL: str
    REDIS_URL: str
    SYSTEM_PASSWORD: str | None

    def __init__(
        self,
        ollama_api_url: str,
        idu_mcp_url: str,
        effects_mcp_url: str,
        chat_storage_url: str,
        urban_api_url: str,
        dvd_mcp_url: str | None = None,
        redis_url: str = "redis://localhost:6379",
        system_password: str | None = None,
    ) -> None:

        if not ollama_api_url:
            raise ValueError("OLLAMA_API_URL must be set")
        self.OLLAMA_URL = ollama_api_url
        if not idu_mcp_url:
            raise ValueError("IDU_MCP_URL must be set")
        self.IDU_MCP_URL = idu_mcp_url
        if not effects_mcp_url:
            raise ValueError("OBJECTS_EFFECTS_MCP_SERVER must be set")
        self.EFFECTS_MCP_URL = effects_mcp_url
        if not self.EFFECTS_MCP_URL:
            raise ValueError("OBJECTS_EFFECTS_MCP_SERVER must be set")
        # Optional: only required by the document-QA (RAG) agent. Kept optional so
        # existing deployments without DVD_MCP_SERVER still start; the DVD endpoints
        # raise a clear error if it is unset (see dependencies.get_dvd_mcp_client).
        self.DVD_MCP_URL = dvd_mcp_url or None
        if not chat_storage_url:
            raise ValueError("CHAT_STORAGE_URL must be set")
        self.CHAT_STORAGE_URL = chat_storage_url
        if not urban_api_url:
            raise ValueError("URBAN_API_URL must be set")
        self.URBAN_API_URL = urban_api_url
        self.REDIS_URL = redis_url
        self.SYSTEM_PASSWORD = system_password

    def to_dict(self) -> dict[str, str]:

        return {
            "OLLAMA_URL": self.OLLAMA_URL,
            "IDU_MCP_URL": self.IDU_MCP_URL,
            "EFFECTS_MCP_URL": self.EFFECTS_MCP_URL,
            "DVD_MCP_URL": self.DVD_MCP_URL or "",
            "CHAT_STORAGE_URL": self.CHAT_STORAGE_URL,
            "URBAN_API_URL": self.URBAN_API_URL,
            "REDIS_URL": self.REDIS_URL,
        }

    def __repr__(self) -> str:

        return str(self.to_dict())
