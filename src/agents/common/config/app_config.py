from src.common.auth.auth_config import AuthConfig


class AgentsAppConfig:
    """
    FastAPI agents service configuration.
    Attributes:
        OLLAMA_URL (str): Ollama URL.
        IDU_MCP_URL (str): IDU MCP URL.
        EFFECTS_MCP_URL (str): Object Effects MCP URL.
        CHAT_STORAGE_URL (str): Chat Storage service URL.
        REDIS_URL (str): Redis URL.
        AUTH_CONFIG (AuthConfig): Keycloak auth settings.
    """

    OLLAMA_URL: str
    IDU_MCP_URL: str
    EFFECTS_MCP_URL: str
    CHAT_STORAGE_URL: str
    REDIS_URL: str
    AUTH_CONFIG: AuthConfig

    def __init__(
        self,
        ollama_api_url: str,
        idu_mcp_url: str,
        effects_mcp_url: str,
        chat_storage_url: str,
        auth_config: AuthConfig,
        redis_url: str = "redis://localhost:6379",
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
        if not chat_storage_url:
            raise ValueError("CHAT_STORAGE_URL must be set")
        self.CHAT_STORAGE_URL = chat_storage_url
        self.REDIS_URL = redis_url
        self.AUTH_CONFIG = auth_config

    def __repr__(self) -> str:
        return str(
            {
                "OLLAMA_URL": self.OLLAMA_URL,
                "IDU_MCP_URL": self.IDU_MCP_URL,
                "EFFECTS_MCP_URL": self.EFFECTS_MCP_URL,
                "CHAT_STORAGE_URL": self.CHAT_STORAGE_URL,
                "REDIS_URL": self.REDIS_URL,
                "AUTH_VERIFY": self.AUTH_CONFIG.verify,
                "AUTH_SERVER_URL": self.AUTH_CONFIG.server_url,
            }
        )
