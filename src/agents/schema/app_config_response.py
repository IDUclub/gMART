from pydantic import BaseModel


class AppConfigResponse(BaseModel):
    """Public view of the agents service runtime configuration."""

    OLLAMA_URL: str
    IDU_MCP_URL: str
    EFFECTS_MCP_URL: str
    CHAT_STORAGE_URL: str
    REDIS_URL: str
