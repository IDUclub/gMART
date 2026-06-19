from pydantic import BaseModel


class AppConfigResponse(BaseModel):
    """
    Response model with the current agents service runtime configuration.
    The system password is intentionally excluded from this response.
    """

    OLLAMA_URL: str
    IDU_MCP_URL: str
    EFFECTS_MCP_URL: str
    CHAT_STORAGE_URL: str
    URBAN_API_URL: str
    REDIS_URL: str
