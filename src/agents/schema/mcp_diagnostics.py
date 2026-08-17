from typing import Any

from pydantic import BaseModel, Field


class McpToolCallRequest(BaseModel):
    """A tool invocation submitted from the MCP diagnostics console."""

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
