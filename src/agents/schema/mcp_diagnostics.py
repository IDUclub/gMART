from typing import Any, Literal

from pydantic import BaseModel, Field


class McpToolCallRequest(BaseModel):
    """A tool invocation submitted from the MCP diagnostics console."""

    name: str = Field(min_length=1)
    source: Literal["idu", "urban", "effects", "dvd", "normgraph"] = "idu"
    group: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
