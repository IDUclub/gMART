from typing import Any

from pydantic import BaseModel, Field


class SynapseRunRequestDTO(BaseModel):
    request: str = Field(min_length=1, max_length=50_000)
    chat_id: str | None = None
    scenario_id: int
    project_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
