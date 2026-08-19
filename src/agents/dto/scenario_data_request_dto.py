from pydantic import Field

from src.agents.dto.llm_request_dto import SimpleRequestDTO


class ScenarioDataRequestDTO(SimpleRequestDTO):
    scenario_id: int | None = Field(
        default=None,
        examples=[772],
        description=(
            "Optional scenario ID from Urban API. Without it the agent only exposes "
            "tools that do not require scenario context."
        ),
    )
    chat_id: str | None = Field(
        min_length=36,
        max_length=36,
        default=None,
        description="Chat ID from Chat Storage",
    )
    request_id: str | None = Field(
        min_length=36,
        max_length=36,
        default=None,
        description="Existing pipeline request ID for buffered event replay",
    )
