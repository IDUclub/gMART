from pydantic import Field

from src.agents.dto.llm_request_dto import SimpleRequestDTO


class ScenarioDataRequestDTO(SimpleRequestDTO):
    scenario_id: int = Field(examples=[772], description="Scenario ID from Urban API")
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
