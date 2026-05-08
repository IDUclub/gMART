from pydantic import Field

from src.agents.dto.llm_request_dto import SimpleRequestDTO


class RestrictionRequestDTO(SimpleRequestDTO):
    """
    Restriction Request DTO to LLM.
    Attributes:
        model (str): Model name.
        request (str): Request text.
        scenario_id (int): Scenario ID from Urban API.
        chat_id (str | None): String representation of chat uuid from Chat Storage. Default to None.
    """

    scenario_id: int = Field(examples=[772], description="Scenario ID from Urban API")
    chat_id: str | None = Field(
        min_length=36,
        max_length=36,
        examples=["550e8400-e29b-41d4-a716-446655440000"],
        default=None,
        description="Chat ID from Chat Storage",
    )
