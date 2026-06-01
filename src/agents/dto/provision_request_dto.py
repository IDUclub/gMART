from pydantic import Field

from src.agents.dto.llm_request_dto import SimpleRequestDTO


class ProvisionRequestDTO(SimpleRequestDTO):
    """
    DTO for the provision effects pipeline endpoint.
    The service name and target population are extracted from the natural language
    request by the pipeline via LLM + IDU MCP service catalog.
    Attributes:
        scenario_id (int): Scenario ID — passed to MCP servers as a tool argument.
        chat_id (str | None): Chat Storage UUID for history continuity.
        request_id (str | None): Existing pipeline request ID to reconnect/resume.
    """

    scenario_id: int = Field(
        examples=[772],
        description="Scenario ID from Urban API",
    )
    chat_id: str | None = Field(
        min_length=36,
        max_length=36,
        examples=["550e8400-e29b-41d4-a716-446655440000"],
        default=None,
        description="Chat ID from Chat Storage",
    )
    request_id: str | None = Field(
        min_length=36,
        max_length=36,
        examples=["550e8400-e29b-41d4-a716-446655440001"],
        default=None,
        description="Existing pipeline request ID. Pass to reconnect and resume a suspended pipeline.",
    )
