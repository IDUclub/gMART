from pydantic import Field

from src.agents.dto.llm_request_dto import SimpleRequestDTO


class OrchestratorRequestDTO(SimpleRequestDTO):
    """
    DTO for the orchestrator endpoint — the single entry point for all agents.

    The orchestrator plans which agents should handle the request and runs them
    sequentially; scenario_id gates the availability of the restriction and
    provision agents.
    Attributes:
        scenario_id (int | None): Scenario ID from Urban API. Required for
            restriction/provision steps; without it the planner only routes to
            the documents/norms agents.
        chat_id (str | None): Chat Storage UUID for history continuity.
        request_id (str | None): Existing pipeline request ID — pass it to
            replay the buffered events of an interrupted stream.
    """

    scenario_id: int | None = Field(
        default=None,
        examples=[772],
        description=(
            "Scenario ID from Urban API (optional). Required for the restriction "
            "and provision agents; when chat_id is not provided, a new chat is "
            "created in ChatStorage tagged with this scenario_id."
        ),
    )
    chat_id: str | None = Field(
        min_length=36,
        max_length=36,
        default=None,
        examples=["550e8400-e29b-41d4-a716-446655440000"],
        description="Chat ID from Chat Storage for history continuity",
    )
    request_id: str | None = Field(
        min_length=36,
        max_length=36,
        default=None,
        examples=["550e8400-e29b-41d4-a716-446655440001"],
        description=(
            "Existing pipeline request ID (from the pipeline_started event). "
            "Pass it to reconnect to an interrupted stream and replay its events."
        ),
    )
