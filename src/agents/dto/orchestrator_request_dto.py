from pydantic import Field

from src.agents.dto.llm_request_dto import SimpleRequestDTO


class OrchestratorRequestDTO(SimpleRequestDTO):
    """
    Request DTO for the orchestrator REST endpoint.
    Extends SimpleRequestDTO with Urban API context required by sub-pipelines.

    Attributes:
        scenario_id (int): Scenario ID — passed to both restriction and provision pipelines.
        project_id (int | None): Project ID for the provision effects calculation.
            Optional: if omitted and the query requires provision, that sub-pipeline
            is skipped with a routing notice.
        chat_id (str | None): Chat Storage UUID for history continuity.
        request_id (str | None): Existing pipeline request ID to reconnect/resume
            a suspended execution.
    """

    scenario_id: int = Field(
        examples=[772],
        description="Scenario ID from Urban API",
    )
    project_id: int | None = Field(
        default=None,
        examples=[1],
        description=(
            "Project ID passed to the provision effects MCP server. "
            "Required when the query involves provision effects calculation."
        ),
    )
    chat_id: str | None = Field(
        default=None,
        min_length=36,
        max_length=36,
        examples=["550e8400-e29b-41d4-a716-446655440000"],
        description="Chat ID from Chat Storage",
    )
    request_id: str | None = Field(
        default=None,
        min_length=36,
        max_length=36,
        examples=["550e8400-e29b-41d4-a716-446655440001"],
        description=(
            "Existing pipeline request ID. "
            "Pass to reconnect and resume a suspended pipeline."
        ),
    )
