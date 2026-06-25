from pydantic import Field

from src.agents.dto.llm_request_dto import SimpleRequestDTO


class DocumentQaRequestDTO(SimpleRequestDTO):
    """
    DTO for the regulatory-documents QA (RAG) endpoint.

    The agent retrieves fragments from IDU_DVD and answers the natural-language question;
    scenario_id / chat_id are optional and used for chat-history persistence.
    Attributes:
        scenario_id (int | None): Scenario ID from Urban API. When chat_id is not provided,
            a new chat is created in ChatStorage tagged with this scenario_id and the
            project_id resolved from it (via Urban API).
        chat_id (str | None): Chat Storage UUID for history continuity.
        request_id (str | None): Existing pipeline request ID — pass it to reconnect to an
            interrupted stream and resume from where it stopped.
    """

    scenario_id: int | None = Field(
        default=None,
        examples=[772],
        description=(
            "Scenario ID from Urban API (optional). When chat_id is not provided, a new "
            "chat is created in ChatStorage tagged with this scenario_id and the "
            "project_id resolved from it."
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
            "Pass it to reconnect to an interrupted stream and resume the pipeline."
        ),
    )
