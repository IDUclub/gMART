from typing import Any, Literal

from pydantic import BaseModel

from src.agents.common.exceptions.sse_exceptions import SseBaseError


class NormsStatusResponse(BaseModel):
    """Status update for a stage of the iterative NormGraph QA pipeline."""

    status: Literal[
        "retrieval_planning",
        "executing",
        "conflict_check",
        "answer_drafting",
        "self_review",
        "finalizing",
    ]
    text: str


class NormsTextResponse(BaseModel):
    """
    A streamed chunk of a drafted answer.
    Attributes:
        text (str): Chunk text.
        done (bool): True only on the final chunk of the accepted answer.
        iteration (int): Draft number the chunk belongs to (1-based); lets the frontend
            replace a rejected draft when the iteration increments.
    """

    text: str
    done: bool
    iteration: int = 0


class ToolCallContent(BaseModel):
    """Describes the NormGraph call(s) executed during a round."""

    execution_mode: str
    tool_calls: list[Any]
    mcp_source: str | None = None


class ChatCreatedEvent(BaseModel):
    storage_event_type: Literal["chat_created"]
    chat_id: str
    chat_title: str


class ServiceEvent(BaseModel):
    event_type: Literal["storage_event"]
    event: ChatCreatedEvent


class PipelineStartedContent(BaseModel):
    """
    Emitted once at the start of a fresh pipeline run. Carries the ``request_id`` the
    client must echo back (as ``request_id``) to reconnect after a connection drop.
    """

    request_id: str


class WarningContent(BaseModel):
    """
    Non-fatal notice — the pipeline continues despite a degraded step. Emitted, e.g.,
    when the project_id could not be resolved from scenario_id, so the chat is created
    without the project filter.
    """

    code: str
    message: str
    scenario_id: int | None = None


class NormsResponse(BaseModel):
    """SSE event envelope for the normative-restrictions QA (NormGraph graph-RAG) pipeline."""

    type: Literal[
        "status",
        "chunk",
        "tool_call",
        "service_event",
        "pipeline_started",
        "warning",
        "error",
    ]
    content: (
        NormsStatusResponse
        | NormsTextResponse
        | ToolCallContent
        | ServiceEvent
        | PipelineStartedContent
        | WarningContent
        | SseBaseError
    )
