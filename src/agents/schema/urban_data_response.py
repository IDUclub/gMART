from typing import Any, Literal

from geojson_pydantic import FeatureCollection
from pydantic import BaseModel

from src.agents.common.exceptions.sse_exceptions import SseBaseError


class UrbanDataStatusResponse(BaseModel):
    """Status update for a stage of the urban-data QA tool-calling loop."""

    status: Literal[
        "tools_loading",
        "executing",
        "answer_drafting",
    ]
    text: str


class TextResponse(BaseModel):
    text: str
    done: bool


class FeatureCollectionResponse(BaseModel):
    name: str
    feature_collection: FeatureCollection


class ToolCallContent(BaseModel):
    """Describes the Urban MCP tool call(s) executed during a round."""

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


class PipelineEventContent(BaseModel):
    request_id: str
    message: str


class WarningContent(BaseModel):
    """
    Non-fatal notice — the pipeline continues despite a degraded step. Emitted, e.g.,
    when the project_id could not be resolved from scenario_id, so the chat is created
    without the project filter.
    """

    code: str
    message: str
    scenario_id: int | None = None


class UrbanDataResponse(BaseModel):
    """SSE event envelope for the urban-data QA (external grouped Urban MCP) pipeline."""

    type: Literal[
        "status",
        "chunk",
        "tool_call",
        "feature_collection",
        "service_event",
        "pipeline_started",
        "warning",
        "token_expired",
        "pipeline_suspended",
        "error",
    ]
    content: (
        UrbanDataStatusResponse
        | TextResponse
        | ToolCallContent
        | FeatureCollectionResponse
        | ServiceEvent
        | PipelineStartedContent
        | PipelineEventContent
        | WarningContent
        | SseBaseError
    )
