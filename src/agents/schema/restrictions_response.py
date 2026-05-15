from typing import Any, Literal

from geojson_pydantic import FeatureCollection
from pydantic import BaseModel

from src.agents.common.exceptions.sse_exceptions import SseBaseError


class StatusResponse(BaseModel):
    """
    Class for status response.
    Attributes:
        status: Stage name.
        text (str): Status message.
    """

    status: Literal[
        "data_retrievement",
        "plan_explanation",
        "buffer_creation",
        "restriction_formation",
        "context_preparation",
    ]
    text: str


class TextResponse(BaseModel):
    text: str
    done: bool


class FeatureCollectionResponse(BaseModel):
    name: str
    feature_collection: FeatureCollection


class ChatCreatedEvent(BaseModel):
    storage_event_type: Literal["chat_created"]
    chat_id: str
    chat_title: str


class ServiceEvent(BaseModel):
    event_type: Literal["storage_event"]
    event: ChatCreatedEvent


class PipelineStartedContent(BaseModel):
    """Emitted once at the start of every pipeline run."""

    request_id: str


class PipelineEventContent(BaseModel):
    """
    Generic pipeline notification that carries a request_id and a
    human-readable message.  Used for ``token_expired`` and
    ``pipeline_suspended`` events.
    """

    request_id: str
    message: str


class ToolCallContent(BaseModel):
    """Describes MCP tool calls executed during the pipeline step."""

    execution_mode: str
    tool_calls: list[Any]


class RestrictionsResponse(BaseModel):
    type: Literal[
        "status",
        "chunk",
        "feature_collection",
        "error",
        "service_event",
        "pipeline_started",
        "token_expired",
        "pipeline_suspended",
        "tool_call",
    ]
    content: (
        StatusResponse
        | TextResponse
        | FeatureCollectionResponse
        | SseBaseError
        | ServiceEvent
        | PipelineStartedContent
        | PipelineEventContent
        | ToolCallContent
    )
