from typing import Any, Literal

from geojson_pydantic import FeatureCollection
from pydantic import BaseModel

from src.agents.common.exceptions.sse_exceptions import SseBaseError


class ProvisionStatusResponse(BaseModel):
    status: Literal[
        "service_lookup",
        "effects_calculation",
        "response_analysis",
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
    request_id: str


class PipelineEventContent(BaseModel):
    request_id: str
    message: str


class ToolCallContent(BaseModel):
    execution_mode: str
    tool_calls: list[Any]


class TableColumn(BaseModel):
    key: str
    label: str


class TableContent(BaseModel):
    """Strict table rendered by code: column keys/labels never change."""

    name: str
    title: str
    columns: list[TableColumn]
    rows: list[dict[str, Any]]


class ProvisionResponse(BaseModel):
    type: Literal[
        "status",
        "chunk",
        "feature_collection",
        "table",
        "error",
        "service_event",
        "pipeline_started",
        "token_expired",
        "pipeline_suspended",
        "tool_call",
    ]
    content: (
        ProvisionStatusResponse
        | TextResponse
        | FeatureCollectionResponse
        | TableContent
        | SseBaseError
        | ServiceEvent
        | PipelineStartedContent
        | PipelineEventContent
        | ToolCallContent
    )
