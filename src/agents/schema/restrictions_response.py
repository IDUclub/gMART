from typing import Literal

from geojson_pydantic import FeatureCollection
from pydantic import BaseModel, Field

from src.agents.common.exceptions.sse_exceptions import SseBaseError


class StatusResponse(BaseModel):
    """
    Class for status response.
    Attributes:
        status (Literal["data_retrievement", "buffer_creation","restriction_formation","context_preparation",]): status
        stage name.
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


class RestrictionsResponse(BaseModel):

    type: Literal["status", "chunk", "feature_collection", "error", "service_event"]
    content: (
        StatusResponse
        | TextResponse
        | FeatureCollectionResponse
        | SseBaseError
        | ServiceEvent
    )
