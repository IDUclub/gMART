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
        "buffer_creation",
        "restriction_formation",
        "context_preparation",
    ]
    text: str


class TextResponse(BaseModel):

    text: str = Field(default="")
    done: bool


class FeatureCollectionResponse(BaseModel):

    name: str
    feature_collection: FeatureCollection


class RestrictionsResponse(BaseModel):

    type: Literal["status", "chunk", "feature_collection", "error"]
    content: StatusResponse | TextResponse | FeatureCollectionResponse | SseBaseError
