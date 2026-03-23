from typing import Literal

from geojson_pydantic import FeatureCollection
from pydantic import BaseModel


class StatusResponse(BaseModel):

    status: Literal[
        "data_retrievement",
        "buffer_creation" "restriction_formation",
        "context_preparation",
    ]
    text: str


class TextResponse(BaseModel):

    text: str
    done: bool


class FeatureCollectionResponse(BaseModel):

    name: str
    layer: FeatureCollection


class RestrictionsResponse(BaseModel):

    type: Literal["status", "chunk", "layer"]
    content: StatusResponse | TextResponse | FeatureCollectionResponse
