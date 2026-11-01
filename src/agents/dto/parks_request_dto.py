from enum import StrEnum
from typing import Any

from geojson_pydantic import FeatureCollection
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agents.dto.norms_request_dto import NormsQaRequestDTO


class ParkTag(StrEnum):
    ART = "ART"
    EXHIBITION = "EXHIBITION"
    TRADE = "TRADE"
    FOOD = "FOOD"
    REST = "REST"
    CHILDREN = "CHILDREN"
    BICYCLE = "BICYCLE"
    SPORTS = "SPORTS"
    PARKING = "PARKING"
    WATER = "WATER"
    ANIMALS = "ANIMALS"
    AMUSEMENT = "AMUSEMENT"
    GREENERY = "GREENERY"


class LocationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    text: str = Field(min_length=1, max_length=4000)


class ParksInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    park_geojson: dict[str, Any] | None = None
    tags: list[ParkTag] | None = Field(default=None, max_length=200)
    location_requests: list[LocationRequest] | None = Field(
        default=None, max_length=200
    )
    confirmed_revision: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_geometry(self):
        if self.park_geojson is None:
            return self
        collection = FeatureCollection.model_validate(self.park_geojson)
        boundaries = []
        for feature in collection.features:
            kind = (feature.properties or {}).get("type")
            geometry = feature.geometry
            if (
                kind == "boundary"
                and geometry
                and geometry.type in {"Polygon", "MultiPolygon"}
            ):
                boundaries.append(geometry)
            elif kind != "entry" or not geometry or geometry.type != "Point":
                raise ValueError("Use one polygon boundary and optional entry points")
        if len(boundaries) != 1:
            raise ValueError("Exactly one boundary is required")
        from shapely.geometry import shape

        for feature in self.park_geojson["features"]:
            geometry = shape(feature["geometry"])
            if not geometry.is_valid or geometry.is_empty:
                raise ValueError("Invalid park geometry")
            if any(
                not (-180 <= x <= 180 and -90 <= y <= 90)
                for x, y, *_ in geometry_coordinates(feature["geometry"])
            ):
                raise ValueError("Coordinates must be WGS84 longitude/latitude")
        return self


def geometry_coordinates(geometry):
    def walk(value):
        if value and isinstance(value[0], (int, float)):
            yield value
        else:
            for child in value:
                yield from walk(child)

    yield from walk(geometry["coordinates"])


class ParksRequestDTO(NormsQaRequestDTO):
    inputs: ParksInputs = Field(default_factory=ParksInputs)
