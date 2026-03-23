from geojson_pydantic import FeatureCollection
from pydantic import BaseModel


class BufferContext(BaseModel):

    objects: dict[str, FeatureCollection]
