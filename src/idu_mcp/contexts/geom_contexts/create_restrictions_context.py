from geojson_pydantic import FeatureCollection
from pydantic import BaseModel


class RestrictionsContext(BaseModel):

    layers: dict[str, FeatureCollection]
