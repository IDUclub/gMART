"""Coordinate precision trimming for the GeoJSON that crosses the MCP wire.

Urban API returns coordinates with full float precision (~14 decimals). The
agent receives those layers and posts them straight back as ``CreateBuffers`` /
``CreateRestrictions`` arguments, so every extra digit is paid for three or four
times per pipeline — and a large scenario overruns the MCP request body limit.

Rounding to ``GEOJSON_COORD_PRECISION`` decimals (4 by default, ~11 m of
latitude) roughly halves the payload. Only geometry coordinates and bounding
boxes are touched; properties keep their exact values.
"""

import os
from typing import Any

DEFAULT_PRECISION = 4


def coord_precision() -> int:
    """Decimals to keep. A negative value disables rounding entirely."""

    try:
        return int(os.getenv("GEOJSON_COORD_PRECISION", str(DEFAULT_PRECISION)))
    except ValueError:
        return DEFAULT_PRECISION


def _round_coords(value: Any, ndigits: int) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return round(value, ndigits)
    if isinstance(value, (list, tuple)):
        return [_round_coords(item, ndigits) for item in value]
    return value


def round_geometry(geometry: Any, ndigits: int | None = None) -> Any:
    """Round one GeoJSON geometry in place (GeometryCollection included)."""

    ndigits = coord_precision() if ndigits is None else ndigits
    if ndigits < 0 or not isinstance(geometry, dict):
        return geometry
    if isinstance(geometry.get("geometries"), list):
        geometry["geometries"] = [
            round_geometry(item, ndigits) for item in geometry["geometries"]
        ]
    elif "coordinates" in geometry:
        geometry["coordinates"] = _round_coords(geometry["coordinates"], ndigits)
    if "bbox" in geometry:
        geometry["bbox"] = _round_coords(geometry["bbox"], ndigits)
    return geometry


def round_feature_collection(
    feature_collection: Any, ndigits: int | None = None
) -> Any:
    """Round every geometry of a FeatureCollection dict in place."""

    ndigits = coord_precision() if ndigits is None else ndigits
    if ndigits < 0 or not isinstance(feature_collection, dict):
        return feature_collection
    features = feature_collection.get("features")
    if isinstance(features, list):
        for feature in features:
            if isinstance(feature, dict) and feature.get("geometry") is not None:
                feature["geometry"] = round_geometry(feature["geometry"], ndigits)
                if "bbox" in feature:
                    feature["bbox"] = _round_coords(feature["bbox"], ndigits)
    if "bbox" in feature_collection:
        feature_collection["bbox"] = _round_coords(
            feature_collection["bbox"], ndigits
        )
    return feature_collection


def round_layers(layers: dict[str, Any], ndigits: int | None = None) -> dict[str, Any]:
    """Round a ``{layer name: FeatureCollection}`` mapping in place."""

    for layer in layers.values():
        round_feature_collection(layer, ndigits)
    return layers
