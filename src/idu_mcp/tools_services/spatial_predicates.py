"""Exact predicates shared by geometry retrieval and normative checks."""

from copy import deepcopy

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def exact_spatial_predicate(
    object_geometry: BaseGeometry,
    zone_geometry: BaseGeometry,
    predicate: str = "intersects",
) -> bool:
    """Evaluate the object relative to the zone, including the actual boundary.

    ``within`` requires the entire object to be within the zone; neither bounding
    boxes nor centroids are used. ``contains`` retains the older Check* contract.
    """
    if predicate not in {"intersects", "within", "contains"}:
        raise ValueError(f"Unknown spatial predicate {predicate!r}")
    if object_geometry.is_empty or zone_geometry.is_empty:
        return False
    return bool(getattr(object_geometry, predicate)(zone_geometry))


def filter_objects_by_zones(
    zones: dict, objects: dict, predicate: str = "intersects"
) -> dict:
    """Pure GeoJSON helper; apply the predicate against the exact union of zones."""
    geometries = [
        shape(feature["geometry"])
        for feature in zones.get("features", [])
        if feature.get("geometry")
    ]
    area = unary_union(geometries)
    result = deepcopy(objects)
    result["features"] = [
        feature
        for feature in result.get("features", [])
        if feature.get("geometry")
        and exact_spatial_predicate(shape(feature["geometry"]), area, predicate)
    ]
    return result
