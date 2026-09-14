"""Lossless operation data, referenced by the model instead of regenerated."""

from copy import deepcopy
from math import isfinite

from geojson_pydantic import FeatureCollection

ZONE_KINDS = {
    "residential",
    "business",
    "industrial",
    "recreation",
    "agriculture",
    "special",
    "transport",
}


def restore_existing_building_attributes(generated, existing):
    """Recover source attributes lost in GenBuilder's unchanged-object placeholders."""
    FeatureCollection.model_validate(generated)
    FeatureCollection.model_validate(existing)
    source = {}
    for feature in existing["features"]:
        props = feature.get("properties") or {}
        identifier = props.get("physical_object_id")
        if identifier is not None:
            # One physical object can have several Urban geometries. GenBuilder
            # retains only the object ID; do not assign an arbitrary geometry ID.
            attributes = {k: v for k, v in props.items() if k != "object_geometry_id"}
            if identifier in source and source[identifier] != attributes:
                raise ValueError(
                    "Conflicting attributes for an existing physical object"
                )
            source[identifier] = attributes
    result = deepcopy(generated)
    for feature in result["features"]:
        props = feature.get("properties") or {}
        identifier = props.get("physical_object_id")
        if props.get("is_excluded") and identifier in source:
            restored = deepcopy(source[identifier])
            building = restored.get("building") or {}
            restored.update(
                {
                    "is_excluded": True,
                    "attribute_source": "existing_buildings_by_physical_object_id",
                    "floors_count": building.get("floors"),
                }
            )
            feature["properties"] = restored
    return result


def property_value(feature, path):
    value = feature.get("properties") or {}
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def select_layer(layer, property_path, values):
    """Select actual features, including those omitted from the model preview."""
    FeatureCollection.model_validate(layer)
    return {
        "type": "FeatureCollection",
        "features": [
            deepcopy(f)
            for f in layer["features"]
            if property_value(f, property_path) in values
        ],
    }


def layer_values(layer, property_path):
    FeatureCollection.model_validate(layer)
    return [property_value(f, property_path) for f in layer["features"]]


def prepare_zoning_constraints(layer, editable_zone_kinds):
    """Derive preservation IDs from every feature of one actual zoning version."""
    FeatureCollection.model_validate(layer)
    if not editable_zone_kinds or not set(editable_zone_kinds) <= ZONE_KINDS:
        raise ValueError("Explicit supported editable zone kinds are required")
    fixed, editable, versions = [], [], set()
    for feature in layer["features"]:
        props = feature.get("properties") or {}
        kind = (props.get("functional_zone_type") or {}).get("name")
        identifier = props.get("functional_zone_id")
        if not kind or not isinstance(identifier, int) or isinstance(identifier, bool):
            raise ValueError(
                "Use the full Urban functional-zone layer with type and ID"
            )
        versions.add((props.get("year"), props.get("source")))
        (editable if kind in editable_zone_kinds else fixed).append(identifier)
    if not editable or len(versions) != 1:
        raise ValueError(
            "One zoning version with at least one editable polygon is required"
        )
    year, source = versions.pop()
    if not isinstance(year, int) or not source:
        raise ValueError("Actual year/source are required")
    return {
        "year": year,
        "source": source,
        "fixed_functional_zones_ids": fixed,
        "editable_functional_zones_ids": editable,
        "fixed_count": len(fixed),
        "editable_count": len(editable),
        "source_feature_count": len(layer["features"]),
    }


def summarize_layer(layer, numeric_properties):
    """Return exact sums and explicit missing counts instead of sampled estimates."""
    FeatureCollection.model_validate(layer)
    features = layer["features"]
    rows = []
    for name in numeric_properties:
        values = [property_value(f, [name]) for f in features]
        numbers = [
            v
            for v in values
            if isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v)
        ]
        rows.append(
            {
                "property": name,
                "sum": sum(numbers),
                "known": len(numbers),
                "missing_or_invalid": len(values) - len(numbers),
            }
        )
    return {"feature_count": len(features), "statistics": rows}


def propose_service(layer, service_type_id, capacity):
    """A candidate location inside one supplied polygon, not a building footprint."""
    from shapely.geometry import mapping, shape

    FeatureCollection.model_validate(layer)
    if len(layer["features"]) != 1:
        raise ValueError("Select exactly one candidate site")
    geom = shape(layer["features"][0]["geometry"])
    if (
        geom.geom_type not in {"Polygon", "MultiPolygon"}
        or not geom.is_valid
        or geom.is_empty
    ):
        raise ValueError("Candidate site must be a valid polygon")
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, (int, float))
        or not isfinite(capacity)
        or capacity <= 0
    ):
        raise ValueError("Candidate capacity must be explicit and positive")
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(geom.representative_point()),
                "properties": {
                    "service_type_id": service_type_id,
                    "capacity": capacity,
                    "design_status": "candidate_location",
                    "site_properties": deepcopy(layer["features"][0]["properties"]),
                },
            }
        ],
    }


def compare_layer_coverage(before, after, tolerance_m2=0.1):
    """Measure retained polygon area in a local metric CRS, independently of IDs."""
    import geopandas as gpd
    from shapely import union_all

    for layer in (before, after):
        FeatureCollection.model_validate(layer)
    original = gpd.GeoDataFrame.from_features(before["features"], crs=4326)
    changed = gpd.GeoDataFrame.from_features(after["features"], crs=4326)
    if original.empty or changed.empty:
        raise ValueError("Both coverage layers must contain polygons")
    for frame in (original, changed):
        if (
            not frame.geometry.is_valid.all()
            or not frame.geom_type.isin(["Polygon", "MultiPolygon"]).all()
        ):
            raise ValueError("Coverage comparison requires valid polygons")
    crs = original.estimate_utm_crs()
    original, changed = original.to_crs(crs), changed.to_crs(crs)
    retained = union_all(changed.geometry)
    lost = original.geometry.difference(retained).area
    return {
        "original_features": len(original),
        "features_with_area_loss": int((lost > tolerance_m2).sum()),
        "lost_area_m2": float(union_all(original.geometry).difference(retained).area),
        "tolerance_per_feature_m2": tolerance_m2,
        "metric_crs": crs.to_string(),
        "method": "polygon coverage; does not prove preservation of attributes or legal status",
    }


def prepare_building_blocks(layer, zone_kinds):
    """Normalize real GenPlanner/Urban zone labels without touching coordinates."""
    FeatureCollection.model_validate(layer)
    if not zone_kinds or not set(zone_kinds) <= ZONE_KINDS:
        raise ValueError("Select supported zone kinds explicitly")
    result = {"type": "FeatureCollection", "features": []}
    for feature in layer["features"]:
        props = feature.get("properties") or {}
        kind = (
            props.get("zone")
            or props.get("territory_zone_name")
            or (props.get("functional_zone_type") or {}).get("name")
        )
        if kind not in zone_kinds:
            continue
        if not feature.get("geometry") or feature["geometry"]["type"] not in {
            "Polygon",
            "MultiPolygon",
        }:
            raise ValueError("Generation blocks must contain polygon geometry")
        normalized = deepcopy(feature)
        normalized["properties"] = {**props, "zone": kind}
        result["features"].append(normalized)
    if not result["features"]:
        raise ValueError("No polygons of the requested zone kinds in this layer")
    return result


def resolve_references(value, artifacts):
    if isinstance(value, dict):
        if "$artifact" in value:
            if set(value) - {"$artifact", "path"}:
                raise ValueError("Artifact references accept only $artifact and path")
            key = value["$artifact"]
            if key not in artifacts:
                raise ValueError(f"Unknown artifact: {key}")
            result = artifacts[key]
            for part in value.get("path", []):
                result = result[int(part)] if isinstance(result, list) else result[part]
            return deepcopy(result)
        return {k: resolve_references(v, artifacts) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_references(v, artifacts) for v in value]
    return value


def inspect_value(value, offset=0, limit=6):
    """Read a bounded page from full evidence, including past the initial preview."""
    if offset < 0 or not 1 <= limit <= 6:
        raise ValueError("Inspection requires offset >= 0 and 1 <= limit <= 6")
    if isinstance(value, list):
        return {
            "items": deepcopy(value[offset : offset + limit]),
            "offset": offset,
            "total": len(value),
            "complete": offset + limit >= len(value),
        }
    return {"value": deepcopy(value)}


def preview(value, depth=0):
    if depth > 5:
        return {"omitted": True, "type": type(value).__name__}
    if isinstance(value, dict):
        if value.get("type") == "FeatureCollection":
            features = value.get("features", [])
            return {
                "type": "FeatureCollection",
                "feature_count": len(features),
                "properties_sample": [f.get("properties") for f in features[:3]],
                "geometry_omitted": True,
            }
        return {k: preview(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        limit = (
            32
            if value
            and all(
                isinstance(v, dict) and {"id", "kind", "profile"} <= v.keys()
                for v in value
            )
            else 6
        )
        return {
            "items": [preview(v, depth + 1) for v in value[:limit]],
            "total": len(value),
            "complete": len(value) <= limit,
        }
    if isinstance(value, str) and len(value) > 2500:
        return {"text": value[:2500], "truncated": True}
    return value


def table_event(name, rows, title=None):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    return {
        "type": "table",
        "content": {
            "name": name,
            "title": title or name,
            "columns": [{"key": k, "label": k} for k in keys],
            "rows": rows,
            "total_rows": len(rows),
            "complete": True,
        },
    }


def result_events(value, name):
    """Emit complete layers and exact service properties, never model-made rows."""
    if isinstance(value, dict):
        if value.get("type") == "FeatureCollection":
            FeatureCollection.model_validate(value)
            yield {
                "type": "feature_collection",
                "content": {
                    "name": name,
                    "feature_collection": value,
                },
            }
            rows = [
                {"feature_id": f.get("id", i), **(f.get("properties") or {})}
                for i, f in enumerate(value["features"])
            ]
            yield table_event(name + "_properties", rows)
            return
        scalar = {
            k: v
            for k, v in value.items()
            if v is None or isinstance(v, (str, int, float, bool))
        }
        if scalar:
            yield table_event(name, [scalar])
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                yield from result_events(v, name + "_" + k)
    elif isinstance(value, list) and value:
        if all(
            isinstance(v, dict) and v.get("type") != "FeatureCollection" for v in value
        ):
            yield table_event(name, value)
        else:
            for i, item in enumerate(value):
                yield from result_events(item, f"{name}_{i}")
