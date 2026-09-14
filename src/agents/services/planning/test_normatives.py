"""Explicit assessment-only normative inputs; never infer actual legal zoning."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from geojson_pydantic import FeatureCollection


def prepare_test_pzz_inputs(layer, buildings, fixture_path):
    raw = Path(fixture_path).read_bytes()
    fixture = json.loads(raw)
    if fixture.get("source_kind") != "test_mock":
        raise ValueError("PZZ assessment fixture must declare source_kind=test_mock")
    for value in (layer, buildings):
        FeatureCollection.model_validate(value)
    provenance = {
        "kind": "test_mock",
        "fixture_id": fixture["id"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "description": fixture["description"],
        "legal_compliance_claim": False,
    }
    zones, objects = deepcopy(layer), deepcopy(buildings)
    used = set()
    for feature in zones["features"]:
        props = feature.get("properties") or {}
        kind = (
            props.get("zone")
            or props.get("territory_zone_name")
            or (props.get("functional_zone_type") or {}).get("name")
        )
        if kind not in fixture["zones"]:
            raise ValueError(f"No explicit test normative for zone kind {kind!r}")
        definition = fixture["zones"][kind]
        feature["properties"] = {
            **props,
            "zone_code": definition["zone_code"],
            "zone_name": definition["zone_name"],
            "normative_source": provenance,
        }
        used.add(kind)
    for feature in objects["features"]:
        props = feature.get("properties") or {}
        kind = props.get("physical_object_type") or {}
        identifier = (
            props.get("physical_object_type_id")
            or kind.get("physical_object_type_id")
            or kind.get("id")
        )
        # GenBuilder's explicit housing categories, not an inference from location.
        if identifier is None and props.get("building_type") in {
            "low",
            "medium",
            "high",
            "individual",
        }:
            identifier = 4
        if identifier is not None:
            props["physical_object_type_id"] = identifier
            props["original_building_type"] = props.get("building_type")
            props["building_type"] = identifier
        feature["properties"] = props
    if not used or not objects["features"]:
        raise ValueError("Real zone and building layers must be nonempty")
    return {
        "zones": zones,
        "buildings": objects,
        "descriptions": [fixture["zones"][k] for k in sorted(used)],
        "provenance": provenance,
    }
