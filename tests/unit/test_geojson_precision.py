"""Tests for the GeoJSON coordinate trimming that keeps MCP payloads small.

Covers:
  * rounding of every geometry kind, including nested GeometryCollections;
  * properties and non-geometry numbers being left alone;
  * ``GEOJSON_COORD_PRECISION`` overriding the default (negative = disabled);
  * ``UrbanApiTool.get_entity_by_names`` returning trimmed layers.
"""

from __future__ import annotations

import pytest

from src.idu_mcp.tools_services.entites.object_type_enum import ObjectTypeEnum
from src.idu_mcp.tools_services.geojson_precision import (
    DEFAULT_PRECISION,
    round_feature_collection,
    round_layers,
)
from src.idu_mcp.tools_services.urb_api_tools import UrbanApiTool


def _fc(coordinates, geometry_type="Point") -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": geometry_type, "coordinates": coordinates},
                "properties": {"buffer_size": 123.456789, "name": "школа"},
            }
        ],
    }


def test_default_precision_is_four_decimals():
    assert DEFAULT_PRECISION == 4
    collection = _fc([30.123456789, 59.987654321])
    round_feature_collection(collection)
    assert collection["features"][0]["geometry"]["coordinates"] == [30.1235, 59.9877]


def test_properties_are_untouched():
    collection = _fc([30.123456789, 59.987654321])
    round_feature_collection(collection)
    assert collection["features"][0]["properties"]["buffer_size"] == 123.456789


def test_polygon_rings_and_bbox():
    polygon = [[[30.111111, 59.222222], [30.333333, 59.444444], [30.111111, 59.222222]]]
    collection = _fc(polygon, geometry_type="Polygon")
    collection["bbox"] = [30.111111, 59.222222, 30.333333, 59.444444]
    round_feature_collection(collection)
    assert collection["features"][0]["geometry"]["coordinates"] == [
        [[30.1111, 59.2222], [30.3333, 59.4444], [30.1111, 59.2222]]
    ]
    assert collection["bbox"] == [30.1111, 59.2222, 30.3333, 59.4444]


def test_geometry_collection_is_walked():
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "GeometryCollection",
                    "geometries": [
                        {"type": "Point", "coordinates": [30.123456, 59.123456]},
                        {
                            "type": "LineString",
                            "coordinates": [
                                [30.123456, 59.123456],
                                [30.654321, 59.654321],
                            ],
                        },
                    ],
                },
                "properties": {},
            }
        ],
    }
    round_feature_collection(collection)
    geometries = collection["features"][0]["geometry"]["geometries"]
    assert geometries[0]["coordinates"] == [30.1235, 59.1235]
    assert geometries[1]["coordinates"] == [[30.1235, 59.1235], [30.6543, 59.6543]]


def test_null_geometry_survives():
    collection = _fc([30.1, 59.1])
    collection["features"][0]["geometry"] = None
    round_feature_collection(collection)
    assert collection["features"][0]["geometry"] is None


def test_precision_from_env(monkeypatch):
    monkeypatch.setenv("GEOJSON_COORD_PRECISION", "6")
    collection = _fc([30.123456789, 59.987654321])
    round_feature_collection(collection)
    assert collection["features"][0]["geometry"]["coordinates"] == [30.123457, 59.987654]


def test_negative_precision_disables_rounding(monkeypatch):
    monkeypatch.setenv("GEOJSON_COORD_PRECISION", "-1")
    collection = _fc([30.123456789, 59.987654321])
    round_feature_collection(collection)
    assert collection["features"][0]["geometry"]["coordinates"] == [
        30.123456789,
        59.987654321,
    ]


def test_round_layers_covers_every_layer():
    layers = {
        "школа": _fc([30.123456789, 59.987654321]),
        "детский сад": _fc([31.111111111, 58.222222222]),
    }
    round_layers(layers)
    assert layers["школа"]["features"][0]["geometry"]["coordinates"] == [30.1235, 59.9877]
    assert layers["детский сад"]["features"][0]["geometry"]["coordinates"] == [
        31.1111,
        58.2222,
    ]


class FakeUrbanClient:
    async def get_service_name_id(self, names, token):
        return {name: index for index, name in enumerate(names)}

    async def get_services(self, scenario_id, ids, token):
        return [_fc([30.123456789, 59.987654321]) for _ in ids]


@pytest.mark.asyncio
async def test_urban_api_tool_returns_trimmed_layers():
    tool = UrbanApiTool(FakeUrbanClient())
    layers = await tool.get_entity_by_names(
        1, ["Школа"], ObjectTypeEnum.SERVICE, "token"
    )
    geometry = layers["Школа"].features[0].geometry
    assert list(geometry.coordinates) == [30.1235, 59.9877]
