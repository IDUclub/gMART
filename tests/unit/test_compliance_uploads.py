"""Parsing, roles and storage of user files for a compliance check."""

import io
import json
import zipfile
from pathlib import Path

import fakeredis.aioredis
import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from src.agents.common.exceptions.base_exceptions import (
    AgentsInputException,
    AgentsNotFound,
)
from src.agents.services.compilance.compliance_uploads import (
    ComplianceUploadService,
    UploadTooLarge,
    detect_role,
    parse_upload,
)
from src.agents.services.pipeline_state import PipelineStateStore

SQUARE = [
    [37.48, 55.67],
    [37.49, 55.67],
    [37.49, 55.68],
    [37.48, 55.68],
    [37.48, 55.67],
]


def collection(features, crs="urn:ogc:def:crs:OGC:1.3:CRS84"):
    data = {"type": "FeatureCollection", "features": features}
    if crs:
        data["crs"] = {"type": "name", "properties": {"name": crs}}
    return data


def polygon(properties, ring=SQUARE):
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": properties,
    }


def as_bytes(data, bom=False):
    text = json.dumps(data, ensure_ascii=False)
    return (("﻿" if bom else "") + text).encode("utf-8")


def test_geojson_with_bom_and_mixed_z_becomes_planar_wgs84():
    ring = [
        [37.48, 55.67, 0],
        [37.49, 55.67],
        [37.49, 55.68, 0.0],
        [37.48, 55.68],
        [37.48, 55.67],
    ]
    layers = parse_upload(
        "город сад (5).geojson",
        as_bytes(collection([polygon({"type": "buildings"}, ring)]), bom=True),
    )
    assert len(layers) == 1
    coordinates = layers[0].features[0]["geometry"]["coordinates"][0]
    assert all(len(position) == 2 for position in coordinates)
    assert layers[0].name == "город сад (5)"
    assert layers[0].role == "project"


def test_projected_crs_is_reprojected_to_wgs84():
    feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [4174000.0, 7495000.0]},
        "properties": {},
    }
    layer = parse_upload(
        "points.geojson",
        as_bytes(collection([feature], crs="urn:ogc:def:crs:EPSG::3857")),
    )[0]
    lon, lat = layer.features[0]["geometry"]["coordinates"]
    assert 37 < lon < 38 and 55 < lat < 56


def test_coordinates_out_of_lon_lat_range_without_crs_are_rejected():
    feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [4174000.0, 7495000.0]},
        "properties": {},
    }
    with pytest.raises(AgentsInputException, match="crs"):
        parse_upload("points.geojson", as_bytes(collection([feature], crs=None)))


@pytest.mark.parametrize(
    ("name", "features", "role"),
    [
        ("Границы (3)", [polygon({"type": "border"})], "border"),
        ("336195_border", [polygon({})], "border"),
        ("336195_osm_surroundings", [polygon({"type": "buildings"})], "context"),
        ("сущпол по ППТ (4)", [polygon({"type": "buildings"})], "context"),
        ("336195_map_objects", [polygon({"type": "buildings"})], "project"),
    ],
)
def test_role_is_detected_from_name_and_content(name, features, role):
    assert detect_role(name, features) == role


def test_zip_with_geojson_and_shapefile(tmp_path: Path):
    frame = gpd.GeoDataFrame(
        {"building": ["residential"], "levels": [9]},
        geometry=[box(37.48, 55.67, 37.49, 55.68)],
        crs="EPSG:4326",
    ).to_crs("EPSG:32637")
    frame.to_file(tmp_path / "houses.shp")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "set/336195_border.geojson",
            as_bytes(collection([polygon({"type": "border"})])),
        )
        for path in tmp_path.iterdir():
            archive.write(path, f"set/{path.name}")
        archive.writestr("__MACOSX/set/._houses.shp", b"junk")
    layers = {layer.name: layer for layer in parse_upload("set.zip", buffer.getvalue())}
    assert set(layers) == {"336195_border", "houses"}
    assert layers["336195_border"].role == "border"
    house = layers["houses"].features[0]
    assert house["properties"] == {"building": "residential", "levels": 9}
    min_x = min(x for x, _ in house["geometry"]["coordinates"][0])
    assert min_x == pytest.approx(37.48, abs=1e-6)


def test_shapefile_without_sidecars_is_rejected(tmp_path: Path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("houses.shp", b"not a shapefile")
    with pytest.raises(AgentsInputException, match=".shx"):
        parse_upload("houses.zip", buffer.getvalue())


def test_geopackage_layers(tmp_path: Path):
    path = tmp_path / "design.gpkg"
    gpd.GeoDataFrame(
        {"type": ["school"]}, geometry=[Point(37.48, 55.67)], crs="EPSG:4326"
    ).to_file(path, layer="schools")
    gpd.GeoDataFrame(
        {"type": ["border"]},
        geometry=[box(37.47, 55.66, 37.5, 55.69)],
        crs="EPSG:4326",
    ).to_file(path, layer="border")
    layers = {
        layer.name: layer for layer in parse_upload("design.gpkg", path.read_bytes())
    }
    assert set(layers) == {"design:schools", "design:border"}
    assert layers["design:border"].role == "border"


@pytest.mark.parametrize(
    ("filename", "message"),
    [("houses.shp", "ZIP"), ("plan.dxf", "не поддерживается")],
)
def test_unsupported_inputs(filename, message):
    with pytest.raises(AgentsInputException, match=message):
        parse_upload(filename, b"data")


def test_upload_size_limit(monkeypatch):
    monkeypatch.setattr(
        "src.agents.services.compilance.compliance_uploads.MAX_UPLOAD_BYTES", 10
    )
    with pytest.raises(UploadTooLarge):
        parse_upload("big.geojson", b"x" * 11)


def test_file_without_geometry_is_rejected():
    data = collection([{"type": "Feature", "geometry": None, "properties": {}}])
    with pytest.raises(AgentsInputException, match="нет объектов"):
        parse_upload("empty.geojson", as_bytes(data))


@pytest.fixture
async def uploads():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield ComplianceUploadService(PipelineStateStore(redis))
    await redis.aclose()


def files():
    return [
        ("map_objects.geojson", as_bytes(collection([polygon({"type": "buildings"})]))),
        ("border.geojson", as_bytes(collection([polygon({"type": "border"})]))),
    ]


async def test_upload_is_stored_for_its_owner_only(uploads):
    summary = await uploads.create("alice", files())
    assert [layer["role"] for layer in summary["layers"]] == ["project", "border"]
    record = await uploads.get(summary["upload_id"], "alice")
    assert record["files"] == ["map_objects.geojson", "border.geojson"]
    with pytest.raises(AgentsNotFound):
        await uploads.get(summary["upload_id"], "bob")


async def test_role_override_at_upload_and_by_layer(uploads):
    summary = await uploads.create("alice", files(), role="context")
    assert {layer["role"] for layer in summary["layers"]} == {"context"}
    updated = await uploads.set_roles(
        summary["upload_id"], "alice", {"map_objects": "project"}
    )
    roles = {layer["name"]: layer["role"] for layer in updated["layers"]}
    assert roles == {"map_objects": "project", "border": "context"}
    with pytest.raises(AgentsInputException, match="нет слоёв"):
        await uploads.set_roles(summary["upload_id"], "alice", {"other": "border"})


async def test_confirmed_mapping_is_loaded_with_the_upload(uploads):
    first = await uploads.create("alice", files()[:1])
    second = await uploads.create("alice", files()[1:])
    ids = [first["upload_id"], second["upload_id"]]
    await uploads.confirm_mapping(
        ids,
        "alice",
        {
            "type=calm_rest": None,
            "type=sport": {"entity": "Спортивная площадка", "entity_type": "service"},
        },
    )
    loaded = await uploads.load(ids, "alice")
    assert [layer.name for layer in loaded.layers] == ["map_objects", "border"]
    assert loaded.has_project
    assert loaded.confirmed_mapping["type=calm_rest"] is None
    assert loaded.confirmed_mapping["type=sport"]["entity"] == "Спортивная площадка"


async def test_duplicate_layer_names_are_made_unique(uploads):
    summary = await uploads.create("alice", files()[:1] * 2)
    assert [layer["name"] for layer in summary["layers"]] == [
        "map_objects",
        "map_objects (2)",
    ]
