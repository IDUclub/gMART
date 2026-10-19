"""Parsing, roles and storage of user files for a compliance check."""

import io
import json
import zipfile
from pathlib import Path

import fakeredis.aioredis
import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon, box

from src.agents.common.exceptions.base_exceptions import (
    AgentsInputException,
    AgentsNotFound,
)
from src.agents.common.files.layer_cache import LayerCache, LayerMissing
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


# A design in МСК-47 zone 2 as QGIS exports it: no ``crs``, the zone only in ``name``.
MSK47_ZONE2_RING = [
    [2077804.261528644, 380959.278000550],
    [2043570.267109855, 409112.136377234],
    [2090985.428587351, 444348.328422423],
    [2151737.998923301, 417014.775082117],
    [2093431.008906657, 373937.079470594],
    [2077804.261528644, 380959.278000550],
]


def msk47_bytes(name="МСК 47 зона 2", crs=None):
    data = collection([polygon({}, MSK47_ZONE2_RING)], crs=crs)
    if name:
        data["name"] = name
    return as_bytes(data)


def first_vertex(layer):
    return layer.features[0]["geometry"]["coordinates"][0][0]


@pytest.mark.parametrize("name", ["МСК 47 зона 2", None])
def test_msk47_without_crs_is_placed_by_name_or_coordinates(name):
    layer = parse_upload("участок.geojson", msk47_bytes(name))[0]
    lon, lat = first_vertex(layer)
    assert lon == pytest.approx(27.915296, abs=1e-5)
    assert lat == pytest.approx(59.406646, abs=1e-5)
    assert "crs_missing_assumed:msk47_zone2" in layer.warnings


def test_msk47_zone_named_in_crs_is_reprojected_without_warning():
    layer = parse_upload("участок.geojson", msk47_bytes(None, crs="MSK47 zone 2"))[0]
    assert first_vertex(layer)[0] == pytest.approx(27.915296, abs=1e-5)
    assert layer.warnings == []


def test_msk47_zone_name_contradicting_coordinates_is_rejected():
    with pytest.raises(AgentsInputException, match="crs"):
        parse_upload("участок.geojson", msk47_bytes("МСК-47 зона 1"))


@pytest.mark.filterwarnings("ignore:.crs. was not provided")
def test_shapefile_in_msk47_without_prj(tmp_path: Path):
    gpd.GeoDataFrame(
        {"levels": [5]}, geometry=[Polygon(MSK47_ZONE2_RING)], crs=None
    ).to_file(tmp_path / "houses.shp")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in tmp_path.iterdir():
            if path.suffix != ".prj":
                archive.write(path, path.name)
    layer = parse_upload("houses.zip", buffer.getvalue())[0]
    assert first_vertex(layer)[1] == pytest.approx(59.406646, abs=1e-5)
    assert layer.warnings == ["crs_missing_assumed:msk47_zone2"]


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
async def uploads(tmp_path: Path):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield ComplianceUploadService(PipelineStateStore(redis), LayerCache(tmp_path))
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


async def test_redis_keeps_layer_summaries_and_the_cache_keeps_geometry(uploads):
    summary = await uploads.create("alice", files())
    stored = await uploads.state_store.get_compliance_upload(summary["upload_id"])
    assert "coordinates" not in json.dumps(stored)
    assert stored["layers"][0] == {
        "name": "map_objects",
        "role": "project",
        "role_source": "auto",
        "feature_count": 1,
        "geometry_types": ["Polygon"],
        "warnings": [],
        "file": f"{summary['upload_id']}/0.geojson",
    }
    assert summary["layers"][0] == {
        key: value for key, value in stored["layers"][0].items() if key != "file"
    }
    loaded = await uploads.load([summary["upload_id"]], "alice")
    assert loaded.layers[0].features[0]["geometry"]["coordinates"] == [SQUARE]


async def test_upload_whose_cached_layers_are_gone_reads_as_expired(uploads):
    summary = await uploads.create("alice", files())
    uploads.layer_cache.delete(summary["upload_id"])
    with pytest.raises(AgentsNotFound, match="срок"):
        await uploads.get(summary["upload_id"], "alice")
    assert await uploads.state_store.get_compliance_upload(summary["upload_id"]) is None


async def test_deleting_an_upload_removes_its_layers(uploads):
    summary = await uploads.create("alice", files())
    await uploads.delete(summary["upload_id"], "alice")
    assert not (uploads.layer_cache.root / summary["upload_id"]).exists()


async def test_failed_record_save_leaves_no_layer_files(uploads, monkeypatch):
    async def failing(*_):
        raise ConnectionError("redis down")

    monkeypatch.setattr(uploads.state_store, "save_compliance_upload", failing)
    with pytest.raises(ConnectionError):
        await uploads.create("alice", files())
    assert list(uploads.layer_cache.root.iterdir()) == []


async def test_record_with_inline_layers_from_before_the_cache_still_loads(uploads):
    record = {
        "upload_id": "legacy",
        "owner": "alice",
        "files": ["map_objects.geojson"],
        "layers": [
            {
                "name": "map_objects",
                "role": "project",
                "role_source": "auto",
                "warnings": [],
                "feature_collection": collection([polygon({"type": "buildings"})]),
            }
        ],
    }
    await uploads.state_store.save_compliance_upload("legacy", record)
    assert (
        uploads.summary(await uploads.get("legacy", "alice"))["layers"][0][
            "feature_count"
        ]
        == 1
    )
    loaded = await uploads.load(["legacy"], "alice")
    assert loaded.has_project


def test_layer_cache_expires_old_uploads_and_resolves_only_its_own_names(
    tmp_path: Path, monkeypatch
):
    cache = LayerCache(tmp_path, ttl_seconds=10)
    old = cache.root / "00000000-0000-0000-0000-000000000000"
    old.mkdir()
    (old / "0.geojson").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "src.agents.common.files.layer_cache.time.time",
        lambda: old.stat().st_mtime + 11,
    )
    assert cache.purge_expired() == 1
    assert not old.exists()
    for name in ["../secret.geojson", "00000000-0000-0000-0000-000000000000/../x"]:
        assert not cache.exists(name)
        with pytest.raises(LayerMissing):
            cache._path(name)


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
