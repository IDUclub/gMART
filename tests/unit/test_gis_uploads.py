"""Synthetic GIS exports exercise encodings, archive budgets and CRS recovery."""

import io
import json
import re
import struct
import zipfile

import geopandas as gpd
import pandas as pd
import pytest
from pyproj import Geod, Transformer
from shapely.geometry import Point

from src.agents.api_clients.pzz_api_client import PzzApiClient
from src.agents.common.exceptions.base_exceptions import AgentsInputException
from src.agents.common.files.vector_reader import (
    complete_datum,
    detect_dbf_encoding,
    read_vector,
)
from src.agents.services.compilance import compliance_uploads as uploads
from src.agents.services.compilance.crs_guess import (
    MSK47_ZONES,
    MSK76_ZONE1,
    guess_crs,
    zone_by_coordinates,
    zone_by_name,
)


def geojson(x=37.5, y=55.7):
    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {"type": "Point", "coordinates": [x, y]},
                }
            ],
        }
    ).encode()


class DosZipInfo(zipfile.ZipInfo):
    def _encodeFilenameFlags(self):
        return self.filename.encode("cp866"), self.flag_bits & ~0x800


def zipped(members, *, dos=False):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(DosZipInfo(name) if dos else name, data)
    return buffer.getvalue()


def dataset(tmp_path, suffix, *, crs="EPSG:4326", xy=(37.5, 55.7), encoding=None):
    path = tmp_path / ("source" + suffix)
    frame = gpd.GeoDataFrame({"title": ["Школа"]}, geometry=[Point(*xy)], crs=crs)
    drivers = {
        ".tab": "MapInfo File",
        ".mif": "MapInfo File",
        ".gml": "GML",
        ".shp": "ESRI Shapefile",
    }
    kwargs = {"encoding": encoding} if encoding else {}
    frame.to_file(path, driver=drivers[suffix], **kwargs)
    return {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


def test_dos_tab_layer_and_duplicate_basenames(tmp_path):
    members = dataset(tmp_path, ".tab")
    content = zipped(
        {
            f"{folder}/{name.replace('source', 'Школа')}": data
            for folder in ("План", "Другой")
            for name, data in members.items()
        },
        dos=True,
    )
    layers = uploads.parse_upload("test.zip", content)
    assert {layer.name for layer in layers} == {"План/Школа", "Другой/Школа"}
    for layer in layers:
        assert layer.features[0]["geometry"]["coordinates"] == pytest.approx(
            (37.5, 55.7), abs=1e-5
        )


def test_tab_missing_map(tmp_path):
    members = dataset(tmp_path, ".tab")
    del members["source.map"]
    with pytest.raises(AgentsInputException, match=r"source.*\.map"):
        uploads.parse_upload("test.zip", zipped(members))


@pytest.mark.parametrize(
    ("header", "extra", "warning"),
    [
        (b'Definition Table\n Type "RASTER"', {}, "raster_tab_skipped"),
        (b"Definition Table\n Type RASTER", {}, "raster_tab_skipped"),
        (
            b'Definition Table\n Type NATIVE\n"\\IsSeamless" = "TRUE"',
            {},
            "seamless_tab_skipped",
        ),
        (
            b"Definition Table\n Type NATIVE",
            {"skip.tda": b"fake"},
            "mapinfo_nativex_unsupported",
        ),
    ],
)
def test_nonvector_tabs_are_skipped(header, extra, warning):
    layers = uploads.parse_upload(
        "test.zip", zipped({"ok.geojson": geojson(), "skip.tab": header, **extra})
    )
    assert len(layers) == 1
    assert warning in layers[0].warnings


def test_mif_mid(tmp_path):
    layer = uploads.parse_upload("test.zip", zipped(dataset(tmp_path, ".mif")))[0]
    assert layer.features[0]["geometry"]["coordinates"] == pytest.approx(
        (37.5, 55.7), abs=1e-5
    )


def test_standalone_mif_without_mid(tmp_path):
    members = dataset(tmp_path, ".mif")
    # GDAL can read the geometry without MID; attributes are unavailable.
    layers = uploads.parse_upload("test.mif", members["source.mif"])
    assert layers[0].features[0]["geometry"]["coordinates"] == pytest.approx(
        (37.5, 55.7), abs=1e-5
    )
    layers = uploads.parse_upload(
        "test.zip", zipped({"source.mif": members["source.mif"]})
    )
    assert len(layers[0].features) == 1


def test_gml_reprojects_and_pzz_uses_the_same_reader(tmp_path):
    members = dataset(tmp_path, ".gml", crs="EPSG:3857", xy=(4174000, 7495000))
    content = members["source.gml"]
    layer = uploads.parse_upload("test.gml", content)[0]
    expected = Transformer.from_crs(3857, 4326, always_xy=True).transform(
        4174000, 7495000
    )
    assert layer.features[0]["geometry"]["coordinates"] == pytest.approx(expected)
    converted = json.loads(PzzApiClient._vector_to_geojson(content, ".gml"))
    assert converted["features"][0]["geometry"]["coordinates"] == pytest.approx(
        expected
    )


def test_gml_unknown_crs_rejected(tmp_path):
    members = dataset(tmp_path, ".gml", crs=None, xy=(349155, 6647806))
    with pytest.raises(AgentsInputException, match="параметр crs"):
        uploads.parse_upload("test.gml", members["source.gml"])


@pytest.mark.parametrize("encoding", ["UTF-8", "CP1251"])
def test_dbf_without_cpg(tmp_path, encoding):
    members = dataset(tmp_path, ".shp", encoding=encoding)
    members.pop("source.cpg", None)
    dbf = bytearray(members["source.dbf"])
    dbf[29] = 0
    members["source.dbf"] = bytes(dbf)
    layer = uploads.parse_upload("test.zip", zipped(members))[0]
    assert layer.features[0]["properties"]["title"] == "Школа"


def test_cpg_wins_over_ldid(tmp_path):
    members = dataset(tmp_path, ".shp", encoding="UTF-8")
    dbf = bytearray(members["source.dbf"])
    dbf[29] = 0xC9
    members["source.dbf"] = bytes(dbf)
    members["source.cpg"] = b"UTF-8"
    layer = uploads.parse_upload("test.zip", zipped(members))[0]
    assert layer.features[0]["properties"]["title"] == "Школа"


@pytest.mark.parametrize(
    ("ldid", "expected"),
    [(0xC9, "cp1251"), (0x57, "cp1251"), (0x26, "cp866"), (0x65, "cp866")],
)
def test_dbf_language_driver(ldid, expected):
    header = bytearray(32)
    header[29] = ldid
    assert detect_dbf_encoding(bytes(header)) == expected
    assert detect_dbf_encoding(bytes(header), "UTF-8") == "utf-8"


def test_nonearth_tab_requires_crs_and_accepts_registry(tmp_path):
    members = dataset(tmp_path, ".tab", crs=None, xy=(1255000, 385000))
    with pytest.raises(AgentsInputException, match="параметр crs"):
        uploads.parse_upload("test.zip", zipped(members))
    layer = uploads.parse_upload("test.zip", zipped(members), crs="msk76_zone1")[0]
    expected = Transformer.from_crs(MSK76_ZONE1.crs(), 4326, always_xy=True).transform(
        1255000, 385000
    )
    assert layer.features[0]["geometry"]["coordinates"] == pytest.approx(expected)
    assert "crs_from_request:msk76_zone1" in layer.warnings


def test_request_does_not_override_declared_crs(tmp_path):
    members = dataset(tmp_path, ".tab")
    layer = uploads.parse_upload("test.zip", zipped(members), crs="msk76_zone1")[0]
    assert layer.features[0]["geometry"]["coordinates"] == pytest.approx(
        (37.5, 55.7), abs=1e-5
    )
    assert not any(w.startswith("crs_from_request:") for w in layer.warnings)


# Generic ESRI CRS definition supplied with the task, independent of pilot data.
MSK76_ESRI = 'PROJCS["unknown",GEOGCS["GCS_unknown",DATUM["D_Unknown_based_on_Krassovsky_1942_ellipsoid_using_towgs84_23_57_-140_95_-79_8_0_0_35_0_79_-0_22",SPHEROID["Krassovsky_1942",6378245.0,298.3]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",1250000.0],PARAMETER["False_Northing",-6014743.504],PARAMETER["Central_Meridian",38.55],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]'


def test_esri_msk76_datum_completed(tmp_path):
    members = dataset(tmp_path, ".shp", crs=MSK76_ESRI, xy=(1255000, 385000))
    members["source.prj"] = MSK76_ESRI.encode()
    layer = uploads.parse_upload("test.zip", zipped(members))[0]
    assert "crs_datum_completed:msk76_zone1" in layer.warnings
    expected = Transformer.from_crs(MSK76_ZONE1.crs(), 4326, always_xy=True).transform(
        1255000, 385000
    )
    actual = layer.features[0]["geometry"]["coordinates"]
    assert Geod(ellps="WGS84").inv(*expected, *actual)[2] < 0.01
    assert zone_by_name("msk76_zone1") == MSK76_ZONE1


def test_registry_wkt_retains_datum_and_bound_crs_is_preserved():
    from pyproj import CRS

    source = CRS.from_wkt(MSK76_ZONE1.wkt())
    completed, warnings = complete_datum(source)
    assert completed == source
    assert warnings == []
    assert source.is_bound
    different = CRS.from_wkt(MSK76_ESRI.replace("38.55", "38.56"))
    assert complete_datum(different) == (different, [])


def test_reader_returns_wgs84_frame(tmp_path):
    dataset(tmp_path, ".gml", crs="EPSG:3857", xy=(4174000, 7495000))
    frame, warnings = read_vector(tmp_path / "source.gml")
    assert frame.crs.to_epsg() == 4326
    assert warnings == []


def test_geojson_engineering_crs_uses_request():
    data = json.loads(geojson(1255000, 385000))
    data["crs"] = {
        "type": "name",
        "properties": {"name": 'LOCAL_CS["Nonearth",UNIT["Meter",1]]'},
    }
    content = json.dumps(data).encode()
    with pytest.raises(AgentsInputException, match="параметр crs"):
        uploads.parse_upload("test.geojson", content)
    layer = uploads.parse_upload("test.geojson", content, crs="msk76_zone1")[0]
    assert "crs_from_request:msk76_zone1" in layer.warnings


def test_documents_do_not_spend_vector_budget(monkeypatch):
    members = {f"scan{i}.jpg": b"fake" for i in range(650)}
    members.update({"huge.pdf": b"x", "ok.geojson": geojson()})
    content = bytearray(zipped(members))
    # Forge only the central-directory declared size of a skipped PDF.
    offset = 0
    while (offset := content.find(b"PK\x01\x02", offset)) >= 0:
        name_size = struct.unpack_from("<H", content, offset + 28)[0]
        if content[offset + 46 : offset + 46 + name_size] == b"huge.pdf":
            struct.pack_into("<I", content, offset + 24, 1_000_000_000)
            break
        offset += 4
    monkeypatch.setattr(uploads, "MAX_ARCHIVE_MEMBERS", 2)
    monkeypatch.setattr(uploads, "MAX_UNPACKED_BYTES", 1000)
    layer = uploads.parse_upload("test.zip", bytes(content))[0]
    assert "skipped_members:jpg=650,pdf=1" in layer.warnings


@pytest.mark.parametrize(
    ("constant", "limit"),
    [("MAX_ARCHIVE_MEMBERS", 1), ("MAX_UNPACKED_BYTES", 10), ("MAX_LAYERS", 1)],
)
def test_vector_limits_enforced(monkeypatch, constant, limit):
    monkeypatch.setattr(uploads, constant, limit)
    with pytest.raises(uploads.UploadTooLarge):
        uploads.parse_upload(
            "test.zip", zipped({"a.geojson": geojson(), "b.geojson": geojson()})
        )


def test_nested_zip_one_level_and_shared_budget(monkeypatch):
    nested = zipped(
        {
            "inside.geojson": geojson(),
            "deeper.zip": zipped({"ignored.geojson": geojson()}),
        }
    )
    outer = zipped({"nested.zip": nested})
    layer = uploads.parse_upload("test.zip", outer)[0]
    assert layer.name == "nested/inside"
    assert "nested_archive_skipped" in layer.warnings
    monkeypatch.setattr(uploads, "MAX_ARCHIVE_MEMBERS", 2)
    with pytest.raises(uploads.UploadTooLarge):
        uploads.parse_upload("test.zip", outer)


@pytest.mark.parametrize("constant", ["MAX_UNPACKED_BYTES", "MAX_LAYERS"])
def test_nested_archives_share_size_and_layer_limits(monkeypatch, constant):
    inside = zipped({"a.geojson": geojson()})
    outside = zipped({"inside.zip": inside, "b.geojson": geojson()})
    limit = len(inside) + len(geojson()) if constant == "MAX_UNPACKED_BYTES" else 1
    monkeypatch.setattr(uploads, constant, limit)
    with pytest.raises(uploads.UploadTooLarge):
        uploads.parse_upload("test.zip", outside)


@pytest.mark.parametrize("suffix", [".rar", ".tab"])
def test_repack_messages(suffix):
    with pytest.raises(AgentsInputException, match="ZIP"):
        uploads.parse_upload("test" + suffix, b"fake")


def test_7z_shapefile(tmp_path):
    import py7zr

    members = dataset(tmp_path, ".shp", encoding="UTF-8")
    buffer = io.BytesIO()
    with py7zr.SevenZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(data, "Папка/" + name)
        archive.writestr(b"fake", "scan.pdf")
    layer = uploads.parse_upload("test.7z", buffer.getvalue())[0]
    assert layer.features[0]["properties"]["title"] == "Школа"
    assert "skipped_members:pdf=1" in layer.warnings
    # Exercise both archive formats as nested members using the same budget.
    nested = uploads.parse_upload("outer.zip", zipped({"inner.7z": buffer.getvalue()}))
    assert nested[0].features[0]["properties"]["title"] == "Школа"
    outer = io.BytesIO()
    with py7zr.SevenZipFile(outer, "w") as archive:
        archive.writestr(zipped({"inner.geojson": geojson()}), "inner.zip")
    assert uploads.parse_upload("outer.7z", outer.getvalue())[0].name == "inner/inner"


def test_kml_standalone():
    content = b'<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><Point><coordinates>37.5,55.7,0</coordinates></Point></Placemark></Document></kml>'
    layer = uploads.parse_upload("test.kml", content)[0]
    assert layer.features[0]["geometry"]["coordinates"] == pytest.approx((37.5, 55.7))


@pytest.mark.parametrize(
    "value", ["EPSG:3857", "+proj=merc +datum=WGS84 +units=m +type=crs"]
)
def test_explicit_custom_crs(value):
    layer = uploads.parse_upload("test.geojson", geojson(4174000, 7495000), crs=value)[
        0
    ]
    assert "crs_from_request:custom" in layer.warnings


def test_invalid_request_crs_rejected_even_with_declared_crs(tmp_path):
    members = dataset(tmp_path, ".gml")
    with pytest.raises(AgentsInputException, match="параметре crs"):
        uploads.parse_upload("test.gml", members["source.gml"], crs="invalid")


def test_request_crs_preserves_lonlat_layers_in_mixed_upload(tmp_path):
    members = dataset(tmp_path, ".shp", crs=None)
    members.pop("source.prj", None)
    members["wgs84.geojson"] = geojson()
    members["projected.geojson"] = geojson(1255000, 385000)
    layers = uploads.parse_upload("mixed.zip", zipped(members), crs="msk76_zone1")
    by_name = {layer.name: layer for layer in layers}
    for name in ("source", "wgs84"):
        layer = by_name[name]
        assert layer.features[0]["geometry"]["coordinates"] == pytest.approx(
            (37.5, 55.7)
        )
        assert not any(w.startswith("crs_from_request:") for w in layer.warnings)
    assert "crs_missing_assumed_wgs84" in by_name["source"].warnings
    assert "crs_from_request:msk76_zone1" in by_name["projected"].warnings


def test_nonearth_and_missing_projected_crs_in_one_upload(tmp_path):
    members = dataset(tmp_path, ".tab", crs=None, xy=(1255000, 385000))
    shape_path = tmp_path / "shapes"
    shape_path.mkdir()
    shapes = dataset(shape_path, ".shp", crs=None, xy=(1255000, 385000))
    shapes.pop("source.prj", None)
    members.update({f"shapes/{name}": data for name, data in shapes.items()})
    layers = uploads.parse_upload("mixed.zip", zipped(members), crs="msk76_zone1")
    expected = Transformer.from_crs(MSK76_ZONE1.crs(), 4326, always_xy=True).transform(
        1255000, 385000
    )
    assert len(layers) == 2
    for layer in layers:
        assert layer.features[0]["geometry"]["coordinates"] == pytest.approx(expected)
        assert "crs_from_request:msk76_zone1" in layer.warnings


def test_nonearth_lonlat_coordinates_require_request(tmp_path):
    members = dataset(tmp_path, ".tab", crs=None)
    with pytest.raises(AgentsInputException, match="параметр crs"):
        uploads.parse_upload("local.zip", zipped(members))
    layer = uploads.parse_upload("local.zip", zipped(members), crs="msk76_zone1")[0]
    assert "crs_from_request:msk76_zone1" in layer.warnings


def test_engineering_geojson_requires_request():
    data = json.loads(geojson())
    data["crs"] = {
        "type": "name",
        "properties": {"name": 'LOCAL_CS["Nonearth",UNIT["metre",1]]'},
    }
    with pytest.raises(AgentsInputException, match="параметр crs"):
        uploads.parse_upload("local.geojson", json.dumps(data).encode())
    layer = uploads.parse_upload(
        "local.geojson", json.dumps(data).encode(), crs="msk76_zone1"
    )[0]
    assert "crs_from_request:msk76_zone1" in layer.warnings


@pytest.mark.parametrize(
    ("cpg", "expected"),
    [(value, "utf-8") for value in ("UTF-8", "UTF8", "65001")]
    + [
        (value, "cp1251")
        for value in ("1251", "CP1251", "ANSI 1251", "WINDOWS-1251", "WIN1251")
    ]
    + [(value, "cp866") for value in ("866", "CP866", "OEM 866", "IBM866")]
    + [("latin1", "iso8859-1")],
)
def test_cpg_codec_aliases(cpg, expected):
    assert detect_dbf_encoding(bytes(32), cpg) == expected


@pytest.mark.parametrize(("encoding", "ldid"), [("UTF-8", 0x57), ("CP1251", 0)])
@pytest.mark.parametrize("cpg", [None, b"not-a-codec"])
def test_dbf_records_override_ldid_and_unknown_cpg_falls_back(
    tmp_path, encoding, ldid, cpg
):
    members = dataset(tmp_path, ".shp", encoding=encoding)
    members.pop("source.cpg", None)
    if cpg is not None:
        members["source.cpg"] = cpg
    dbf = bytearray(members["source.dbf"])
    dbf[29] = ldid
    members["source.dbf"] = bytes(dbf)
    layer = uploads.parse_upload("encoding.zip", zipped(members))[0]
    assert layer.features[0]["properties"]["title"] == "Школа"


def test_msk76_siblings_resolve_missing_shapefile_and_gml_crs(tmp_path):
    xy = Transformer.from_crs(4326, MSK76_ZONE1.crs(), always_xy=True).transform(
        39.9, 57.6
    )
    members = {}
    for name in ("declared_a", "declared_b", "missing"):
        folder = tmp_path / name
        folder.mkdir()
        files = dataset(folder, ".shp", crs=MSK76_ESRI, xy=xy)
        if name == "missing":
            files.pop("source.prj", None)
        else:
            files["source.prj"] = MSK76_ESRI.encode()
        members.update(
            {name + p[len("source") :]: data for p, data in files.items()}
        )
    gml_folder = tmp_path / "gml"
    gml_folder.mkdir()
    gml = dataset(gml_folder, ".gml", crs=None, xy=xy)["source.gml"]
    # Missing layers precede their declarations, including across nesting.
    content = zipped(
        {
            "missing.gml": gml,
            "nested.zip": zipped(
                {p: data for p, data in members.items() if p.startswith("missing")}
            ),
            **{p: data for p, data in members.items() if not p.startswith("missing")},
        }
    )
    layers = uploads.parse_upload("mixed.zip", content)
    assert len(layers) == 4
    for layer in layers:
        assert layer.features[0]["geometry"]["coordinates"] == pytest.approx(
            (39.9, 57.6), abs=1e-6
        )
        if "missing" in layer.name:
            assert "crs_from_sibling:msk76_zone1" in layer.warnings
        else:
            assert "crs_datum_completed:msk76_zone1" in layer.warnings


@pytest.mark.parametrize("suffix", [".shp", ".gml"])
def test_lone_missing_crs_in_overlapping_local_zones_is_rejected(tmp_path, suffix):
    members = dataset(tmp_path, suffix, crs=None, xy=(1255000, 385000))
    members.pop("source.prj", None)
    with pytest.raises(AgentsInputException, match="параметр crs"):
        uploads.parse_upload("ambiguous.zip", zipped(members))


def test_unique_msk47_zone_and_named_ambiguous_zone_still_work():
    ambiguous = (1255000, 385000, 1255000, 385000)
    assert zone_by_coordinates(ambiguous) is None
    assert guess_crs(("unknown",), ambiguous) is None
    assert guess_crs(("МСК 47 зона 1",), ambiguous) == MSK47_ZONES[1]
    layer = uploads.parse_upload("unique.geojson", geojson(2255000, 385000))[0]
    assert "crs_missing_assumed:msk47_zone2" in layer.warnings


@pytest.mark.parametrize(
    ("suffix", "missing"),
    [
        (".tab", ".map"),
        (".tab", ".dat"),
        (".tab", ".id"),
        (".shp", ".shx"),
        (".shp", ".dbf"),
    ],
)
def test_incomplete_layer_is_skipped_only_with_usable_siblings(
    tmp_path, suffix, missing
):
    members = dataset(tmp_path, suffix)
    del members["source" + missing]
    with pytest.raises(AgentsInputException, match=r"source.*" + re.escape(missing)):
        uploads.parse_upload("incomplete.zip", zipped(members))
    layers = uploads.parse_upload(
        "mixed.zip", zipped({**members, "ok.geojson": geojson()})
    )
    assert len(layers) == 1
    assert layers[0].name == "ok"
    assert f"incomplete_layer_skipped:source:{missing}" in layers[0].warnings


@pytest.mark.parametrize("crs", [None, "msk76_zone1"])
def test_attribute_only_dbf_tab_does_not_break_geometry_tab(tmp_path, crs):
    shape_folder = tmp_path / "shape"
    shape_folder.mkdir()
    dbf = dataset(shape_folder, ".shp")["source.dbf"]
    tab_folder = tmp_path / "tab"
    tab_folder.mkdir()
    members = dataset(tab_folder, ".tab")
    members.update(
        {
            "attributes.tab": (
                "!table\n!version 300\n!charset WindowsCyrillic\n"
                'Definition Table\n File "attributes.dbf"\n'
                ' Type DBF Charset "WindowsCyrillic"\n'
                " Fields 1\n title Char (80) ;\n"
            ).encode(),
            "attributes.dbf": dbf,
        }
    )
    layers = uploads.parse_upload("tables.zip", zipped(members), crs=crs)
    assert len(layers) == 1
    assert layers[0].features[0]["geometry"]["coordinates"] == pytest.approx(
        (37.5, 55.7), abs=1e-5
    )
    assert "no_geometry_layer_skipped:attributes" in layers[0].warnings


@pytest.mark.parametrize(
    "frame", [pd.DataFrame({"a": [1]}), gpd.GeoDataFrame({"a": [1]})]
)
@pytest.mark.parametrize("crs", [None, "msk76_zone1"])
def test_reader_skips_tables_before_accessing_crs(monkeypatch, tmp_path, frame, crs):
    monkeypatch.setattr(gpd, "read_file", lambda *args, **kwargs: frame)
    result, warnings = read_vector(
        tmp_path / "attributes.tab", explicit_crs=crs, names=("attributes",)
    )
    assert result.empty
    assert warnings == ["no_geometry_layer_skipped:attributes"]


def test_driver_errors_are_input_errors(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise AttributeError("synthetic driver failure")

    monkeypatch.setattr(gpd, "read_file", fail)
    with pytest.raises(AgentsInputException, match="Не удалось прочитать слой"):
        read_vector(tmp_path / "broken.tab", explicit_crs="msk76_zone1")


def declared_geojson(x, y, crs):
    data = json.loads(geojson(x, y))
    data["crs"] = {"type": "name", "properties": {"name": crs}}
    return json.dumps(data).encode()


def test_sibling_crs_must_be_unique_and_spatially_close():
    declared = declared_geojson(1255000, 385000, "msk76_zone1")
    for others in (
        {"far.geojson": geojson(1355000, 385000)},
        {
            "other.geojson": declared_geojson(2255000, 385000, "msk47_zone2"),
            "missing.geojson": geojson(1255000, 385000),
        },
    ):
        with pytest.raises(AgentsInputException, match="параметр crs"):
            uploads.parse_upload(
                "mixed.zip", zipped({"declared.geojson": declared, **others})
            )


def test_request_wins_over_sibling_and_lonlat_is_preserved():
    members = {
        "declared.geojson": declared_geojson(1255000, 385000, "msk76_zone1"),
        "missing.geojson": geojson(1255000, 385000),
        "lonlat.geojson": geojson(),
    }
    layers = uploads.parse_upload("mixed.zip", zipped(members), crs="msk47_zone1")
    by_name = {layer.name: layer for layer in layers}
    assert "crs_from_request:msk47_zone1" in by_name["missing"].warnings
    assert by_name["lonlat"].features[0]["geometry"]["coordinates"] == pytest.approx(
        (37.5, 55.7)
    )


def test_sibling_crs_does_not_escape_one_parse_upload_call():
    uploads.parse_upload(
        "declared.geojson", declared_geojson(1255000, 385000, "msk76_zone1")
    )
    with pytest.raises(AgentsInputException, match="параметр crs"):
        uploads.parse_upload("missing.geojson", geojson(1255000, 385000))


@pytest.mark.parametrize("suffix", [".gpkg", ".gml"])
def test_multilayer_container_skips_attribute_tables(monkeypatch, tmp_path, suffix):
    import pyogrio

    monkeypatch.setattr(
        pyogrio, "list_layers", lambda path: [("attributes", None), ("points", "Point")]
    )

    def read(path, *, layer=None, **kwargs):
        if layer == "attributes":
            return pd.DataFrame({"title": ["table"]})
        return gpd.GeoDataFrame(geometry=[Point(37.5, 55.7)], crs=4326)

    monkeypatch.setattr(gpd, "read_file", read)
    layers = uploads.parse_upload("mixed" + suffix, b"synthetic", crs="msk76_zone1")
    assert len(layers) == 1
    assert "no_geometry_layer_skipped:mixed:attributes" in layers[0].warnings


def test_custom_sibling_uses_union_and_fifty_kilometre_margin():
    members = {
        "left.geojson": declared_geojson(500000, 6200000, "EPSG:32637"),
        "right.geojson": declared_geojson(600000, 6200000, "EPSG:32637"),
        "inside.geojson": geojson(550000, 6200000),
        "near.geojson": geojson(649000, 6200000),
    }
    layers = uploads.parse_upload("custom.zip", zipped(members))
    for layer in layers:
        if layer.name in {"inside", "near"}:
            assert "crs_from_sibling:custom" in layer.warnings


def test_engineering_layer_cannot_inherit_sibling_crs(tmp_path):
    members = dataset(tmp_path, ".tab", crs=None, xy=(1255000, 385000))
    members["declared.geojson"] = declared_geojson(1255000, 385000, "msk76_zone1")
    with pytest.raises(AgentsInputException, match="параметр crs"):
        uploads.parse_upload("engineering.zip", zipped(members))


def test_dbf_only_layer_is_skipped(tmp_path):
    dbf = dataset(tmp_path, ".shp")["source.dbf"]
    layers = uploads.parse_upload(
        "mixed.zip", zipped({"attributes.dbf": dbf, "ok.geojson": geojson()})
    )
    assert len(layers) == 1
    assert "no_geometry_layer_skipped:attributes" in layers[0].warnings


def test_incomplete_nested_archive_does_not_reject_outer_usable_layer():
    layers = uploads.parse_upload(
        "mixed.zip",
        zipped(
            {
                "nested.zip": zipped({"broken.shp": b"synthetic"}),
                "ok.geojson": geojson(),
            }
        ),
    )
    assert len(layers) == 1
    assert "incomplete_layer_skipped:broken:.shx,.dbf" in layers[0].warnings


@pytest.mark.parametrize("crs", [None, "msk76_zone1"])
def test_native_attribute_tab_without_map_and_id(monkeypatch, tmp_path, crs):
    members = dataset(tmp_path, ".tab")
    members.update(
        {
            "attributes.tab": b"Definition Table\n Type NATIVE\n",
            "attributes.dat": b"synthetic attribute data",
        }
    )
    original_read = gpd.read_file

    def read(path, **kwargs):
        if path.stem == "attributes":
            return pd.DataFrame({"title": ["table"]})
        return original_read(path, **kwargs)

    monkeypatch.setattr(gpd, "read_file", read)
    layers = uploads.parse_upload("tables.zip", zipped(members), crs=crs)
    assert len(layers) == 1
    assert "no_geometry_layer_skipped:attributes" in layers[0].warnings
