"""Read vector files consistently, including exports with incomplete CRS metadata."""

from __future__ import annotations

import codecs
import math
import re
from pathlib import Path

import geopandas as gpd
from pyproj import CRS
from pyproj.exceptions import CRSError

from src.agents.common.exceptions.base_exceptions import AgentsInputException
from src.agents.services.compilance.crs_guess import (
    CRS_REGISTRY,
    LOCAL_ZONES,
    guess_crs,
)

UNKNOWN_CRS = (
    "«{name}»: в файле не указана система координат, а координаты не похожи ни на "
    "широту и долготу (EPSG:4326), ни на МСК-47, ни на EPSG:3857. Загрузите файл "
    "заново с указанной системой координат — GeoJSON с полем crs, Shapefile в ZIP "
    "вместе с файлом .prj или GeoPackage — либо пересохраните его в EPSG:4326 "
    "или укажите параметр crs при загрузке"
)
WRONG_CRS = (
    "«{name}»: координаты не соответствуют указанной в файле системе координат. "
    "Проверьте её и загрузите файл заново или пересохраните его в EPSG:4326"
)


def detect_dbf_encoding(dbf_bytes: bytes, cpg: str | None = None) -> str:
    """Avoid GDAL's Latin-1 fallback for Russian DBFs with missing metadata.

    0x26/0x65 are OEM Russian (866), 0xC9 is Windows Russian (1251).
    0x57 normally denotes ANSI/Latin-1 in GDAL; the upload convention deliberately
    treats it as 1251 because these exports carry Cyrillic under that marker.
    """
    if cpg and cpg.strip():
        encoding = cpg.lstrip("\ufeff").strip()
        compact = re.sub(r"[\s_-]+", "", encoding).casefold()
        aliases = {
            "65001": "utf-8",
            "utf8": "utf-8",
            "1251": "cp1251",
            "ansi1251": "cp1251",
            "win1251": "cp1251",
            "windows1251": "cp1251",
            "866": "cp866",
            "oem866": "cp866",
            "ibm866": "cp866",
        }
        try:
            return codecs.lookup(aliases.get(compact, encoding)).name
        except LookupError:
            pass
    header_size = int.from_bytes(dbf_bytes[8:10], "little")
    records = dbf_bytes[header_size:]
    if any(byte >= 128 for byte in records):
        try:
            records.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            pass
        else:
            return "utf-8"
    ldid = dbf_bytes[29] if len(dbf_bytes) > 29 else 0
    return "cp866" if ldid in (0x26, 0x65) else "cp1251"


def resolve_crs(value: str) -> tuple[CRS, str]:
    """Validate the request even when every layer already declares a CRS."""
    zone = CRS_REGISTRY.get(value.strip().casefold())
    if zone is not None:
        return zone.crs(), zone.key
    try:
        source = CRS.from_user_input(value)
        if source.is_engineering:
            raise CRSError("engineering CRS")
        return source, "custom"
    except (CRSError, ValueError) as exc:
        raise AgentsInputException(
            "Некорректная система координат в параметре crs"
        ) from exc


def complete_datum(source: CRS) -> tuple[CRS, list[str]]:
    """Recover a known datum transform hidden inside an ESRI datum name.

    Only exact projection and ellipsoid matches qualify; a real bound CRS keeps
    its declared transformation even if its projection matches the registry.
    """
    operation = source.coordinate_operation
    if source.is_bound or not source.is_projected or operation is None:
        return source, []
    if operation.method_name != "Transverse Mercator":
        return source, []
    ellipsoid = source.ellipsoid
    if not (
        math.isclose(ellipsoid.semi_major_metre, 6378245, abs_tol=1e-6, rel_tol=0)
        and math.isclose(ellipsoid.inverse_flattening, 298.3, abs_tol=1e-8, rel_tol=0)
    ):
        return source, []
    params = {str(param.code): param.value for param in operation.params}
    for zone in LOCAL_ZONES.values():
        expected = {
            "8801": zone.latitude_of_origin,
            "8802": zone.central_meridian,
            "8805": zone.scale,
            "8806": zone.false_easting,
            "8807": zone.false_northing,
        }
        if zone.towgs84 and all(
            key in params and math.isclose(params[key], value, abs_tol=1e-8, rel_tol=0)
            for key, value in expected.items()
        ):
            return zone.crs(), [f"crs_datum_completed:{zone.key}"]
    return source, []


def within_wgs84(bounds) -> bool:
    min_x, min_y, max_x, max_y = bounds
    return -180 <= min_x <= max_x <= 180 and -90 <= min_y <= max_y <= 90


def read_vector(
    path: Path,
    *,
    layer: str | None = None,
    explicit_crs: str | None = None,
    names: tuple[str | None, ...] = (),
    require_crs: bool = False,
    native: bool = False,
) -> tuple[gpd.GeoDataFrame, list[str]]:
    """Return WGS84 data while preserving declared CRSs and reporting assumptions."""
    requested = resolve_crs(explicit_crs) if explicit_crs is not None else None
    name = next((name for name in names if name), path.stem)
    try:
        options = {}
        if path.suffix.casefold() == ".shp":
            dbf = path.with_suffix(".dbf")
            cpg = path.with_suffix(".cpg")
            if dbf.exists():
                options["encoding"] = detect_dbf_encoding(
                    dbf.read_bytes(),
                    cpg.read_text(encoding="utf-8-sig") if cpg.exists() else None,
                )
        frame = (
            gpd.read_parquet(path)
            if path.suffix.casefold() in {".parquet", ".geoparquet"}
            else gpd.read_file(path, layer=layer, **options)
        )
    except Exception as exc:
        raise AgentsInputException(
            f"Не удалось прочитать слой «{name}»: {exc}"
        ) from exc
    if not has_geometry(frame):
        return gpd.GeoDataFrame(geometry=[]), [f"no_geometry_layer_skipped:{name}"]
    if native:
        return frame, []
    return reproject_vector(
        frame, names=names or (name,), requested=requested, require_crs=require_crs
    )


def has_geometry(frame) -> bool:
    """Drivers can return ordinary tables or frames without active geometry."""
    return (
        isinstance(frame, gpd.GeoDataFrame) and frame.active_geometry_name is not None
    )


def reproject_vector(
    frame, *, names=(), requested=None, require_crs=False, sibling=None
):
    """Resolve native data after all declared sibling CRSs have been collected."""
    name = next((name for name in names if name), "layer")
    try:
        return _reproject_vector(frame, name, names, requested, require_crs, sibling)
    except AgentsInputException:
        raise
    except Exception as exc:
        raise AgentsInputException(
            f"Не удалось прочитать слой «{name}»: {exc}"
        ) from exc


def _reproject_vector(frame, name, names, requested, require_crs, sibling):
    warnings = []
    source = frame.crs
    engineering = source is not None and (
        source.is_engineering or 'LOCAL_CS["Nonearth' in source.to_wkt()
    )
    if engineering:
        source = None
    declared = source is not None
    if frame.empty or not (frame.geometry.notna() & ~frame.geometry.is_empty).any():
        return frame, warnings
    if require_crs and source is None and requested is None:
        raise AgentsInputException("Uploaded vector layer must declare its CRS")
    if source is None:
        bounds = tuple(float(value) for value in frame.total_bounds)
        if requested is not None and (engineering or not within_wgs84(bounds)):
            source, key = requested
            warnings.append(f"crs_from_request:{key}")
        elif not engineering and not within_wgs84(bounds) and sibling is not None:
            source, key = sibling
            warnings.append(f"crs_from_sibling:{key}")
        else:
            # Nonearth describes an arbitrary local plane. Its coordinates can
            # accidentally fit an MSK-47 range, so require a request for it.
            guessed = (
                None
                if engineering or within_wgs84(bounds)
                else guess_crs(names, bounds)
            )
            if guessed is not None:
                source = guessed.crs()
                warnings.append(f"crs_missing_assumed:{guessed.key}")
            elif not engineering and within_wgs84(bounds):
                source = CRS.from_epsg(4326)
                warnings.append("crs_missing_assumed_wgs84")
            else:
                raise AgentsInputException(UNKNOWN_CRS.format(name=name))
    else:
        source, warnings = complete_datum(source)
    try:
        frame = frame.set_crs(source, allow_override=True).to_crs(4326)
    except Exception as exc:
        raise AgentsInputException(WRONG_CRS.format(name=name)) from exc
    if not within_wgs84(frame.total_bounds):
        raise AgentsInputException(
            (WRONG_CRS if declared or requested else UNKNOWN_CRS).format(name=name)
        )
    return frame, warnings
