"""User files for a compliance check: parsing, layer roles and storage.

A user checks their own design against the norms by uploading vector layers,
including MapInfo, GML and KML, directly or in ZIP/7z archives. Each dataset
(each GeoPackage layer) becomes one layer in WGS84 with 2D geometry, and
gets a role:

- ``project`` — the designed objects; only they are checked;
- ``context`` — the surroundings (OSM, the existing state); they are sources and
  neighbours of distance rules, never checked themselves;
- ``border`` — the boundary of the design; it tells where the design is.

A layer without a CRS is WGS84 unless its coordinates fall outside lon/lat. Then
an explicit request or a unique nearby declared sibling CRS takes precedence over
name/coordinate guesses (see ``crs_guess``); ambiguity requires the user's CRS.

The role is guessed from the layer name and content and can be set explicitly.
An upload lives for a day under an ``upload_id`` owned by the user who uploaded it:
its record (roles, mapping answers, a summary of every layer) in Redis, the parsed
layers themselves as files in the application's ``LayerCache``.
"""

from __future__ import annotations

import io
import json
import math
import re
import tempfile
import uuid
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from shapely.geometry import mapping, shape
from shapely.ops import transform

from src.agents.common.config.app_config import ComplianceUploadLimits
from src.agents.common.exceptions.base_exceptions import (
    AgentsInputException,
    AgentsNotFound,
)
from src.agents.common.files.gis_classifier import class_id
from src.agents.common.files.layer_cache import LayerCache, LayerMissing
from src.agents.common.files.vector_reader import (
    UNKNOWN_CRS,
    complete_datum,
    read_vector,
    reproject_vector,
    resolve_crs,
)
from src.agents.services.compilance.crs_guess import (
    CRS_REGISTRY,
    guess_crs,
    zone_by_name,
)

if TYPE_CHECKING:
    from src.agents.services.pipeline_state import PipelineStateStore

LayerRole = Literal["project", "context", "border"]
LAYER_ROLES: tuple[str, ...] = ("project", "context", "border")
# Whose mapping answers an upload keeps: «не проверять» in a compliance check is not
# «не создавать» in a project created from the same files.
MappingNamespace = Literal["compliance", "project"]
_MAPPING_FIELDS: dict[str, tuple[str, str]] = {
    "compliance": ("confirmed_mapping", "classified"),
    "project": ("project_confirmed_mapping", "project_classified"),
}

_LIMITS = ComplianceUploadLimits.from_env()
MAX_UPLOAD_BYTES = _LIMITS.max_upload_mb * 1024 * 1024
# Archives may hold far more than they weigh; only vector members spend limits.
MAX_UNPACKED_BYTES = _LIMITS.max_unpacked_mb * 1024 * 1024
MAX_ARCHIVE_MEMBERS = _LIMITS.max_archive_members
MAX_LAYERS = _LIMITS.max_layers
MAX_FEATURES = 200_000
UPLOAD_TTL_SECONDS = 24 * 60 * 60

_GEOJSON_SUFFIXES = {".geojson", ".json"}
_SHAPEFILE_SIDECARS = (".shx", ".dbf")
_WGS84_NAMES = {
    "urn:ogc:def:crs:ogc:1.3:crs84",
    "urn:ogc:def:crs:ogc::crs84",
    "urn:ogc:def:crs:epsg::4326",
    "epsg:4326",
    "crs84",
    "ogc:crs84",
}
_BORDER_HINTS = ("border", "boundary", "границ")
_CONTEXT_HINTS = (
    "surrounding",
    "osm",
    "context",
    "existing",
    "контекст",
    "окружен",
    "сущ",
)


class UploadTooLarge(AgentsInputException):
    status_code = 413


@dataclass
class UploadedLayer:
    """One layer of an upload, ready to be mapped to norm entities."""

    name: str
    role: LayerRole
    feature_collection: dict[str, Any]
    role_source: Literal["auto", "user"] = "auto"
    warnings: list[str] = field(default_factory=list)

    @property
    def features(self) -> list[dict[str, Any]]:
        return self.feature_collection.get("features") or []

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "role_source": self.role_source,
            "feature_count": len(self.features),
            "geometry_types": sorted(
                {
                    feature["geometry"]["type"]
                    for feature in self.features
                    if feature.get("geometry")
                }
            ),
            "warnings": list(self.warnings),
        }

    def to_record(self, file: str) -> dict[str, Any]:
        """What the upload record keeps of the layer: its summary and its file."""
        return {**self.summary(), "file": file}

    @classmethod
    def from_record(
        cls, item: dict[str, Any], feature_collection: dict[str, Any]
    ) -> "UploadedLayer":
        return cls(
            name=item["name"],
            role=item["role"],
            role_source=item.get("role_source") or "auto",
            warnings=list(item.get("warnings") or []),
            feature_collection=feature_collection,
        )


def _record_summary(item: dict[str, Any]) -> dict[str, Any]:
    if "file" not in item:  # a record written before layers moved to the cache
        return UploadedLayer.from_record(item, item["feature_collection"]).summary()
    return {key: value for key, value in item.items() if key != "file"}


def detect_role(name: str, features: list[dict[str, Any]]) -> LayerRole:
    """Guess a layer role from its name and the ``type`` of its features."""
    types = {
        str((feature.get("properties") or {}).get("type") or "").casefold()
        for feature in features
    }
    lowered = name.casefold()
    codes = [class_id(feature.get("properties") or {}) for feature in features]
    basename = PurePosixPath(name.replace("\\", "/")).name.casefold()
    if basename.startswith("gr_dpt") or (
        codes and all(code.startswith("6010") for code in codes)
    ):
        return "border"
    if (types == {"border"}) or any(hint in lowered for hint in _BORDER_HINTS):
        return "border"
    if codes and all(
        code.startswith(("701", "702", "603", "604", "606", "609", "704", "706"))
        for code in codes
    ):
        return "context"
    if any(hint in lowered for hint in _CONTEXT_HINTS):
        return "context"
    return "project"


def parse_upload(
    filename: str, content: bytes, *, crs: str | None = None
) -> list[UploadedLayer]:
    """Every layer of one uploaded file; raises ``AgentsInputException`` on bad input."""
    try:
        return _parse_upload(filename, content, crs=crs)
    except AgentsInputException:
        raise
    except Exception as exc:
        raise AgentsInputException(
            f"Не удалось прочитать файл «{filename}»: {exc}"
        ) from exc


def _parse_upload(
    filename: str, content: bytes, *, crs: str | None = None
) -> list[UploadedLayer]:
    if crs is not None:
        resolve_crs(crs)
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadTooLarge(
            f"Файл «{filename}» больше {MAX_UPLOAD_BYTES // (1024 * 1024)} МиБ"
        )
    name = PurePosixPath(filename.replace("\\", "/")).name or "layer.geojson"
    suffix = PurePosixPath(name).suffix.casefold()
    if suffix in _GEOJSON_SUFFIXES:
        layers = [_geojson_layer(_stem(name), content, crs=crs)]
    elif suffix in {".zip", ".7z"}:
        layers = _archive_layers(name, content, crs=crs)
    elif suffix == ".gpkg":
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.gpkg"
            path.write_bytes(content)
            layers = _gpkg_layers(_stem(name), path, crs=crs)
    elif suffix in {".gml", ".kml", ".mif"}:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ("upload" + suffix)
            path.write_bytes(content)
            try:
                layers = _vector_layers(_stem(name), path, crs=crs)
            except AgentsInputException as exc:
                if suffix == ".mif":
                    raise AgentsInputException(
                        f"MIF «{name}» загрузите ZIP-архивом вместе с файлом .mid: {exc}"
                    ) from exc
                raise
    elif suffix == ".tab":
        raise AgentsInputException(
            f"MapInfo TAB «{name}» загрузите ZIP-архивом вместе с файлами .dat, .map, .id"
        )
    elif suffix == ".rar":
        raise AgentsInputException(
            "RAR не поддерживается. Перепакуйте архив в ZIP или 7z"
        )
    elif suffix == ".shp":
        raise AgentsInputException(
            f"Shapefile «{name}» загрузите ZIP-архивом вместе с файлами .shx, .dbf "
            "и .prj"
        )
    else:
        raise AgentsInputException(
            f"Формат файла «{name}» не поддерживается. Загрузите GeoJSON, ZIP "
            "или 7z (GeoJSON, Shapefile, TAB, MIF/MID, GML, KML, GeoPackage), GML, KML или MIF"
        )
    if len(layers) > MAX_LAYERS:
        raise UploadTooLarge(f"В файле «{name}» больше {MAX_LAYERS} слоёв")
    layers = [layer for layer in layers if layer.features]
    if not layers:
        raise AgentsInputException(f"В файле «{name}» нет объектов с геометрией")
    if sum(len(layer.features) for layer in layers) > MAX_FEATURES:
        raise UploadTooLarge(f"В файле «{name}» больше {MAX_FEATURES} объектов")
    return layers


def _stem(name: str) -> str:
    path = PurePosixPath(name)
    return path.stem if path.suffix else path.name


def _layer(name: str, features: list[dict[str, Any]], warnings: list[str]):
    return UploadedLayer(
        name=name,
        role=detect_role(name, features),
        feature_collection={"type": "FeatureCollection", "features": features},
        warnings=warnings,
    )


@dataclass
class _ReadBatch:
    """Keep native bounds and defer missing CRSs across nested archive members."""

    declared: list = field(default_factory=list)
    pending: list = field(default_factory=list)

    def add(self, source, bounds):
        if source is not None and source.is_projected and bounds is not None:
            self.declared.append((source, bounds))

    def sibling(self, bounds):
        if bounds is None or _within_wgs84(bounds) or not self.declared:
            return None
        source = self.declared[0][0]
        if any(not source.equals(other) for other, _ in self.declared):
            return None
        # Native projected units need not be metres (e.g. a custom foot CRS).
        unit = source.axis_info[0].unit_conversion_factor
        margin = 50_000 / unit
        union = (
            min(box[0] for _, box in self.declared),
            min(box[1] for _, box in self.declared),
            max(box[2] for _, box in self.declared),
            max(box[3] for _, box in self.declared),
        )
        if not (
            union[0] - margin <= bounds[0] <= bounds[2] <= union[2] + margin
            and union[1] - margin <= bounds[1] <= bounds[3] <= union[3] + margin
        ):
            return None
        key = next(
            (key for key, zone in CRS_REGISTRY.items() if source.equals(zone.crs())),
            "custom",
        )
        return source, key

    def finish(self):
        for placeholder, bounds, resolve in self.pending:
            result = resolve(self.sibling(bounds))
            result.name = placeholder.name
            placeholder.__dict__.update(result.__dict__)
        self.pending.clear()


def _geojson_layer(
    name: str, content: bytes, *, crs: str | None = None, _batch=None, _sibling=None
) -> UploadedLayer:
    try:
        data = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgentsInputException(f"«{name}» — не корректный GeoJSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AgentsInputException(f"«{name}» — не корректный GeoJSON")
    kind = data.get("type")
    if kind == "FeatureCollection":
        raw_features = data.get("features") or []
    elif kind == "Feature":
        raw_features = [data]
    elif isinstance(data.get("coordinates"), list) or kind == "GeometryCollection":
        raw_features = [{"type": "Feature", "geometry": data, "properties": {}}]
    else:
        raise AgentsInputException(f"«{name}» — не корректный GeoJSON")
    source = _geojson_source(name, data.get("crs"))
    declared = source is not None
    warnings: list[str] = []
    if source is not None:
        source, warnings = complete_datum(source)
    project = _to_wgs84(source) if source is not None else None
    bounds = _raw_bounds(raw_features)
    engineering = source is None and bool(
        ((data.get("crs") or {}).get("properties") or {}).get("name")
    )
    if _batch is not None and not declared and not engineering:
        placeholder = _layer(name, [], [])
        _batch.pending.append(
            (
                placeholder,
                bounds,
                lambda sibling: _geojson_layer(
                    name, content, crs=crs, _sibling=sibling
                ),
            )
        )
        return placeholder
    if engineering and crs is None:
        raise AgentsInputException(UNKNOWN_CRS.format(name=name))
    if (
        not declared
        and crs is not None
        and (engineering or (bounds is not None and not _within_wgs84(bounds)))
    ):
        source, key = resolve_crs(crs)
        project = _to_wgs84(source)
        warnings.append(f"crs_from_request:{key}")
    elif not declared and not engineering and _sibling is not None:
        source, key = _sibling
        project = _to_wgs84(source)
        warnings.append(f"crs_from_sibling:{key}")
    elif not declared and data.get("crs") is None:
        project, warnings = _local_reprojection(
            (name, data.get("name")), _raw_bounds(raw_features)
        )
    features, feature_warnings = _normalized_features(
        name, raw_features, project, declared=declared or crs is not None
    )
    if _batch is not None and features and declared:
        _batch.add(source, bounds)
    return _layer(name, features, warnings + feature_warnings)


def _reprojection(name: str, crs: Any):
    """A coordinate transform to WGS84 lon/lat, or ``None`` when already WGS84."""
    source = _geojson_source(name, crs)
    if source is None or source.equals("EPSG:4326"):
        return None
    return _to_wgs84(complete_datum(source)[0])


def _geojson_source(name: str, crs: Any):
    """Keep engineering metadata from masquerading as a transformable CRS."""
    crs_name = ((crs or {}).get("properties") or {}).get("name") if crs else None
    if not crs_name:
        return None
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    if str(crs_name).casefold() in _WGS84_NAMES:
        return CRS.from_epsg(4326)
    try:
        source = CRS.from_user_input(crs_name)
    except CRSError as exc:
        zone = zone_by_name(str(crs_name))
        if zone is None:
            raise AgentsInputException(
                f"«{name}»: неизвестная система координат {crs_name}"
            ) from exc
        source = zone.crs()
    if source.is_engineering:
        return None
    return source


def _to_wgs84(source: Any):
    from pyproj import Transformer

    return Transformer.from_crs(source, "EPSG:4326", always_xy=True).transform


def _local_reprojection(
    names: tuple[str | None, ...], bounds: tuple[float, float, float, float] | None
):
    """A transform from the local CRS coordinates without a CRS are drawn in.

    Coordinates within the lon/lat range are taken for WGS84 as before; beyond it
    they are placed in the system the layer name or the coordinates tell.
    """
    if bounds is None or _within_wgs84(bounds):
        return None, []
    guessed = guess_crs(names, bounds)
    if guessed is None:
        return None, []
    logger.info("Upload layer {} has no CRS; {} is assumed", names[0], guessed.title)
    return _to_wgs84(guessed.crs()), [f"crs_missing_assumed:{guessed.key}"]


def _raw_bounds(
    raw_features: list[Any],
) -> tuple[float, float, float, float] | None:
    """Bounds of every position in raw GeoJSON features, or ``None`` without any."""
    xs: list[float] = []
    ys: list[float] = []

    def walk(value: Any) -> None:
        if (
            isinstance(value, list)
            and len(value) >= 2
            and all(isinstance(item, (int, float)) for item in value[:2])
        ):
            xs.append(float(value[0]))
            ys.append(float(value[1]))
        elif isinstance(value, list):
            for item in value:
                walk(item)

    def geometry(value: Any) -> None:
        if not isinstance(value, dict):
            return
        walk(value.get("coordinates"))
        for item in value.get("geometries") or []:
            geometry(item)

    for raw in raw_features:
        if isinstance(raw, dict):
            geometry(raw.get("geometry"))
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _normalized_features(
    name: str, raw_features: list[Any], project=None, *, declared: bool = False
) -> tuple[list[dict[str, Any]], list[str]]:
    features: list[dict[str, Any]] = []
    without_geometry = invalid = repaired = 0
    for raw in raw_features:
        if not isinstance(raw, dict) or not isinstance(raw.get("geometry"), dict):
            without_geometry += 1
            continue
        try:
            geometry = shape(_planar(raw["geometry"]))
        except (ValueError, TypeError, KeyError, AttributeError):
            invalid += 1
            continue
        if geometry.is_empty:
            without_geometry += 1
            continue
        if project is not None:
            geometry = transform(project, geometry)
        if not _within_wgs84(geometry.bounds):
            raise AgentsInputException(
                _WRONG_CRS.format(name=name)
                if declared
                else _UNKNOWN_CRS.format(name=name)
            )
        if not geometry.is_valid and geometry.geom_type in {
            "Polygon",
            "MultiPolygon",
        }:
            fixed = geometry.buffer(0)
            if not fixed.is_empty and fixed.geom_type in {"Polygon", "MultiPolygon"}:
                geometry = fixed
                repaired += 1
        feature = {
            "type": "Feature",
            "geometry": mapping(geometry),
            "properties": _json_safe(raw.get("properties") or {}),
        }
        if raw.get("id") is not None:
            feature["id"] = raw["id"]
        features.append(feature)
    warnings = []
    if without_geometry:
        warnings.append(f"features_without_geometry:{without_geometry}")
    if invalid:
        warnings.append(f"invalid_geometry_skipped:{invalid}")
    if repaired:
        warnings.append(f"invalid_geometry_repaired:{repaired}")
    return features, warnings


def _planar(geometry: dict[str, Any]) -> dict[str, Any]:
    """Drop Z/M from every position; generators mix 2D and 3D in one ring."""

    def positions(value: Any) -> Any:
        if (
            isinstance(value, (list, tuple))
            and value
            and isinstance(value[0], (int, float))
        ):
            return value[:2]
        if isinstance(value, (list, tuple)):
            return [positions(item) for item in value]
        return value

    if geometry.get("type") == "GeometryCollection":
        return {
            "type": "GeometryCollection",
            "geometries": [
                _planar(item)
                for item in geometry.get("geometries") or []
                if isinstance(item, dict)
            ],
        }
    return {**geometry, "coordinates": positions(geometry.get("coordinates"))}


_UNKNOWN_CRS = UNKNOWN_CRS
_WRONG_CRS = (
    "«{name}»: координаты не соответствуют указанной в файле системе координат. "
    "Проверьте её и загрузите файл заново или пересохраните его в EPSG:4326"
)


def _within_wgs84(bounds: tuple[float, float, float, float]) -> bool:
    min_x, min_y, max_x, max_y = bounds
    return -180 <= min_x <= max_x <= 180 and -90 <= min_y <= max_y <= 90


def _json_safe(value: Any) -> Any:
    """Properties as plain JSON: NaN becomes null, dates become ISO strings."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalars
        return _json_safe(value.item())
    return str(value)


_VECTOR_SUFFIXES = {
    ".geojson",
    ".json",
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
    ".sbn",
    ".sbx",
    ".qix",
    ".tab",
    ".dat",
    ".map",
    ".id",
    ".ind",
    ".mif",
    ".mid",
    ".gml",
    ".gfs",
    ".xsd",
    ".gpkg",
    ".kml",
    ".zip",
    ".7z",
}
_DATASET_SUFFIXES = _GEOJSON_SUFFIXES | {
    ".dbf",
    ".shp",
    ".tab",
    ".mif",
    ".gml",
    ".gpkg",
    ".kml",
}


@dataclass
class _ArchiveBudget:
    """Nested archives spend the same budget, including their container bytes."""

    members: int = 0
    unpacked: int = 0
    layers: int = 0
    skipped: Counter = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)
    batch: _ReadBatch = field(default_factory=_ReadBatch)
    incomplete: list[AgentsInputException] = field(default_factory=list)

    def reserve(self, size: int) -> None:
        self.members += 1
        self.unpacked += size
        if self.members > MAX_ARCHIVE_MEMBERS:
            raise UploadTooLarge(
                f"В архиве больше {MAX_ARCHIVE_MEMBERS} векторных файлов"
            )
        if size < 0 or self.unpacked > MAX_UNPACKED_BYTES:
            raise UploadTooLarge(
                f"Архив распаковывается в больше {MAX_UNPACKED_BYTES // (1024 * 1024)} МиБ"
            )

    def add_layers(self, count: int) -> None:
        self.layers += count
        if self.layers > MAX_LAYERS:
            raise UploadTooLarge(f"В архиве больше {MAX_LAYERS} слоёв")


def _zip_member_name(info: zipfile.ZipInfo) -> str:
    if not info.flag_bits & 0x800:
        try:
            return info.filename.encode("cp437").decode("cp866")
        except UnicodeError:
            pass
    return info.filename


def _archive_members(content: bytes, suffix: str, budget: _ArchiveBudget):
    """Filter metadata before opening streams; never extract archive paths to disk."""

    def select(name, size, is_dir):
        path = PurePosixPath(name.replace("\\", "/"))
        if is_dir or any(part.startswith(("__MACOSX", ".")) for part in path.parts):
            return None
        ext = path.suffix.casefold()
        if ext not in _VECTOR_SUFFIXES:
            budget.skipped[ext.lstrip(".") or "none"] += 1
            return None
        budget.reserve(size)
        return str(path)

    if suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            selected = []
            all_names = {
                _zip_member_name(info).replace("\\", "/").casefold()
                for info in archive.infolist()
            }
            for info in archive.infolist():
                name = select(_zip_member_name(info), info.file_size, info.is_dir())
                if name is not None:
                    selected.append((name, info))
            for name, info in selected:
                with archive.open(info) as stream:
                    data = stream.read(info.file_size + 1)
                if len(data) != info.file_size:
                    raise AgentsInputException(
                        "Размер файла в архиве не соответствует заголовку"
                    )
                yield name, data, all_names
    else:
        try:
            import py7zr
        except ImportError as exc:
            raise AgentsInputException(
                "Для чтения 7z нужен модуль py7zr. Перепакуйте архив в ZIP"
            ) from exc
        from py7zr.io import Py7zIO, WriterFactory

        with py7zr.SevenZipFile(io.BytesIO(content), "r") as archive:
            infos = archive.list()
            all_names = {info.filename.replace("\\", "/").casefold() for info in infos}
            selected = {}
            for info in infos:
                name = select(info.filename, info.uncompressed, info.is_directory)
                if name is not None:
                    if info.filename in selected:
                        raise AgentsInputException(
                            "Повторяющееся имя файла в 7z-архиве"
                        )
                    selected[info.filename] = (name, info.uncompressed)

            class LimitedIO(Py7zIO):
                def __init__(self, limit):
                    self.buffer = io.BytesIO()
                    self.limit = limit

                def write(self, data):
                    if self.buffer.tell() + len(data) > self.limit:
                        raise UploadTooLarge(
                            "Размер файла в 7z превышает заявленный размер"
                        )
                    return self.buffer.write(data)

                def read(self, size=None):
                    return self.buffer.read(size)

                def seek(self, offset, whence=0):
                    return self.buffer.seek(offset, whence)

                def flush(self):
                    return self.buffer.flush()

                def size(self):
                    return len(self.buffer.getbuffer())

            class Factory(WriterFactory):
                def __init__(self):
                    self.products = {}

                def create(self, filename):
                    product = LimitedIO(selected[filename][1])
                    self.products[filename] = product
                    return product

            if selected:
                factory = Factory()
                archive.extract(targets=list(selected), factory=factory)
                for original, (name, size) in selected.items():
                    product = factory.products[original]
                    if product.size() != size:
                        raise AgentsInputException(
                            "Размер файла в 7z не соответствует заголовку"
                        )
                    yield name, product.buffer.getvalue(), all_names


def _zip_layers(
    name: str, content: bytes, *, crs: str | None = None
) -> list[UploadedLayer]:
    """Keep the existing entry point for callers reading ZIP uploads."""
    return _archive_layers(name, content, crs=crs)


def _archive_layers(
    name: str,
    content: bytes,
    *,
    crs: str | None = None,
    _budget: _ArchiveBudget | None = None,
    _depth: int = 0,
) -> list[UploadedLayer]:
    budget = _budget or _ArchiveBudget()
    layers: list[UploadedLayer] = []
    groups: dict[str, dict[str, bytes]] = {}
    all_names = set()
    try:
        for member, data, names in _archive_members(
            content, PurePosixPath(name).suffix.casefold(), budget
        ):
            all_names.update(names)
            path = PurePosixPath(member)
            suffix = path.suffix.casefold()
            if suffix in {".zip", ".7z"}:
                if _depth >= 1:
                    budget.warnings.append("nested_archive_skipped")
                else:
                    nested = _archive_layers(
                        member, data, crs=crs, _budget=budget, _depth=_depth + 1
                    )
                    for item in nested:
                        item.name = f"{path.with_suffix('')}/{item.name}"
                    layers.extend(nested)
            else:
                stem = str(path.with_suffix(""))
                groups.setdefault(stem, {})[suffix] = data
    except AgentsInputException:
        raise
    except Exception as exc:
        raise AgentsInputException(f"«{name}» — повреждённый архив: {exc}") from exc

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index, (stem, siblings) in enumerate(groups.items()):
            for suffix in sorted(_DATASET_SUFFIXES & siblings.keys()):
                if suffix == ".dbf" and ({".shp", ".tab"} & siblings.keys()):
                    continue
                title = _stem(stem + suffix)
                # Retain folders only where basenames would otherwise collide.
                if (
                    sum(
                        PurePosixPath(other).name == PurePosixPath(stem).name
                        for other in groups
                    )
                    > 1
                ):
                    title = stem
                if suffix == ".tab":
                    header = siblings[suffix].decode("latin-1")
                    warning = None
                    if stem.casefold() + ".tda" in all_names:
                        warning = "mapinfo_nativex_unsupported"
                    elif re.search(r'\bType\s+"?RASTER\b', header, re.IGNORECASE):
                        warning = "raster_tab_skipped"
                    elif re.search(
                        r'(?:\bType\s+"?SEAMLESS|"?\\IsSeamless"?\s*=\s*"?TRUE)',
                        header,
                        re.IGNORECASE,
                    ):
                        warning = "seamless_tab_skipped"
                    if warning:
                        budget.warnings.append(warning)
                        continue
                attribute_tab = suffix == ".tab" and bool(
                    re.search(r'\bType\s+"?(?:DBF|ASCII|CSV)\b', header, re.IGNORECASE)
                )
                required = (
                    _SHAPEFILE_SIDECARS
                    if suffix == ".shp"
                    else (
                        (".dat", ".map", ".id")
                        if suffix == ".tab" and not attribute_tab
                        else ()
                    )
                )
                missing = [ext for ext in required if ext not in siblings]
                dataset_root = root / str(index)
                dataset_root.mkdir(exist_ok=True)
                path = dataset_root / (PurePosixPath(stem).name + suffix)
                for ext, data in siblings.items():
                    path.with_suffix(ext).write_bytes(data)
                if suffix == ".tab" and missing == [".map", ".id"]:
                    # Native attribute tables legitimately have only TAB/DAT.
                    # Probe once; a failed read still uses the sidecar error below.
                    try:
                        _, table_warnings = read_vector(
                            path, names=(title,), native=True
                        )
                    except AgentsInputException:
                        table_warnings = []
                    if table_warnings:
                        budget.warnings.extend(table_warnings)
                        continue
                if missing:
                    budget.incomplete.append(
                        AgentsInputException(
                            f"В архиве «{name}» у слоя «{stem}» нет файлов {', '.join(missing)}"
                        )
                    )
                    budget.warnings.append(
                        f"incomplete_layer_skipped:{stem}:{','.join(missing)}"
                    )
                    continue
                if suffix in _GEOJSON_SUFFIXES:
                    budget.add_layers(1)
                    layers.append(
                        _geojson_layer(
                            title, siblings[suffix], crs=crs, _batch=budget.batch
                        )
                    )
                    continue
                if suffix == ".gpkg":
                    result = _gpkg_layers(
                        title, path, crs=crs, _budget=budget, _batch=budget.batch
                    )
                else:
                    budget.add_layers(1)
                    result = _vector_layers(title, path, crs=crs, _batch=budget.batch)
                layers.extend(result)
    if _depth == 0:
        budget.batch.finish()
        if not any(item.features for item in layers) and budget.incomplete:
            raise budget.incomplete[0]
        warnings = list(dict.fromkeys(budget.warnings))
        warnings.extend(
            w for item in layers if not item.features for w in item.warnings
        )
        if budget.skipped:
            warnings.append(
                "skipped_members:"
                + ",".join(
                    f"{ext}={count}" for ext, count in sorted(budget.skipped.items())
                )
            )
        if not layers:
            details = ", ".join(warnings)
            raise AgentsInputException(
                f"В архиве «{name}» нет поддерживаемых векторных слоёв"
                + (f": {details}" if details else "")
            )
        first = next((item for item in layers if item.features), layers[0])
        first.warnings.extend(warnings)
    return layers


def _gpkg_layers(
    name: str,
    path: Path,
    *,
    crs: str | None = None,
    _budget: _ArchiveBudget | None = None,
    _batch: _ReadBatch | None = None,
) -> list[UploadedLayer]:
    import pyogrio

    try:
        listed = pyogrio.list_layers(path)
    except Exception as exc:  # pyogrio raises driver-specific errors
        raise AgentsInputException(f"«{name}» — не корректный GeoPackage") from exc
    if len(listed) > MAX_LAYERS:
        raise UploadTooLarge(f"В файле «{name}» больше {MAX_LAYERS} слоёв")
    if _budget is not None:
        _budget.add_layers(len(listed))
    batch = _batch or _ReadBatch()
    layers: list[UploadedLayer] = []
    for row in listed:
        layer_name = str(row[0])
        title = name if len(listed) == 1 else f"{name}:{layer_name}"
        layers.extend(
            _vector_layers(title, path, layer=layer_name, crs=crs, _batch=batch)
        )
    if _batch is None:
        batch.finish()
        warnings = [w for item in layers if not item.features for w in item.warnings]
        if warnings:
            first = next((item for item in layers if item.features), None)
            if first is not None:
                first.warnings.extend(warnings)
    return layers


def _vector_layers(
    name: str,
    path: Path,
    *,
    layer: str | None = None,
    crs: str | None = None,
    _batch: _ReadBatch | None = None,
) -> list[UploadedLayer]:
    if layer is None and path.suffix.casefold() == ".gml":
        import pyogrio

        try:
            listed = pyogrio.list_layers(path)
        except Exception as exc:
            raise AgentsInputException(
                f"Не удалось прочитать слой «{name}»: {exc}"
            ) from exc
        if len(listed) > 1:
            batch = _batch or _ReadBatch()
            result = []
            for row in listed:
                result.extend(
                    _vector_layers(
                        f"{name}:{row[0]}",
                        path,
                        layer=str(row[0]),
                        crs=crs,
                        _batch=batch,
                    )
                )
            if _batch is None:
                batch.finish()
                first = next((item for item in result if item.features), None)
                if first is not None:
                    first.warnings.extend(
                        w for item in result if not item.features for w in item.warnings
                    )
            return result
    frame, local_warnings = read_vector(
        path, layer=layer, explicit_crs=crs, names=(name,), native=True
    )
    if frame.empty:
        return [_layer(name, [], local_warnings)]
    bounds = tuple(float(value) for value in frame.total_bounds)
    source = frame.crs
    if _batch is not None and source is None:
        placeholder = _layer(name, [], [])
        _batch.pending.append(
            (
                placeholder,
                bounds,
                lambda sibling: _frame_layer(name, frame, crs, sibling),
            )
        )
        return [placeholder]
    result = _frame_layer(name, frame, crs)
    if _batch is not None and result.features and source is not None:
        _batch.add(complete_datum(source)[0], bounds)
    return [result]


def _frame_layer(name, frame, crs, sibling=None):
    """Serialize only after native CRS resolution, without reading files twice."""
    frame, local_warnings = reproject_vector(
        frame,
        names=(name,),
        requested=resolve_crs(crs) if crs else None,
        sibling=sibling,
    )
    declared = True
    columns = [column for column in frame.columns if column != frame.geometry.name]
    raw_features = []
    for _, row in frame.iterrows():
        geometry = row[frame.geometry.name]
        raw_features.append(
            {
                "type": "Feature",
                "geometry": None if geometry is None else mapping(geometry),
                "properties": {column: row[column] for column in columns},
            }
        )
    features, warnings = _normalized_features(name, raw_features, declared=declared)
    if frame.crs is None:
        warnings.append("crs_missing_assumed_wgs84")
    return _layer(name, features, local_warnings + warnings)


@dataclass
class UserFiles:
    """The layers of every upload a request names, with confirmed entity mappings."""

    upload_ids: tuple[str, ...]
    layers: list[UploadedLayer]
    confirmed_mapping: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    # LLM classifications of earlier runs: a rerun maps the same groups the same way.
    classified: dict[str, dict[str, Any]] = field(default_factory=dict)

    def by_role(self, role: LayerRole) -> list[UploadedLayer]:
        return [layer for layer in self.layers if layer.role == role]

    @property
    def has_project(self) -> bool:
        return any(layer.features for layer in self.by_role("project"))

    def summary(self) -> list[dict[str, Any]]:
        return [layer.summary() for layer in self.layers]


class ComplianceUploadService:
    """Create, read and update uploads; every access checks the owner."""

    def __init__(
        self,
        state_store: "PipelineStateStore",
        layer_cache: LayerCache | None = None,
    ) -> None:
        self.state_store = state_store
        self.layer_cache = layer_cache or LayerCache(ttl_seconds=UPLOAD_TTL_SECONDS)

    async def create(
        self,
        owner: str,
        files: list[tuple[str, bytes]],
        role: LayerRole | None = None,
        crs: str | None = None,
    ) -> dict[str, Any]:
        if not files:
            raise AgentsInputException("Не передано ни одного файла")
        layers: list[UploadedLayer] = []
        for filename, content in files:
            layers.extend(parse_upload(filename, content, crs=crs))
            if len(layers) > MAX_LAYERS:
                raise UploadTooLarge(f"В загрузке больше {MAX_LAYERS} слоёв")
        names: dict[str, int] = {}
        for layer in layers:
            # Layer names are the handle for role overrides: keep them unique.
            names[layer.name] = names.get(layer.name, 0) + 1
            if names[layer.name] > 1:
                layer.name = f"{layer.name} ({names[layer.name]})"
            if role is not None:
                layer.role = role
                layer.role_source = "user"
        upload_id = str(uuid.uuid4())
        self.layer_cache.purge_expired()
        try:
            stored = [
                layer.to_record(
                    await self.layer_cache.write(
                        upload_id, index, layer.feature_collection
                    )
                )
                for index, layer in enumerate(layers)
            ]
            created_at = datetime.now(timezone.utc)
            record = {
                "upload_id": upload_id,
                "owner": owner,
                "created_at": created_at.isoformat(),
                "expires_at": (
                    created_at + timedelta(seconds=UPLOAD_TTL_SECONDS)
                ).isoformat(),
                "files": [filename for filename, _ in files],
                "crs": crs,
                "layers": stored,
                "confirmed_mapping": {},
                "classified": {},
            }
            await self.state_store.save_compliance_upload(upload_id, record)
        except BaseException:
            self.layer_cache.delete(upload_id)
            raise
        logger.info(
            "Compliance upload {} stored: files={} layers={} features={}",
            record["upload_id"],
            len(files),
            len(layers),
            sum(len(layer.features) for layer in layers),
        )
        return self.summary(record)

    async def get(self, upload_id: str, owner: str) -> dict[str, Any]:
        record = await self.state_store.get_compliance_upload(upload_id)
        # Someone else's upload is reported exactly like a missing one.
        if not record or record.get("owner") != owner:
            raise _expired(upload_id)
        if not all(
            self.layer_cache.exists(item["file"])
            for item in record["layers"]
            if "file" in item
        ):
            # The cache lost the layers (a recreated container): the record is moot.
            await self.state_store.delete_compliance_upload(upload_id)
            raise _expired(upload_id)
        return record

    async def set_roles(
        self, upload_id: str, owner: str, roles: dict[str, LayerRole]
    ) -> dict[str, Any]:
        record = await self.get(upload_id, owner)
        known = {layer["name"] for layer in record["layers"]}
        unknown = sorted(set(roles) - known)
        if unknown:
            raise AgentsInputException(
                "В загрузке нет слоёв: " + ", ".join(unknown), sorted(known)
            )
        for layer in record["layers"]:
            if layer["name"] in roles:
                layer["role"] = roles[layer["name"]]
                layer["role_source"] = "user"
        await self.state_store.update_compliance_upload(upload_id, record)
        return self.summary(record)

    async def delete(self, upload_id: str, owner: str) -> None:
        await self.get(upload_id, owner)
        await self.state_store.delete_compliance_upload(upload_id)
        self.layer_cache.delete(upload_id)

    async def load(
        self,
        upload_ids: list[str] | tuple[str, ...],
        owner: str,
        namespace: MappingNamespace = "compliance",
    ):
        confirmed_field, classified_field = _MAPPING_FIELDS[namespace]
        layers: list[UploadedLayer] = []
        confirmed: dict[str, dict[str, Any] | None] = {}
        classified: dict[str, dict[str, Any]] = {}
        for upload_id in dict.fromkeys(upload_ids):
            record = await self.get(upload_id, owner)
            for item in record["layers"]:
                layers.append(await self._layer(upload_id, item))
            confirmed.update(record.get(confirmed_field) or {})
            classified.update(record.get(classified_field) or {})
        return UserFiles(
            upload_ids=tuple(dict.fromkeys(upload_ids)),
            layers=layers,
            confirmed_mapping=confirmed,
            classified=classified,
        )

    async def cache_classifications(
        self,
        upload_ids: list[str] | tuple[str, ...],
        owner: str,
        classified: dict[str, dict[str, Any]],
        namespace: MappingNamespace = "compliance",
    ) -> None:
        """Keep the LLM's answers so a rerun asks neither it nor the user again."""
        if not classified:
            return
        field_name = _MAPPING_FIELDS[namespace][1]
        for upload_id in dict.fromkeys(upload_ids):
            record = await self.get(upload_id, owner)
            record[field_name] = {**(record.get(field_name) or {}), **classified}
            await self.state_store.update_compliance_upload(upload_id, record)

    async def confirm_mapping(
        self,
        upload_ids: list[str] | tuple[str, ...],
        owner: str,
        mapping_update: dict[str, dict[str, Any] | None],
        namespace: MappingNamespace = "compliance",
    ) -> None:
        """Keep the user's answer with the uploads so a rerun does not ask again."""
        field_name = _MAPPING_FIELDS[namespace][0]
        for upload_id in dict.fromkeys(upload_ids):
            record = await self.get(upload_id, owner)
            record[field_name] = {**(record.get(field_name) or {}), **mapping_update}
            await self.state_store.update_compliance_upload(upload_id, record)

    async def _layer(self, upload_id: str, item: dict[str, Any]) -> UploadedLayer:
        if "file" not in item:  # a record written before layers moved to the cache
            return UploadedLayer.from_record(item, item["feature_collection"])
        try:
            feature_collection = await self.layer_cache.read(item["file"])
        except LayerMissing:
            await self.state_store.delete_compliance_upload(upload_id)
            raise _expired(upload_id) from None
        return UploadedLayer.from_record(item, feature_collection)

    @staticmethod
    def summary(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "upload_id": record["upload_id"],
            "created_at": record.get("created_at"),
            "expires_at": record.get("expires_at"),
            "files": record.get("files") or [],
            "layers": [_record_summary(item) for item in record["layers"]],
        }


def _expired(upload_id: str) -> AgentsNotFound:
    return AgentsNotFound("Загрузка не найдена или срок её хранения истёк", upload_id)
