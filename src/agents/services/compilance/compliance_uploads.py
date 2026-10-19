"""User files for a compliance check: parsing, layer roles and storage.

A user checks their own design against the norms by uploading its layers: GeoJSON,
a ZIP of GeoJSON files or Shapefiles, or a GeoPackage. Each file (each archive
member, each GeoPackage layer) becomes one layer in WGS84 with 2D geometry, and
gets a role:

- ``project`` — the designed objects; only they are checked;
- ``context`` — the surroundings (OSM, the existing state); they are sources and
  neighbours of distance rules, never checked themselves;
- ``border`` — the boundary of the design; it tells where the design is.

A layer without a CRS is WGS84 unless its coordinates fall outside lon/lat; then it
is placed in МСК-47 or Web Mercator when its name or coordinates tell so (see
``crs_guess``), and is sent back to be uploaded with its CRS otherwise.

The role is guessed from the layer name and content and can be set explicitly.
An upload lives for a day under an ``upload_id`` owned by the user who uploaded it:
its record (roles, mapping answers, a summary of every layer) in Redis, the parsed
layers themselves as files in the application's ``LayerCache``.
"""

from __future__ import annotations

import io
import json
import math
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from shapely.geometry import mapping, shape
from shapely.ops import transform

from src.agents.common.exceptions.base_exceptions import (
    AgentsInputException,
    AgentsNotFound,
)
from src.agents.common.files.layer_cache import LayerCache, LayerMissing
from src.agents.services.compilance.crs_guess import guess_crs, zone_by_name

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

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
# A ZIP may hold far more than it weighs; unpacking stops past this many bytes.
MAX_UNPACKED_BYTES = 200 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 200
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
    if (types == {"border"}) or any(hint in lowered for hint in _BORDER_HINTS):
        return "border"
    if any(hint in lowered for hint in _CONTEXT_HINTS):
        return "context"
    return "project"


def parse_upload(filename: str, content: bytes) -> list[UploadedLayer]:
    """Every layer of one uploaded file; raises ``AgentsInputException`` on bad input."""
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadTooLarge(
            f"Файл «{filename}» больше {MAX_UPLOAD_BYTES // (1024 * 1024)} МиБ"
        )
    name = PurePosixPath(filename.replace("\\", "/")).name or "layer.geojson"
    suffix = PurePosixPath(name).suffix.casefold()
    if suffix in _GEOJSON_SUFFIXES:
        layers = [_geojson_layer(_stem(name), content)]
    elif suffix == ".zip":
        layers = _zip_layers(name, content)
    elif suffix == ".gpkg":
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.gpkg"
            path.write_bytes(content)
            layers = _gpkg_layers(_stem(name), path)
    elif suffix == ".shp":
        raise AgentsInputException(
            f"Shapefile «{name}» загрузите ZIP-архивом вместе с файлами .shx, .dbf "
            "и .prj"
        )
    else:
        raise AgentsInputException(
            f"Формат файла «{name}» не поддерживается. Загрузите GeoJSON, ZIP "
            "(GeoJSON или Shapefile) или GeoPackage"
        )
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


def _geojson_layer(name: str, content: bytes) -> UploadedLayer:
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
    project = _reprojection(name, data.get("crs"))
    warnings: list[str] = []
    if project is None and data.get("crs") is None:
        project, warnings = _local_reprojection(
            (name, data.get("name")), _raw_bounds(raw_features)
        )
    features, feature_warnings = _normalized_features(
        name, raw_features, project, declared=data.get("crs") is not None
    )
    return _layer(name, features, warnings + feature_warnings)


def _reprojection(name: str, crs: Any):
    """A coordinate transform to WGS84 lon/lat, or ``None`` when already WGS84."""
    crs_name = ((crs or {}).get("properties") or {}).get("name") if crs else None
    if not crs_name or str(crs_name).casefold() in _WGS84_NAMES:
        return None
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    try:
        source = CRS.from_user_input(crs_name)
    except CRSError as exc:
        zone = zone_by_name(str(crs_name))
        if zone is None:
            raise AgentsInputException(
                f"«{name}»: неизвестная система координат {crs_name}"
            ) from exc
        source = zone.crs()
    if source == CRS.from_epsg(4326):
        return None
    return _to_wgs84(source)


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
        if isinstance(value, list) and value and isinstance(value[0], (int, float)):
            return value[:2]
        if isinstance(value, list):
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


_UNKNOWN_CRS = (
    "«{name}»: в файле не указана система координат, а координаты не похожи ни на "
    "широту и долготу (EPSG:4326), ни на МСК-47, ни на EPSG:3857. Загрузите файл "
    "заново с указанной системой координат — GeoJSON с полем crs, Shapefile в ZIP "
    "вместе с файлом .prj или GeoPackage — либо пересохраните его в EPSG:4326"
)
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


def _zip_layers(name: str, content: bytes) -> list[UploadedLayer]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise AgentsInputException(f"«{name}» — повреждённый ZIP-архив") from exc
    members = [
        info
        for info in archive.infolist()
        if not info.is_dir()
        and not any(
            part.startswith(("__MACOSX", "."))
            for part in PurePosixPath(info.filename).parts
        )
    ]
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise UploadTooLarge(f"В архиве «{name}» больше {MAX_ARCHIVE_MEMBERS} файлов")
    if sum(info.file_size for info in members) > MAX_UNPACKED_BYTES:
        raise UploadTooLarge(
            f"Архив «{name}» распаковывается в больше "
            f"{MAX_UNPACKED_BYTES // (1024 * 1024)} МиБ"
        )
    by_suffix: dict[str, list[zipfile.ZipInfo]] = {}
    for info in members:
        by_suffix.setdefault(PurePosixPath(info.filename).suffix.casefold(), []).append(
            info
        )

    layers: list[UploadedLayer] = []
    for info in by_suffix.get(".geojson", []) + by_suffix.get(".json", []):
        layers.append(_geojson_layer(_stem(info.filename), archive.read(info)))

    shapefiles = by_suffix.get(".shp", [])
    packages = by_suffix.get(".gpkg", [])
    if shapefiles or packages:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, info in enumerate(packages):
                # Members are written under names of our own: no path from the
                # archive reaches the file system.
                path = root / f"package_{index}.gpkg"
                path.write_bytes(archive.read(info))
                layers.extend(_gpkg_layers(_stem(info.filename), path))
            for index, info in enumerate(shapefiles):
                stem = str(PurePosixPath(info.filename).with_suffix(""))
                siblings = {
                    PurePosixPath(other.filename).suffix.casefold(): other
                    for other in members
                    if str(PurePosixPath(other.filename).with_suffix("")) == stem
                }
                missing = [s for s in _SHAPEFILE_SIDECARS if s not in siblings]
                if missing:
                    raise AgentsInputException(
                        f"В архиве «{name}» у Shapefile «{PurePosixPath(stem).name}» "
                        f"нет файлов {', '.join(missing)}"
                    )
                for suffix, other in siblings.items():
                    (root / f"shape_{index}{suffix}").write_bytes(archive.read(other))
                layers.extend(
                    _vector_layers(
                        PurePosixPath(stem).name, root / f"shape_{index}.shp"
                    )
                )
    if not layers:
        raise AgentsInputException(
            f"В архиве «{name}» нет GeoJSON, Shapefile или GeoPackage"
        )
    return layers


def _gpkg_layers(name: str, path: Path) -> list[UploadedLayer]:
    import pyogrio

    try:
        listed = pyogrio.list_layers(path)
    except Exception as exc:  # pyogrio raises driver-specific errors
        raise AgentsInputException(f"«{name}» — не корректный GeoPackage") from exc
    layers: list[UploadedLayer] = []
    for row in listed:
        layer_name = str(row[0])
        title = name if len(listed) == 1 else f"{name}:{layer_name}"
        layers.extend(_vector_layers(title, path, layer=layer_name))
    return layers


def _vector_layers(
    name: str, path: Path, *, layer: str | None = None
) -> list[UploadedLayer]:
    import geopandas as gpd

    try:
        frame = gpd.read_file(path, layer=layer)
    except Exception as exc:  # pyogrio raises driver-specific errors
        raise AgentsInputException(
            f"Не удалось прочитать слой «{name}»: {exc}"
        ) from exc
    if frame.empty or frame.geometry is None:
        return []
    local_warnings: list[str] = []
    declared = frame.crs is not None
    if not declared:
        bounds = tuple(float(value) for value in frame.total_bounds)
        guessed = None if _within_wgs84(bounds) else guess_crs((name,), bounds)
        if guessed is None:
            logger.warning("Uploaded layer {} has no CRS; WGS84 is assumed", name)
        else:
            logger.info(
                "Uploaded layer {} has no CRS; {} is assumed", name, guessed.title
            )
            frame = frame.set_crs(guessed.crs()).to_crs("EPSG:4326")
            local_warnings.append(f"crs_missing_assumed:{guessed.key}")
    elif not frame.crs.equals("EPSG:4326"):
        frame = frame.to_crs("EPSG:4326")
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
    return [_layer(name, features, local_warnings + warnings)]


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
    ) -> dict[str, Any]:
        if not files:
            raise AgentsInputException("Не передано ни одного файла")
        layers: list[UploadedLayer] = []
        for filename, content in files:
            layers.extend(parse_upload(filename, content))
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
