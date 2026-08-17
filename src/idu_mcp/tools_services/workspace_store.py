"""Ephemeral multi-process workspace backed by tmpfs and Redis metadata."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import geopandas as gpd
import pandas as pd
from fastmcp.exceptions import ToolError
from shapely.geometry import shape


class WorkspaceStore:
    """Store immutable frames outside Redis and expose opaque capability handles."""

    KEY_PREFIX = "idu_workspace:artifact:"

    def __init__(
        self,
        redis_client,
        *,
        root: str,
        ttl_seconds: int = 3600,
        max_dataset_bytes: int = 128 * 1024 * 1024,
        max_total_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        self.redis = redis_client
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.max_dataset_bytes = max_dataset_bytes
        self.max_total_bytes = max_total_bytes
        self.watermark_bytes = int(max_total_bytes * 0.8)
        self._write_lock = asyncio.Lock()

    async def create(
        self,
        frame: pd.DataFrame,
        *,
        owner_id: str,
        chat_id: str,
        lineage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist one immutable frame and register a hard-expiry handle."""

        if not owner_id or not chat_id:
            raise ToolError("owner_id и chat_id обязательны для изоляции workspace")
        handle = str(uuid4())
        is_geo = isinstance(frame, gpd.GeoDataFrame) and "geometry" in frame.columns
        suffix = ".geoparquet" if is_geo else ".parquet"
        path = self._safe_path(handle + suffix)
        created = time.time()
        async with self._write_lock:
            async with self._maintenance_lock():
                await self._sweep()
                try:
                    # tmpfs I/O is bounded by max_dataset_bytes and executes under
                    # process-local and container-wide file locks. Python 3.14
                    # currently leaves the default executor alive after pyarrow
                    # calls, so asyncio.to_thread would leak workers on shutdown.
                    frame.to_parquet(path, index=False)
                except Exception as exc:
                    path.unlink(missing_ok=True)
                    raise ToolError(
                        f"Не удалось сохранить workspace-набор: {exc}"
                    ) from exc
                size = path.stat().st_size
                if size > self.max_dataset_bytes:
                    path.unlink(missing_ok=True)
                    raise ToolError(
                        "Набор превышает лимит workspace "
                        f"({size} > {self.max_dataset_bytes} байт)"
                    )
                await self._evict_to_watermark()

                metadata = {
                    "handle": handle,
                    "owner_id": owner_id,
                    "chat_id": chat_id,
                    "path": str(path),
                    "format": "geoparquet" if is_geo else "parquet",
                    "rows": int(len(frame)),
                    "columns": [str(column) for column in frame.columns],
                    "profile": self._profile(frame),
                    "size_bytes": size,
                    "created_at": created,
                    "expires_at": created + self.ttl_seconds,
                    "lineage": lineage or {},
                }
                await self.redis.set(
                    self.KEY_PREFIX + handle,
                    json.dumps(metadata, ensure_ascii=False),
                    ex=self.ttl_seconds,
                )
        return self.public_metadata(metadata)

    @asynccontextmanager
    async def _maintenance_lock(self):
        """Serialize file registration across all workers in this container."""

        lock_path = self._safe_path(".workspace.lock")
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        deadline = time.monotonic() + 60
        try:
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise ToolError(
                            "Workspace занят другим процессом; повторите вызов"
                        )
                    await asyncio.sleep(0.05)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    async def load(
        self, handle: str, *, owner_id: str, chat_id: str
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        metadata = await self._metadata(handle, owner_id=owner_id, chat_id=chat_id)
        path = self._safe_registered_path(metadata["path"])
        if not path.exists():
            await self.redis.delete(self.KEY_PREFIX + handle)
            raise ToolError(
                "Workspace-набор недоступен; выполните получение данных заново"
            )
        try:
            if metadata["format"] == "geoparquet":
                frame = gpd.read_parquet(path)
            else:
                frame = pd.read_parquet(path)
        except Exception as exc:
            raise ToolError(f"Не удалось прочитать workspace-набор: {exc}") from exc
        return frame, metadata

    async def derive(
        self,
        frame: pd.DataFrame,
        *,
        parent_handle: str,
        owner_id: str,
        chat_id: str,
        operation: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.create(
            frame,
            owner_id=owner_id,
            chat_id=chat_id,
            lineage={
                "parent_handle": parent_handle,
                "operation": operation,
                "arguments": arguments,
            },
        )

    async def release(self, handle: str, *, owner_id: str, chat_id: str) -> bool:
        metadata = await self._metadata(handle, owner_id=owner_id, chat_id=chat_id)
        path = self._safe_registered_path(metadata["path"])
        path.unlink(missing_ok=True)
        await self.redis.delete(self.KEY_PREFIX + handle)
        return True

    async def describe(
        self, handle: str, *, owner_id: str, chat_id: str
    ) -> dict[str, Any]:
        frame, metadata = await self.load(handle, owner_id=owner_id, chat_id=chat_id)
        result = self.public_metadata(metadata)
        result["dtypes"] = {name: str(dtype) for name, dtype in frame.dtypes.items()}
        result["null_counts"] = {
            name: int(value) for name, value in frame.isna().sum().items()
        }
        if isinstance(frame, gpd.GeoDataFrame):
            result["crs"] = str(frame.crs) if frame.crs else None
            result["geometry_type_counts"] = {
                str(name): int(count)
                for name, count in frame.geometry.geom_type.value_counts().items()
            }
        return result

    async def _metadata(
        self, handle: str, *, owner_id: str, chat_id: str
    ) -> dict[str, Any]:
        raw = await self.redis.get(self.KEY_PREFIX + handle)
        if not raw:
            raise ToolError("Workspace handle не найден или истёк")
        metadata = json.loads(raw)
        if metadata.get("owner_id") != owner_id or metadata.get("chat_id") != chat_id:
            raise ToolError(
                "Workspace handle принадлежит другому пользователю или чату"
            )
        if float(metadata["expires_at"]) <= time.time():
            path = self._safe_registered_path(metadata["path"])
            path.unlink(missing_ok=True)
            await self.redis.delete(self.KEY_PREFIX + handle)
            raise ToolError("Workspace handle истёк")
        return metadata

    async def _sweep(self) -> None:
        now = time.time()
        async for key in self.redis.scan_iter(match=self.KEY_PREFIX + "*"):
            raw = await self.redis.get(key)
            if not raw:
                continue
            metadata = json.loads(raw)
            if float(metadata.get("expires_at", 0)) > now:
                continue
            try:
                self._safe_registered_path(metadata["path"]).unlink(missing_ok=True)
            finally:
                await self.redis.delete(key)
        registered: set[Path] = set()
        async for key in self.redis.scan_iter(match=self.KEY_PREFIX + "*"):
            raw = await self.redis.get(key)
            if raw:
                registered.add(self._safe_registered_path(json.loads(raw)["path"]))
        for path in self.root.glob("*.parquet"):
            if path.resolve() not in registered:
                path.unlink(missing_ok=True)
        for path in self.root.glob("*.geoparquet"):
            if path.resolve() not in registered:
                path.unlink(missing_ok=True)

    async def _evict_to_watermark(self) -> None:
        files = [
            path
            for pattern in ("*.parquet", "*.geoparquet")
            for path in self.root.glob(pattern)
        ]
        total = sum(path.stat().st_size for path in files if path.exists())
        if total <= self.watermark_bytes:
            return
        for path in sorted(files, key=lambda item: item.stat().st_mtime):
            handle = path.name.split(".", 1)[0]
            metadata_raw = await self.redis.get(self.KEY_PREFIX + handle)
            path_size = path.stat().st_size if path.exists() else 0
            path.unlink(missing_ok=True)
            if metadata_raw:
                await self.redis.delete(self.KEY_PREFIX + handle)
            total -= path_size
            if total <= self.watermark_bytes:
                break
        if total > self.watermark_bytes:
            raise ToolError("Недостаточно памяти workspace для нового набора")

    def _safe_path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if path.parent != self.root:
            raise ToolError("Недопустимый путь workspace")
        return path

    def _safe_registered_path(self, value: str) -> Path:
        path = Path(value).resolve()
        if path.parent != self.root:
            raise ToolError("Повреждён путь workspace")
        return path

    @staticmethod
    def public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in metadata.items()
            if key not in {"path", "owner_id"}
        }

    @classmethod
    def _profile(cls, frame: pd.DataFrame) -> dict[str, Any]:
        """Return bounded schema metadata suitable for Redis and an LLM prompt."""

        uniques: dict[str, list[Any]] = {}
        for name in list(frame.columns)[:24]:
            if name == "geometry":
                continue
            series = frame[name]
            cardinality = int(series.nunique(dropna=False))
            if cardinality <= 20 and len(uniques) < 8:
                uniques[str(name)] = [
                    cls._json_scalar(value)
                    for value in series.drop_duplicates().head(20).tolist()
                ]
        return {
            "dtypes": {str(name): str(dtype) for name, dtype in frame.dtypes.items()},
            "null_counts": {
                str(name): int(value) for name, value in frame.isna().sum().items()
            },
            "bounded_unique_values": uniques,
        }

    @staticmethod
    def _json_scalar(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return None if pd.isna(value) else value
        if hasattr(value, "item"):
            value = value.item()
            if isinstance(value, (str, int, float, bool)):
                return None if pd.isna(value) else value
        return str(value)


def frame_from_payload(
    records: list[dict[str, Any]] | None,
    feature_collection: dict[str, Any] | None,
) -> pd.DataFrame:
    """Build a frame from trusted inline MCP data without paths or URLs."""

    if (records is None) == (feature_collection is None):
        raise ToolError("Передайте ровно один источник: records или feature_collection")
    if feature_collection is not None:
        if feature_collection.get("type") != "FeatureCollection":
            raise ToolError("Ожидается GeoJSON FeatureCollection")
        features = feature_collection.get("features") or []
        properties = [
            feature.get("properties") or {}
            for feature in features
            if isinstance(feature, dict)
        ]
        geometries = [
            shape(feature["geometry"]) if feature.get("geometry") else None
            for feature in features
            if isinstance(feature, dict)
        ]
        frame = gpd.GeoDataFrame(
            pd.json_normalize(properties, sep="."),
            geometry=geometries,
            crs="EPSG:4326",
        )
        return _normalize_object_columns(frame)
    return _normalize_object_columns(pd.json_normalize(records or [], sep="."))


def _normalize_object_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep Arrow storage deterministic for nested or heterogeneous values."""

    for name in frame.columns:
        if name == "geometry" or frame[name].dtype != "object":
            continue
        if frame[name].map(lambda value: isinstance(value, (dict, list))).any():
            frame[name] = frame[name].map(
                lambda value: (
                    json.dumps(value, ensure_ascii=False, default=str)
                    if isinstance(value, (dict, list))
                    else value
                )
            )
    return frame
