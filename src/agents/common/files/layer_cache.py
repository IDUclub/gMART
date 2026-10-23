"""Parsed layers of user uploads, kept on the container's disk instead of Redis.

Redis holds an upload's record — owner, roles, mapping answers and, per layer, the
name of its file here — so reading or updating a record never moves geometry. The
files live under ``GMART_CACHE_DIR`` (the system temp directory by default), one
directory per upload, and share the record's lifetime: ``purge_expired`` drops the
directories older than it. Every worker of the container sees the same files; a
recreated container does not, and its uploads read as expired.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

LAYER_CACHE_TTL_SECONDS = 24 * 60 * 60

_UPLOAD_ID = re.compile(r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$")
_LAYER_FILE = re.compile(
    r"^(?P<upload>[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})/\d{1,6}\.geojson$"
)


class LayerMissing(LookupError):
    """The layer file is gone: expired, purged or written by another container."""


def default_cache_dir() -> Path:
    configured = os.getenv("GMART_CACHE_DIR")
    base = Path(configured) if configured else Path(tempfile.gettempdir()) / "gmart"
    return base / "cache"


class LayerCache:
    """Write, read and expire the GeoJSON layers of uploads."""

    def __init__(
        self,
        directory: Path | None = None,
        *,
        ttl_seconds: int = LAYER_CACHE_TTL_SECONDS,
    ) -> None:
        self.root = (directory or default_cache_dir()) / "upload_layers"
        self.ttl_seconds = ttl_seconds
        self.root.mkdir(parents=True, exist_ok=True)

    async def write(
        self, upload_id: str, index: int, feature_collection: dict[str, Any]
    ) -> str:
        """Store one layer; returns the file name a record keeps."""
        if not _UPLOAD_ID.match(upload_id):
            raise ValueError(f"Invalid upload id: {upload_id!r}")
        file = f"{upload_id}/{int(index)}.geojson"
        await asyncio.to_thread(self._write, self.root / file, feature_collection)
        return file

    async def read(self, file: str) -> dict[str, Any]:
        path = self._path(file)
        try:
            text = await asyncio.to_thread(path.read_text, "utf-8")
        except FileNotFoundError as exc:
            raise LayerMissing(file) from exc
        return json.loads(text)

    def exists(self, file: str) -> bool:
        try:
            return self._path(file).is_file()
        except LayerMissing:
            return False

    def delete(self, upload_id: str) -> None:
        if _UPLOAD_ID.match(upload_id):
            shutil.rmtree(self.root / upload_id, ignore_errors=True)

    def purge_expired(self) -> int:
        """Drop upload directories older than the TTL; returns how many."""
        removed = 0
        deadline = time.time() - self.ttl_seconds
        for directory in self.root.iterdir():
            try:
                expired = directory.is_dir() and directory.stat().st_mtime < deadline
            except OSError:  # removed meanwhile by another worker
                continue
            if expired:
                shutil.rmtree(directory, ignore_errors=True)
                removed += 1
        return removed

    def _path(self, file: str) -> Path:
        # File names come from Redis records: only names this cache issues resolve.
        if not _LAYER_FILE.match(str(file)):
            raise LayerMissing(file)
        return self.root / file

    @staticmethod
    def _write(path: Path, feature_collection: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        partial.write_text(
            json.dumps(feature_collection, ensure_ascii=False), encoding="utf-8"
        )
        # A reader sees either no file or the whole layer.
        os.replace(partial, path)
