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

from src.agents.common.files.temporary_file_store import FILES_TTL_SECONDS

LAYER_CACHE_TTL_SECONDS = 24 * 60 * 60
COMPLIANCE_CACHE_TTL_SECONDS = FILES_TTL_SECONDS + 10 * 60

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
        compliance_ttl_seconds: int | None = None,
    ) -> None:
        self.root = (directory or default_cache_dir()) / "upload_layers"
        self.ttl_seconds = ttl_seconds
        self.compliance_root = self.root.parent / "compliance"
        self.compliance_ttl_seconds = (
            compliance_ttl_seconds
            if compliance_ttl_seconds is not None
            else int(
                os.getenv(
                    "GMART_COMPLIANCE_CACHE_TTL_SECONDS",
                    str(COMPLIANCE_CACHE_TTL_SECONDS),
                )
            )
        )
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
        """Drop expired uploads and compliance runs by their directory mtime."""
        removed = 0
        for root, ttl in (
            (self.root, self.ttl_seconds),
            (self.compliance_root, self.compliance_ttl_seconds),
        ):
            if not root.exists():
                continue
            deadline = time.time() - ttl
            for directory in root.iterdir():
                try:
                    if (
                        directory.is_symlink()
                        or not directory.is_dir()
                        or directory.stat().st_mtime >= deadline
                    ):
                        continue
                    shutil.rmtree(directory)
                    removed += 1
                except FileNotFoundError:  # removed meanwhile by another worker
                    continue
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
