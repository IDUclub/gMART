"""Short-lived generated files served to their author under ``/files``.

Files live on the container's local disk and do not survive a restart. Each file
has a sidecar with its owner and expiry; ``OwnedFilesApp`` checks both before
delegating the actual transfer to ``StaticFiles``.
"""

from __future__ import annotations

import json
import re
import secrets
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, quote

from loguru import logger
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.types import Message, Receive, Scope, Send

from src.common.service_auth import user_id_from_jwt

FILES_MOUNT_PATH = "/files"
FILES_TTL_SECONDS = 60 * 60
FILES_DIR = Path(tempfile.gettempdir()) / "gmart-files"

_SLOT = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_FILE_ID = re.compile(r"^[A-Za-z0-9_-]{32,64}$")


@dataclass(frozen=True)
class StoredFile:
    file_id: str
    url: str
    download_url: str


class TemporaryFileStore:
    """Write generated files and build their public links."""

    def __init__(
        self,
        directory: Path = FILES_DIR,
        *,
        public_base_url: str | None = None,
        ttl_seconds: int = FILES_TTL_SECONDS,
    ) -> None:
        self.data_dir = directory / "data"
        self.meta_dir = directory / "meta"
        self.public_base_url = (public_base_url or "").rstrip("/")
        self.ttl_seconds = ttl_seconds
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        slot: str,
        content: bytes,
        *,
        owner: str,
        filename: str,
        mime_type: str,
    ) -> StoredFile:
        if not _SLOT.match(slot):
            raise ValueError(f"Invalid file slot: {slot!r}")
        if not owner:
            raise ValueError("A generated file requires an owner")
        self.purge_expired()
        file_id = secrets.token_urlsafe(32)
        (self.data_dir / slot).mkdir(parents=True, exist_ok=True)
        (self.meta_dir / slot).mkdir(parents=True, exist_ok=True)
        (self.data_dir / slot / file_id).write_bytes(content)
        self._meta_path(slot, file_id).write_text(
            json.dumps(
                {
                    "owner": owner,
                    "filename": filename,
                    "mime_type": mime_type,
                    "expires_at": time.time() + self.ttl_seconds,
                }
            ),
            encoding="utf-8",
        )
        url = f"{self.public_base_url}{FILES_MOUNT_PATH}/{slot}/{file_id}"
        return StoredFile(file_id=file_id, url=url, download_url=f"{url}?download=1")

    def metadata(self, slot: str, file_id: str) -> dict | None:
        """Return live metadata; an expired file is deleted on the way out."""

        if not _SLOT.match(slot) or not _FILE_ID.match(file_id):
            return None
        try:
            meta = json.loads(self._meta_path(slot, file_id).read_text("utf-8"))
        except (OSError, ValueError):
            return None
        if meta.get("expires_at", 0) <= time.time():
            self._delete(slot, file_id)
            return None
        return meta

    def purge_expired(self) -> int:
        removed = 0
        now = time.time()
        for meta_path in self.meta_dir.glob("*/*.json"):
            try:
                expires_at = json.loads(meta_path.read_text("utf-8"))["expires_at"]
            except (OSError, ValueError, KeyError):
                expires_at = 0
            if expires_at <= now:
                self._delete(meta_path.parent.name, meta_path.stem)
                removed += 1
        return removed

    def _meta_path(self, slot: str, file_id: str) -> Path:
        return self.meta_dir / slot / f"{file_id}.json"

    def _delete(self, slot: str, file_id: str) -> None:
        (self.data_dir / slot / file_id).unlink(missing_ok=True)
        self._meta_path(slot, file_id).unlink(missing_ok=True)


class OwnedFilesApp:
    """``StaticFiles`` guarded by a Bearer token, file ownership and expiry."""

    def __init__(self, store: TemporaryFileStore) -> None:
        self.store = store
        self.static = StaticFiles(directory=store.data_dir)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        if scope["method"] not in {"GET", "HEAD"}:
            await JSONResponse({"detail": "Method not allowed"}, 405)(
                scope, receive, send
            )
            return
        owner = self._requester(scope)
        if owner is None:
            await JSONResponse({"detail": "Authorization header missing"}, 401)(
                scope, receive, send
            )
            return
        segments = self.static.get_path(scope).split("/")
        meta = self.store.metadata(*segments) if len(segments) == 2 else None
        # Foreign, expired and unknown files are indistinguishable to the caller.
        if meta is None or meta.get("owner") != owner:
            await JSONResponse({"detail": "File not found"}, 404)(scope, receive, send)
            return

        query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        disposition = "attachment" if query.get("download") == ["1"] else "inline"
        filename = quote(meta.get("filename") or segments[1])
        headers = [
            (b"content-type", f"{meta['mime_type']}; charset=utf-8".encode()),
            (
                b"content-disposition",
                f"{disposition}; filename*=UTF-8''{filename}".encode(),
            ),
            (b"cache-control", b"no-store"),
        ]

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                replaced = {name for name, _ in headers}
                message["headers"] = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() not in replaced
                ] + headers
            await send(message)

        await self.static(scope, receive, send_with_headers)

    @staticmethod
    def _requester(scope: Scope) -> str | None:
        authorization = dict(scope.get("headers") or []).get(b"authorization", b"")
        scheme, _, token = authorization.decode("latin-1").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return None
        try:
            return user_id_from_jwt(token.strip())
        except ValueError:
            logger.info("Rejected file request with an unreadable Bearer token")
            return None
