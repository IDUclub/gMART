"""``file`` SSE event: a link to a generated file, in the GenBuilder frame format.

Pipelines yield it internally as ``{"type": "file", "content": {...}}`` so it is
buffered and replayed like every other event. Controllers put it on the wire as
``event: file`` with the flat descriptor as ``data`` (no ``type`` field).
"""

import json
from typing import Any, Literal

from fastapi.sse import ServerSentEvent
from pydantic import BaseModel


class FileEventContent(BaseModel):
    name: str
    title: str
    role: Literal["result", "input"]
    url: str
    download_url: str | None
    filename: str
    mime_type: str
    source_service: str


def file_sse_event(chunk: dict[str, Any]) -> ServerSentEvent | None:
    """Return the wire frame for an internal ``file`` event, else ``None``."""

    if chunk.get("type") != "file":
        return None
    content = FileEventContent.model_validate(chunk.get("content") or {})
    return ServerSentEvent(
        event="file",
        raw_data=json.dumps(content.model_dump(mode="json"), ensure_ascii=False),
    )
