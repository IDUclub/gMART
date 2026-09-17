from typing import Any, Literal

from pydantic import BaseModel


class PzzResponse(BaseModel):
    """The common gMART SSE envelope, including structured PZZ reports."""

    type: Literal[
        "pipeline_started",
        "service_event",
        "status",
        "chunk",
        "tool_call",
        "warning",
        "error",
        "clarification",
        "object_zone_fit",
        "classify_summary",
        "feature_collection",
    ]
    content: dict[str, Any]
