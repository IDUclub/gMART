"""Source snapshots come from retrieval responses, never from generated citations."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceEvidence(BaseModel):
    system: Literal["documents", "norms"]
    sources: list[dict[str, Any]] = Field(min_length=1)


def source_event(system, records):
    sources = [dict(record) for record in records if record.get("id")]
    if not sources:
        return None
    return {
        "type": "source_evidence",
        "content": SourceEvidence(system=system, sources=sources).model_dump(),
    }
