from typing import Any, Literal

from pydantic import BaseModel, Field

SynapseRunStatus = Literal[
    "starting", "running", "done", "failed", "cancelled", "start_unknown"
]


class SynapseRunResponse(BaseModel):
    request_id: str
    chat_id: str | None = None
    synapse_project_id: str | None = None
    run_id: str | None = None
    status: SynapseRunStatus
    events_url: str


class SynapseRunStateResponse(SynapseRunResponse):
    last_event_id: str | None = None
    last_stream_id: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class SynapseEvent(BaseModel):
    type: Literal["synapse_event"] = "synapse_event"
    source_type: str
    source_event_id: str
    stream_id: str | None = None
    request_id: str
    synapse_project_id: str
    run_id: str | None = None
    timestamp: str | None = None
    content: dict[str, Any] = Field(default_factory=dict)
