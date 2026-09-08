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
    workflow_id: str | None = None
    run_config_id: str | None = None
    status: SynapseRunStatus
    events_url: str


class SynapseConfigurationOption(BaseModel):
    id: str
    name: str
    description: str = ""
    is_default: bool = False


class SynapseWorkflowOption(SynapseConfigurationOption):
    display_name: str | None = None
    execution_mode: str | None = None


class SynapseConfigurationOptionsResponse(BaseModel):
    workflows: list[SynapseWorkflowOption] = Field(default_factory=list)
    run_configurations: list[SynapseConfigurationOption] = Field(default_factory=list)
    default_workflow_id: str | None = None
    default_run_config_id: str | None = None


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
