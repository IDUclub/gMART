from typing import Any, Literal

from geojson_pydantic import FeatureCollection
from pydantic import BaseModel

from src.agents.common.exceptions.sse_exceptions import SseBaseError
from src.agents.schema.restrictions_response import (
    PipelineEventContent,
    PipelineStartedContent,
    ServiceEvent,
)


class ScenarioDataStatus(BaseModel):
    status: Literal[
        "tool_discovery",
        "planning",
        "tool_execution",
        "workspace",
        "response_analysis",
        # The answer-review loop. Every status the service can emit must be listed here:
        # the SSE payload is validated against this model, so an unlisted one does not
        # degrade to an unknown label — it raises mid-stream and the client hangs waiting
        # for a terminal event that never arrives.
        "answer_review",
        "answer_retry",
    ]
    text: str


class ScenarioDataText(BaseModel):
    text: str
    done: bool


class ScenarioDataFeatureCollection(BaseModel):
    name: str
    feature_collection: FeatureCollection


class ScenarioDataToolCall(BaseModel):
    execution_mode: str
    tool_calls: list[Any]
    mcp_source: str | None = None


class ScenarioDataTableColumn(BaseModel):
    key: str
    label: str


class ScenarioDataTable(BaseModel):
    name: str
    title: str
    columns: list[ScenarioDataTableColumn]
    rows: list[dict[str, Any]]


class ScenarioDataResponse(BaseModel):
    type: Literal[
        "status",
        "chunk",
        "feature_collection",
        "table",
        "tool_call",
        "service_event",
        "pipeline_started",
        "token_expired",
        "pipeline_suspended",
        "warning",
        "error",
        "plan_created",
        "mapping_started",
        "mapping_completed",
        "plan_revision_created",
        "step_started",
        "step_completed",
        "artifact_created",
        "validation_started",
        "validation_completed",
        "budget_warning",
        "clarification_required",
        "replanning",
        "pipeline_failed",
    ]
    content: (
        ScenarioDataStatus
        | ScenarioDataText
        | ScenarioDataFeatureCollection
        | ScenarioDataTable
        | ScenarioDataToolCall
        | ServiceEvent
        | PipelineStartedContent
        | PipelineEventContent
        | SseBaseError
        | dict[str, Any]
    )
