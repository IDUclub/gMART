from typing import Any, Literal

from geojson_pydantic import FeatureCollection
from pydantic import BaseModel

from src.agents.common.exceptions.sse_exceptions import SseBaseError
from src.agents.services.service_entities.compliance import CheckPlan, ComplianceResult


class StatusResponse(BaseModel):
    """
    Class for status response.
    Attributes:
        status: Stage name.
        text (str): Status message.
    """

    status: Literal[
        "norm_retrieval",
        "data_retrievement",
        "plan_explanation",
        "buffer_creation",
        "restriction_formation",
        "context_preparation",
        "check_plan_validation",
        "requirements_resolution",
        "template_execution",
        "verdict_aggregation",
        "compliance_result_analysis",
    ]
    text: str


class TextResponse(BaseModel):
    text: str
    done: bool


class FeatureCollectionResponse(BaseModel):
    name: str
    feature_collection: FeatureCollection


class ChatCreatedEvent(BaseModel):
    storage_event_type: Literal["chat_created"]
    chat_id: str
    chat_title: str


class ServiceEvent(BaseModel):
    event_type: Literal["storage_event"]
    event: ChatCreatedEvent


class PipelineStartedContent(BaseModel):
    """Emitted once at the start of every pipeline run."""

    request_id: str


class PipelineEventContent(BaseModel):
    """
    Generic pipeline notification that carries a request_id and a
    human-readable message.  Used for ``token_expired`` and
    ``pipeline_suspended`` events.
    """

    request_id: str
    message: str


class ToolCallContent(BaseModel):
    """Describes MCP tool calls executed during the pipeline step."""

    execution_mode: str
    tool_calls: list[Any]
    mcp_source: str | None = None


class CheckPlanEventContent(BaseModel):
    restriction_id: str
    plan: CheckPlan


class RequirementResolutionEventContent(BaseModel):
    restriction_id: str
    effective_requirements: dict[str, Any]
    resolved_requirements: list[dict[str, Any]]
    missing_requirements: list[str]


class ComplianceSummaryEventContent(BaseModel):
    request_id: str
    total_norms: int
    violated_norms: int
    passed_norms: int
    unverifiable_norms: int
    unsupported_norms: int
    not_applicable_norms: int
    partial_norms: int
    results: list[ComplianceResult]


class ComplianceProgressEventContent(BaseModel):
    total_norms: int
    completed_norms: int
    pending_norms: int
    passed_norms: int
    violated_norms: int
    unverifiable_norms: int
    unsupported_norms: int


class RestrictionsResponse(BaseModel):
    type: Literal[
        "status",
        "chunk",
        "feature_collection",
        "error",
        "service_event",
        "pipeline_started",
        "token_expired",
        "pipeline_suspended",
        "tool_call",
        "check_plan",
        "requirement_resolution",
        "compliance_result",
        "compliance_progress",
        "compliance_summary",
    ]
    content: (
        StatusResponse
        | TextResponse
        | FeatureCollectionResponse
        | SseBaseError
        | ServiceEvent
        | PipelineStartedContent
        | PipelineEventContent
        | ToolCallContent
        | CheckPlanEventContent
        | RequirementResolutionEventContent
        | ComplianceResult
        | ComplianceProgressEventContent
        | ComplianceSummaryEventContent
    )
