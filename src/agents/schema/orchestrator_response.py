from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from src.agents.common.exceptions.sse_exceptions import SseBaseError
from src.agents.schema.provision_response import (
    FeatureCollectionResponse,
    PipelineEventContent,
    PipelineStartedContent,
    ProvisionStatusResponse,
    ServiceEvent,
    TextResponse,
    ToolCallContent,
)
from src.agents.schema.restrictions_response import StatusResponse


class OrchestratorRoutingContent(BaseModel):
    """
    Content for orchestrator-specific routing and classification events.
    Attributes:
        text (str): Human-readable routing or classification status message.
    """

    model_config = ConfigDict(extra="forbid")

    text: str


class CritiqueContent(BaseModel):
    """
    Content for critic-agent evaluation events.

    Emitted once per sub-pipeline after its final response text is produced.
    When ``quality == "poor"`` and ``retried == True`` the orchestrator has
    already re-run the sub-pipeline with a refined query; the client can use
    ``feedback`` to explain the quality issue to the user.

    Attributes:
        agent (str): Name of the evaluated sub-pipeline
            (e.g. ``"restriction-creation-agent"``).
        quality (str): ``"good"`` or ``"poor"``.
        feedback (str): Human-readable critique in Russian.
        retried (bool): Whether an automatic retry was triggered.
    """

    model_config = ConfigDict(extra="forbid")

    agent: str
    quality: Literal["good", "poor"]
    feedback: str
    retried: bool


class OrchestratorResponse(BaseModel):
    """
    Unified SSE event schema for the orchestrator REST endpoint.

    Passes through events from restriction and provision sub-pipelines verbatim,
    and adds orchestrator-specific events for routing, classification, and
    critic evaluation.

    Event types:
        routing           — orchestrator routing/classification status
        critique          — critic-agent quality assessment of a sub-pipeline result
        status            — pipeline step status from restriction or provision sub-agent
        chunk             — LLM response text chunk (streaming)
        feature_collection— GeoJSON layer produced by a sub-pipeline
        error             — pipeline error with traceback
        service_event     — side-effect event (e.g. chat created in Chat Storage)
        pipeline_started  — pipeline execution started; carries request_id
        token_expired     — auth token expired; client should refresh and reconnect
        pipeline_suspended— pipeline suspended due to token refresh timeout
        tool_call         — MCP tool call metadata

    Attributes:
        type: Event discriminator literal.
        content: Typed payload; depends on ``type``.
    """

    type: Literal[
        "routing",
        "critique",
        "status",
        "chunk",
        "feature_collection",
        "error",
        "service_event",
        "pipeline_started",
        "token_expired",
        "pipeline_suspended",
        "tool_call",
    ]
    content: (
        OrchestratorRoutingContent
        | CritiqueContent
        | StatusResponse
        | ProvisionStatusResponse
        | TextResponse
        | FeatureCollectionResponse
        | SseBaseError
        | ServiceEvent
        | PipelineStartedContent
        | PipelineEventContent
        | ToolCallContent
        | dict[str, Any]
    )
