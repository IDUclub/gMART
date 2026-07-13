from typing import Any, Literal

from pydantic import BaseModel

from src.agents.common.exceptions.sse_exceptions import SseBaseError
from src.agents.schema.dvd_response import (
    DvdTextResponse,
    PipelineStartedContent,
    ServiceEvent,
    WarningContent,
)

StepStatus = Literal["completed", "failed", "suspended"]


class OrchestratorStatusResponse(BaseModel):
    """Status update for an orchestrator-level stage."""

    status: Literal["planning"]
    text: str


class PlanStepInfo(BaseModel):
    """One planned agent step as announced in the ``plan`` event."""

    step: int
    agent: str
    agent_title: str
    task: str


class PlanContent(BaseModel):
    """The full orchestration plan announced before execution starts."""

    steps: list[PlanStepInfo]


class StepStartedContent(BaseModel):
    """
    Emitted when a step's sub-pipeline starts.
    Attributes:
        step (int): 1-based step number.
        agent (str): Agent key from the catalogue.
        step_request_id (str): The sub-pipeline's own request ID — use it with
            ``POST /pipelines/{step_request_id}/token`` to refresh an expired token.
        task (str): The sub-task the agent was given.
    """

    step: int
    agent: str
    step_request_id: str
    task: str


class StepEventContent(BaseModel):
    """
    An event of a sub-agent forwarded verbatim inside the orchestrator envelope.
    Attributes:
        step (int): 1-based step number the event belongs to.
        agent (str): Agent key from the catalogue.
        event (dict): The original sub-agent SSE event (``{"type", "content"}``)
            in the agent's native format.
    """

    step: int
    agent: str
    event: dict[str, Any]


class StepFinishedContent(BaseModel):
    """Terminal event of a single step with its text digest."""

    step: int
    agent: str
    status: StepStatus
    summary: str


class ClarificationContent(BaseModel):
    """The planner could not route the request — a question for the user."""

    question: str


class OrchestratorSummaryStep(BaseModel):
    step: int
    agent: str
    task: str
    status: StepStatus | Literal["skipped"]
    summary: str


class OrchestratorFinalContent(BaseModel):
    """Structured per-step summary emitted once at the end of the run."""

    steps: list[OrchestratorSummaryStep]


class OrchestratorResponse(BaseModel):
    """SSE event envelope for the orchestrator pipeline."""

    type: Literal[
        "pipeline_started",
        "service_event",
        "status",
        "plan",
        "step_started",
        "step_event",
        "step_finished",
        "clarification",
        "orchestrator_final",
        "chunk",
        "warning",
        "error",
    ]
    content: (
        PipelineStartedContent
        | ServiceEvent
        | OrchestratorStatusResponse
        | PlanContent
        | StepStartedContent
        | StepEventContent
        | StepFinishedContent
        | ClarificationContent
        | OrchestratorFinalContent
        | DvdTextResponse
        | WarningContent
        | SseBaseError
    )
