"""Versioned plan contracts for the linear scenario-data workflow."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MappingDirection(StrEnum):
    NAME_TO_ID = "name_to_id"
    ID_TO_NAME = "id_to_name"


class MappingNeed(BaseModel):
    domain: str = Field(min_length=1)
    direction: MappingDirection
    values: list[str | int] = Field(default_factory=list)
    required: bool = True


class RequiredOutput(BaseModel):
    answer: bool = True
    tables: list[str] = Field(default_factory=list)
    layers: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    completeness: str = Field(default="verified", pattern="^(verified|partial)$")


class DataRequirement(BaseModel):
    requirement_id: str = Field(pattern=r"^[a-z][a-z0-9_\-]{0,63}$")
    description: str = Field(min_length=1)
    mapping_needs: list[MappingNeed] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)


class AcquisitionPlan(BaseModel):
    """Logical plan: what evidence is required, without invented tool arguments."""

    objective: str = Field(min_length=1)
    clarification: str | None = None
    requirements: list[DataRequirement] = Field(default_factory=list)
    required_output: RequiredOutput = Field(default_factory=RequiredOutput)


class PlanStepKind(StrEnum):
    URBAN_TOOL = "urban_tool"
    WORKSPACE = "workspace"


class PlanStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    step_id: str = Field(pattern=r"^[a-z][a-z0-9_\-]{0,63}$")
    kind: PlanStepKind = PlanStepKind.URBAN_TOOL
    purpose: str = Field(min_length=1)
    group: str | None = None
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    satisfies: list[str] = Field(default_factory=list)
    expected_output: str = Field(min_length=1)
    parallel_group: str | None = None
    layer_name: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "PlanStep":
        if self.kind == PlanStepKind.URBAN_TOOL and not self.group:
            raise ValueError("urban_tool step requires group")
        if self.kind == PlanStepKind.WORKSPACE and self.group is not None:
            raise ValueError("workspace step must not define Urban MCP group")
        return self


class ExecutionPlanRevision(BaseModel):
    """Immutable executable revision. Replanning creates a new object."""

    model_config = ConfigDict(frozen=True)

    revision: int = Field(ge=1)
    reason: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    steps: list[PlanStep] = Field(default_factory=list)
    required_output: RequiredOutput = Field(default_factory=RequiredOutput)

    @model_validator(mode="after")
    def validate_graph(self) -> "ExecutionPlanRevision":
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("plan step_id values must be unique")
        known: set[str] = set()
        plan_ids = set(ids)
        for step in self.steps:
            later = {
                dependency
                for dependency in step.depends_on
                if dependency in plan_ids and dependency not in known
            }
            if later:
                raise ValueError(
                    f"step {step.step_id} depends on later steps: {sorted(later)}"
                )
            known.add(step.step_id)
        return self


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionRecord(BaseModel):
    step_id: str
    revision: int
    status: StepStatus
    call_fingerprint: str | None = None
    observation_index: int | None = None
    error: str | None = None


class ExecutionLedger(BaseModel):
    records: list[ExecutionRecord] = Field(default_factory=list)
    urban_calls: int = 0
    workspace_calls: int = 0
    replans: int = 0

    @property
    def completed_step_ids(self) -> set[str]:
        return {
            record.step_id
            for record in self.records
            if record.status == StepStatus.COMPLETED
        }
