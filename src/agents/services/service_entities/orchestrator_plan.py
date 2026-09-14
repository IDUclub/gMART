from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Upper bound on the number of agent steps in a single orchestration plan.
MAX_PLAN_STEPS = 3
MAX_ANALYSIS_STEPS = 12


class OrchestratorAgent(StrEnum):
    RESTRICTION = "restriction"
    COMPLIANCE = "compliance"
    PROVISION = "provision"
    SCENARIO_DATA = "scenario_data"
    DOCUMENTS = "documents"
    NORMS = "norms"
    GENPLANNER = "genplanner"
    GENBUILDER = "genbuilder"
    PZZ = "pzz"


class OrchestratorPlanMode(StrEnum):
    EXECUTE = "execute"
    NEEDS_CLARIFICATION = "needs_clarification"


class MetricValue(BaseModel):
    artifact_id: str
    row: int = Field(ge=0)
    column: str


class PopulationAdjustment(BaseModel):
    base: MetricValue
    multiplier: Decimal = Field(gt=0)


class EntitySelection(BaseModel):
    subject: str = Field(min_length=1)
    kind: Literal["services", "physical_objects"]


class OrchestratorStep(BaseModel):
    agent: OrchestratorAgent
    task: str
    scenario_id: int | None = Field(default=None, gt=0)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    population_adjustment: PopulationAdjustment | None = None
    requirement_id: str | None = None
    support: bool = False
    entity_selection: EntitySelection | None = None

    @field_validator("task")
    @classmethod
    def task_must_be_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("step task must not be blank")
        return value


class OrchestratorPlan(BaseModel):
    mode: OrchestratorPlanMode
    analytical: bool = False
    steps: list[OrchestratorStep] = Field(default_factory=list)
    clarification_question: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "OrchestratorPlan":
        if (
            self.mode == OrchestratorPlanMode.EXECUTE
            and not self.steps
            and not self.analytical
        ):
            raise ValueError("execute plan must contain at least one step")
        if self.mode == OrchestratorPlanMode.NEEDS_CLARIFICATION and self.steps:
            raise ValueError("needs_clarification plan must not contain steps")
        return self


class NeededInput(BaseModel):
    missing: str
    reason: str
    question: str
    example: str = ""
    owner: Literal["user", "service", "budget"] = "user"


class ArtifactSlice(BaseModel):
    artifact_id: str
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=50)


class MetricComparison(BaseModel):
    name: str
    unit: str
    before: MetricValue
    after: MetricValue


class AnalysisReview(BaseModel):
    action: Literal["continue", "inspect", "complete", "blocked"]
    steps: list[OrchestratorStep] = Field(
        default_factory=list, max_length=MAX_ANALYSIS_STEPS
    )
    inspect: list[ArtifactSlice] = Field(default_factory=list, max_length=4)
    answer: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    missing: list[NeededInput] = Field(default_factory=list, max_length=8)
    hypotheses: list[str] = Field(default_factory=list, max_length=8)
    comparisons: list[MetricComparison] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def consistent(self):
        if self.action == "continue" and not self.steps:
            raise ValueError("Continue requires a concrete next step")
        if self.action == "inspect" and not self.inspect:
            raise ValueError("Inspect requires artifact references")
        if self.action == "blocked" and not self.missing:
            raise ValueError("Blocked requires actionable missing information")
        if self.action == "complete" and not self.answer.strip():
            raise ValueError("Complete requires a supported final answer")
        return self
