from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

# Upper bound on the number of agent steps in a single orchestration plan.
MAX_PLAN_STEPS = 3


class OrchestratorAgent(StrEnum):
    RESTRICTION = "restriction"
    COMPLIANCE = "compliance"
    PROVISION = "provision"
    SCENARIO_DATA = "scenario_data"
    DOCUMENTS = "documents"
    NORMS = "norms"


class OrchestratorPlanMode(StrEnum):
    EXECUTE = "execute"
    NEEDS_CLARIFICATION = "needs_clarification"


class OrchestratorStep(BaseModel):
    agent: OrchestratorAgent
    task: str

    @field_validator("task")
    @classmethod
    def task_must_be_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("step task must not be blank")
        return value


class OrchestratorPlan(BaseModel):
    mode: OrchestratorPlanMode
    steps: list[OrchestratorStep] = Field(default_factory=list)
    clarification_question: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "OrchestratorPlan":
        if self.mode == OrchestratorPlanMode.EXECUTE and not self.steps:
            raise ValueError("execute plan must contain at least one step")
        if self.mode == OrchestratorPlanMode.NEEDS_CLARIFICATION and self.steps:
            raise ValueError("needs_clarification plan must not contain steps")
        return self
