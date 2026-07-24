from enum import StrEnum

from pydantic import BaseModel, model_validator

# Upper bound on the number of agent steps in a single orchestration plan.
MAX_PLAN_STEPS = 3


class OrchestratorAgent(StrEnum):
    RESTRICTION = "restriction"
    PROVISION = "provision"
    DOCUMENTS = "documents"
    NORMS = "norms"
    URBAN_DATA = "urban_data"


class OrchestratorPlanMode(StrEnum):
    EXECUTE = "execute"
    NEEDS_CLARIFICATION = "needs_clarification"


class OrchestratorStep(BaseModel):
    agent: OrchestratorAgent
    task: str


class OrchestratorPlan(BaseModel):
    mode: OrchestratorPlanMode
    steps: list[OrchestratorStep] = []
    clarification_question: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "OrchestratorPlan":
        if self.mode == OrchestratorPlanMode.EXECUTE and not self.steps:
            raise ValueError("execute plan must contain at least one step")
        if self.mode == OrchestratorPlanMode.NEEDS_CLARIFICATION and self.steps:
            raise ValueError("needs_clarification plan must not contain steps")
        return self
