from enum import StrEnum

from pydantic import BaseModel


class ProvisionPlanMode(StrEnum):
    FOUND = "found"
    NEEDS_CLARIFICATION = "needs_clarification"


class ProvisionPlan(BaseModel):
    mode: ProvisionPlanMode
    service_name: str | None = None
    target_population: int | None = None
    clarification_question: str | None = None
