from enum import StrEnum

from pydantic import BaseModel, model_validator


class ProvisionPlanMode(StrEnum):
    EFFECTS = "effects"
    PROVISION = "provision"
    SUMMARY = "summary"
    LIST_SERVICES = "list_services"
    NEEDS_CLARIFICATION = "needs_clarification"


class ProvisionPlan(BaseModel):
    mode: ProvisionPlanMode
    service_name: str | None = None
    service_names: list[str] = []
    layer_service_names: list[str] = []
    target_population: int | None = None
    clarification_question: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_mode(cls, data):
        # Redis checkpoints written before intent routing store mode="found";
        # it always meant the single-service effects pipeline.
        if isinstance(data, dict) and data.get("mode") == "found":
            data = {**data, "mode": ProvisionPlanMode.EFFECTS.value}
        return data
