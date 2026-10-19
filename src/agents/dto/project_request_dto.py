from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.agents.dto.norms_request_dto import NormsQaRequestDTO


class ProjectInputs(BaseModel):
    """Explicit parameters of a project created from files; the rest is derived.

    Shared by REST, A2A and the orchestrator. Each field omitted is taken from the
    request text or the files; a question is asked only when they settle nothing.
    """

    model_config = ConfigDict(extra="forbid")

    project_name: str | None = Field(default=None, max_length=200)
    scenario_name: str | None = Field(default=None, max_length=200)
    territory_id: int | None = Field(
        default=None,
        gt=0,
        description="Region of the project (a child of the top-level territory); "
        "detected from the boundary when omitted.",
    )
    functional_zone_type_id: int | None = Field(
        default=None,
        gt=0,
        description="Profile of the scenario; derived from the buildings when omitted.",
    )
    existing_objects: Literal["replace_same", "delete_all", "keep"] | None = Field(
        default=None,
        description=(
            "What to do with the base scenario's objects inside the boundary: "
            "replace those of the files' types, delete all, or keep. Asked when "
            "omitted and the boundary has objects."
        ),
    )


class ProjectFromFilesRequestDTO(NormsQaRequestDTO):
    upload_ids: list[str] | None = Field(
        default=None,
        max_length=20,
        description=(
            "Uploads of POST /compliance/uploads. Omit them to answer a pending "
            "question by chat_id or to use the files already uploaded to the chat."
        ),
    )
    inputs: ProjectInputs = Field(default_factory=ProjectInputs)
