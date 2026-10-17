from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agents.dto.restriction_request_dto import RestrictionRequestDTO


class ComplianceRequestDTO(RestrictionRequestDTO):
    """
    Compliance check request; the scenario is optional when user files are checked.
    Attributes:
        scenario_id (int | None): Scenario ID from Urban API. With ``upload_ids`` the
            files overlay the scenario; without it the files are checked on their own.
        upload_ids (list[str]): Uploads from ``POST /compliance/uploads`` to check
            instead of the scenario's objects.
    """

    scenario_id: int | None = Field(
        default=None,
        examples=[772],
        description=(
            "Scenario ID from Urban API. Required unless upload_ids are given; with "
            "upload_ids the uploaded objects replace the scenario's objects of the same "
            "kind inside the design's boundary."
        ),
    )
    upload_ids: list[str] = Field(
        default_factory=list,
        examples=[["5b1f0c3e-6a77-4d0e-9e43-1f3b1d6b2c11"]],
        description="Uploads (POST /compliance/uploads) to check.",
    )

    @model_validator(mode="after")
    def scenario_or_files(self) -> "ComplianceRequestDTO":
        if self.scenario_id is None and not self.upload_ids:
            raise ValueError("scenario_id or upload_ids is required")
        return self


class UploadRolesDTO(BaseModel):
    """Roles of uploaded layers by layer name."""

    model_config = ConfigDict(extra="forbid")

    roles: dict[str, Literal["project", "context", "border"]] = Field(
        examples=[{"336195_osm_surroundings": "context"}],
        description=(
            "project — checked objects of the design; context — surroundings, used "
            "only as sources and neighbours of distance rules; border — the design's "
            "boundary."
        ),
    )
