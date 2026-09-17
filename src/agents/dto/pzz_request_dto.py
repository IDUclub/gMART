from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.agents.dto.norms_request_dto import NormsQaRequestDTO


class PzzInputs(BaseModel):
    """Trusted input references, shared by REST, A2A and the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    mode: (
        Literal["pzz_check", "classify_only", "building_pzz_check", "scenario"] | None
    ) = None
    cadastral_geojson: dict[str, Any] | None = None
    pzz_zones_geojson: dict[str, Any] | None = None
    cadastral_upload_id: str | None = None
    buildings_upload_id: str | None = None
    pzz_zones_upload_id: str | None = None
    descriptions_upload_id: str | None = None
    labels_upload_id: str | None = None
    classifier_upload_id: str | None = None
    confirmed_zone_map: dict[str, str] | None = None
    cadastral_vri_col: str | None = None
    pzz_zone_code_col: str | None = None
    pzz_zone_name_col: str | None = None
    year: int | None = None
    source: Literal["User", "OSM", "PZZ"] | None = None
    physical_object_type_id: int = Field(default=4, gt=0)
    group_by: Literal["zone", "object"] = "zone"
    priority: int = Field(default=1, ge=1, le=10)
    force_recompute: bool = False


class PzzRequestDTO(NormsQaRequestDTO):
    inputs: PzzInputs = Field(default_factory=PzzInputs)
