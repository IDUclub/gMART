"""Versioned executable-norm contract shared by the compliance pipeline.

The models in this module deliberately accept data, not executable expressions.  A
``CheckPlan`` is therefore safe to persist and replay after the template registry
has validated its ``params`` against the exact template version.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


RoleName = Annotated[
    str, Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
]


class EntityType(StrEnum):
    SERVICE = "service"
    PHYSICAL_OBJECT = "physical_object"
    FUNCTIONAL_ZONE = "functional_zone"


class LayerRequirement(StrictModel):
    role: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    entity: str = Field(min_length=1, max_length=200)
    entity_type: EntityType
    geometry_types: list[
        Literal[
            "Point",
            "MultiPoint",
            "LineString",
            "MultiLineString",
            "Polygon",
            "MultiPolygon",
        ]
    ] = Field(default_factory=list, max_length=6)
    required: bool = True


class AttributeCandidate(StrictModel):
    field: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-zА-Яа-яЁё0-9_.:-]+$",
    )
    unit: str = Field(min_length=1, max_length=32)
    derive: Literal["height_to_floors_v1"] | None = None
    quality: Literal["direct", "derived"]

    @model_validator(mode="after")
    def derivation_matches_quality(self) -> "AttributeCandidate":
        if (self.derive is None) != (self.quality == "direct"):
            raise ValueError("derive is required exactly for derived candidates")
        return self


class AttributeRequirement(StrictModel):
    role: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    on: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    required: bool = True
    accepts: list[AttributeCandidate] = Field(min_length=1, max_length=12)
    min_fill_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class DeclaredRequirements(StrictModel):
    layers: list[LayerRequirement] = Field(default_factory=list, max_length=16)
    attributes: list[AttributeRequirement] = Field(default_factory=list, max_length=24)

    @model_validator(mode="after")
    def roles_are_unique_and_referential(self) -> "DeclaredRequirements":
        layer_roles = [item.role for item in self.layers]
        attribute_roles = [item.role for item in self.attributes]
        if len(layer_roles) != len(set(layer_roles)):
            raise ValueError("layer requirement roles must be unique")
        if len(attribute_roles) != len(set(attribute_roles)):
            raise ValueError("attribute requirement roles must be unique")
        unknown = sorted({item.on for item in self.attributes} - set(layer_roles))
        if unknown:
            raise ValueError(
                f"attribute requirements reference unknown layer roles: {unknown}"
            )
        return self


class CheckPlanSource(StrictModel):
    restriction_id: str = Field(min_length=1, max_length=128)
    document_name: str | None = Field(default=None, max_length=300)
    clause_number: str | None = Field(default=None, max_length=100)
    extraction_text: str | None = Field(default=None, max_length=8000)


class CheckPlan(StrictModel):
    schema_version: Literal["1.0"]
    template: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    template_version: int = Field(ge=1, le=1000)
    params: dict[str, Any]
    declared_requirements: DeclaredRequirements | None = None
    source: CheckPlanSource
    planner_status: Literal["auto", "reviewed", "unsupported"]


class DistanceFromSourceParams(StrictModel):
    source_layer: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    targets: list[RoleName] = Field(
        min_length=1,
        max_length=16,
    )
    geometry_mode: Literal["buffered", "source_geometry"]
    predicate: Literal["intersects", "within", "contains"]
    violation_when: Literal["matched", "not_matched"]
    result_mode: Literal["violated", "passed", "both"] = "both"
    distance_m: float | None = Field(default=None, gt=0, le=100_000)

    @model_validator(mode="after")
    def buffered_mode_requires_distance(self) -> "DistanceFromSourceParams":
        if self.geometry_mode == "buffered" and self.distance_m is None:
            raise ValueError("distance_m is required for buffered geometry")
        if self.geometry_mode == "source_geometry" and self.distance_m is not None:
            raise ValueError("distance_m is forbidden for source_geometry")
        if len(self.targets) != len(set(self.targets)):
            raise ValueError("targets must be unique")
        return self


class DistanceBand(StrictModel):
    min: float = Field(ge=-1_000_000, le=1_000_000)
    max: float | None = Field(default=None, ge=-1_000_000, le=1_000_000)
    distance_m: float = Field(gt=0, le=100_000)

    @model_validator(mode="after")
    def valid_bounds(self) -> "DistanceBand":
        if self.max is not None and self.max < self.min:
            raise ValueError("band max must be greater than or equal to min")
        return self


class DistanceTableParams(StrictModel):
    source_layer: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    attribute_role: str = Field(
        min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$"
    )
    bands: list[DistanceBand] = Field(min_length=1, max_length=50)
    targets: list[RoleName] = Field(min_length=1, max_length=16)
    predicate: Literal["intersects", "within", "contains"] = "intersects"
    violation_when: Literal["matched", "not_matched"] = "matched"
    result_mode: Literal["violated", "passed", "both"] = "both"
    null_policy: Literal["unchecked"] = "unchecked"
    out_of_range_policy: Literal["unchecked"] = "unchecked"

    @model_validator(mode="after")
    def bands_are_ordered_and_unambiguous(self) -> "DistanceTableParams":
        previous_max: float | None = None
        for index, band in enumerate(self.bands):
            if index and previous_max is None:
                raise ValueError("only the last band may have max=null")
            if previous_max is not None and band.min <= previous_max:
                raise ValueError("bands must be ordered and must not overlap")
            previous_max = band.max
        return self


class PresenceWithinParams(StrictModel):
    objects_layer: str = Field(
        min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$"
    )
    required_neighbor_layers: list[RoleName] = Field(min_length=1, max_length=16)
    distance_m: float = Field(gt=0, le=100_000)
    minimum_neighbors: int = Field(default=1, ge=1, le=1000)
    result_mode: Literal["violated", "passed", "both"] = "both"


class ConstantThreshold(StrictModel):
    kind: Literal["constant"]
    value: float = Field(ge=-1_000_000_000, le=1_000_000_000)
    unit: str = Field(min_length=1, max_length=32)


class ZoneAttributeThreshold(StrictModel):
    kind: Literal["attribute_role"]
    role: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")


class ZonalAttributeThresholdParams(StrictModel):
    objects_layer: str = Field(
        min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$"
    )
    zones_layer: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    attribute_role: str = Field(
        min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$"
    )
    operator: Literal["<", "<=", ">", ">=", "=="]
    threshold_source: ConstantThreshold | ZoneAttributeThreshold
    join_predicate: Literal["intersects", "within", "contains"] = "intersects"
    multiple_zone_policy: Literal["strictest_threshold"] = "strictest_threshold"
    result_mode: Literal["violated", "passed", "both"] = "both"


class RatioNumerator(StrictModel):
    layer: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    measure: Literal["area"]


class RatioDenominator(StrictModel):
    measure: Literal["zone_area"]


class ZonalRatioParams(StrictModel):
    zones_layer: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    numerator: RatioNumerator
    denominator: RatioDenominator
    operator: Literal["<", "<=", ">", ">=", "=="]
    threshold: float = Field(ge=0, le=100)
    unit: Literal["%"] = "%"
    exclusions: list[Literal["exclude_invalid_geometry_v1"]] = Field(
        default_factory=list, max_length=4
    )
    invalid_geometry_policy: Literal["repair", "reject"] = "repair"
    result_mode: Literal["violated", "passed", "both"] = "both"


class ResolvedRequirement(StrictModel):
    role: str
    requirement_type: Literal["layer", "attribute"]
    resolved: bool
    layer: str | None = None
    field: str | None = None
    unit: str | None = None
    quality: Literal["direct", "derived"] | None = None
    derive: Literal["height_to_floors_v1"] | None = None
    fill_rate: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = None


class VerificationCoverage(StrictModel):
    applicable_objects: int = Field(ge=0)
    checked_objects: int = Field(ge=0)
    unchecked_objects: int = Field(ge=0)
    fill_rate: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def counts_are_consistent(self) -> "VerificationCoverage":
        if self.checked_objects + self.unchecked_objects != self.applicable_objects:
            raise ValueError("checked + unchecked must equal applicable")
        return self


class ComplianceSummary(StrictModel):
    violated_objects: int = Field(ge=0)
    passed_objects: int = Field(ge=0)


class ComplianceEvidence(StrictModel):
    restriction_id: str
    template: str
    template_version: int
    object_ref: dict[str, Any]
    generator_ref: dict[str, Any] | None = None
    generator_refs: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    zone_ref: dict[str, Any] | None = None
    zone_refs: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    operation: str
    measured_value: float | int | None = None
    unit: str | None = None
    threshold: float | int | None = None
    operator: str | None = None
    violated: bool
    used_fields: list[dict[str, Any]] = Field(default_factory=list, max_length=24)
    provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list, max_length=50)
    input_revision: str | None = None
    neighbor_count: int | None = Field(default=None, ge=0)
    radius_m: float | None = Field(default=None, gt=0)
    neighbor_layers: list[str] = Field(default_factory=list, max_length=16)
    numerator_area_m2: float | None = Field(default=None, ge=0)
    denominator_area_m2: float | None = Field(default=None, ge=0)


class ComplianceResult(StrictModel):
    restriction_id: str
    template: str
    template_version: int
    verification_status: Literal[
        "complete", "partial", "unverifiable", "not_applicable", "unsupported"
    ]
    compliance_status: Literal["passed", "violated", "unknown"]
    coverage: VerificationCoverage
    summary: ComplianceSummary
    effective_requirements: DeclaredRequirements = Field(
        default_factory=DeclaredRequirements
    )
    resolved_requirements: list[ResolvedRequirement] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence_schema_version: Literal["1.0"] = "1.0"
    source: dict[str, Any] = Field(default_factory=dict)
    evidence: list[ComplianceEvidence] = Field(default_factory=list)
    violated_features: dict[str, Any] | None = None
    passed_features: dict[str, Any] | None = None

    @model_validator(mode="after")
    def status_pair_is_valid(self) -> "ComplianceResult":
        if self.verification_status in {
            "unverifiable",
            "not_applicable",
            "unsupported",
        }:
            if self.compliance_status != "unknown":
                raise ValueError(
                    "non-executed verification statuses require compliance=unknown"
                )
        if self.compliance_status == "violated" and self.summary.violated_objects == 0:
            raise ValueError(
                "violated compliance requires at least one violated object"
            )
        return self


TEMPLATE_PARAM_MODELS: dict[str, type[StrictModel]] = {
    "distance_from_source": DistanceFromSourceParams,
    "distance_table": DistanceTableParams,
    "presence_within": PresenceWithinParams,
    "zonal_attribute_threshold": ZonalAttributeThresholdParams,
    "zonal_ratio": ZonalRatioParams,
}
