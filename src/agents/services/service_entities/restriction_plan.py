from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RestrictionTaskMode(StrEnum):
    BUFFERS_ONLY = "buffers_only"
    RESTRICTIONS = "restrictions"
    NEEDS_CLARIFICATION = "needs_clarification"


class EntityRef(BaseModel):
    name: str = Field(description="Canonical entity name from the provided catalog.")
    entity_type: Literal["service", "physical_object"]


class RestrictionProvenance(BaseModel):
    """Grounding of an executable rule in NormGraph or the current user request."""

    model_config = ConfigDict(extra="allow")

    document_id: str | None = None
    document_name: str | None = None
    document_version: str | None = None
    clause_id: str | None = None
    clause_number: str | None = None
    breadcrumb: str | None = None
    extraction_text: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class BufferRule(BaseModel):
    source_name: str = Field(description="Canonical source entity name.")
    buffer_size: float = Field(gt=0, description="Buffer distance in meters.")
    buffer_type: Literal["round", "flat", "square"] = "round"
    title: str = Field(description="Human readable buffer or restriction title.")
    origin: Literal["normgraph", "user"] = "user"
    restriction_id: str | None = None
    provenance: RestrictionProvenance | None = None


class RestrictionRule(BaseModel):
    source_name: str = Field(description="Canonical buffer source entity name.")
    # Required, and required to be non-empty, because the pipeline has always
    # treated it that way: `_canonicalize_restriction_rules` drops any rule whose
    # targets are empty, which degrades the whole plan to needs_clarification. It
    # was optional only because that is what `list[str]` defaults to, and no rule
    # anywhere is meaningful without targets. Models read the schema literally --
    # gemma3 left it out of 88.7% of its rules and llama3.1 of 99.8%, while naming
    # the very same entity in `target_entities` and in the rule's own prose -- so
    # their plans died on a field nothing had asked them to fill. Documenting it
    # in the prompt moved nothing; only the schema is binding.
    target_names: list[str] = Field(
        min_length=1,
        description=(
            "Canonical names, taken from target_entities, of the entities this "
            "rule counts inside the source buffer. Must be non-empty: a rule "
            "with no targets is dropped and the plan fails. Naming the targets "
            "in title or description does not substitute for this field."
        ),
    )
    title: str
    description: str
    origin: Literal["normgraph", "user"] = "user"
    restriction_id: str | None = None
    provenance: RestrictionProvenance | None = None


class SelectionReason(BaseModel):
    step: Literal[
        "mode",
        "source_entities",
        "target_entities",
        "buffer_rules",
        "restriction_rules",
        "needs_clarification",
    ]
    reason: str


class RestrictionPlan(BaseModel):
    mode: RestrictionTaskMode
    source_entities: list[EntityRef] = Field(default_factory=list)
    target_entities: list[EntityRef] = Field(default_factory=list)
    buffer_rules: list[BufferRule] = Field(default_factory=list)
    restriction_rules: list[RestrictionRule] = Field(default_factory=list)
    selection_reasons: list[SelectionReason] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0)
    clarification_question: str | None = None
    original: str
