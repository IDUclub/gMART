from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    field_validator,
)


class SearchKind(StrEnum):
    """Which IDU_DVD search surface to query."""

    TEXT = "text"
    TABLE = "table"
    ALL = "all"


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _BaseRetrievalPlan(BaseModel):
    """
    LLM-produced retrieval parameters for a single RAG search round.
    Attributes:
        search_query (str): Reformulated query for the vector search.
        kind (SearchKind): Search surface — text fragments, tables, or both.
        limit (int): Number of fragments to retrieve (1..20).
        context_height (int): Neighbour fragments to attach per hit (0..5).
        document_names (list[str] | None): Restrict the search to these document names
            (any of); ``None`` searches across the whole base.
        block (str | None): Restrict to ``main`` (base text) or ``amendment`` (changes/
            поправки); ``None`` searches both.
        types (list[str] | None): Restrict to these structural levels (``chapter`` /
            ``section`` / ``clause`` / ``subclause`` / ``table`` / ``definition`` / ...);
            ``None`` searches all levels.
    """

    model_config = ConfigDict(extra="forbid")

    search_query: str = ""
    kind: SearchKind = SearchKind.ALL
    limit: int = Field(default=10, ge=1, le=20)
    context_height: int = Field(default=1, ge=0, le=5)
    document_names: list[str] | None = None
    block: str | None = None
    types: list[str] | None = None
    pattern: str | None = None
    name_query: str | None = None
    name_mode: Literal["strict", "expanded"] = "strict"
    name_scope: Literal["self", "path"] = "self"
    doc_id: str | None = None
    version: str | None = None
    include_children: bool = True
    allow_multiple: bool = False

    @field_validator(
        "pattern", "name_query", "doc_id", "version", "block", mode="before"
    )
    @classmethod
    def normalize_absent_filter(cls, value):
        # Some compatible LLMs encode an absent optional filter as the string
        # "null". Never use it as an actual document/fragment identifier.
        if isinstance(value, str) and value.strip().casefold() in {"", "null", "none"}:
            return None
        return value


class SemanticRetrievalPlan(_BaseRetrievalPlan):
    """Vector retrieval without a structural or fragment-name constraint."""

    retrieval_mode: Literal["semantic"]
    pattern: None = None
    name_query: None = None


class StructureRetrievalPlan(_BaseRetrievalPlan):
    """Structural retrieval, optionally narrowed by a fragment name."""

    retrieval_mode: Literal["structure"]
    pattern: NonEmptyStr
    name_query: NonEmptyStr | None = None


class NameRetrievalPlan(_BaseRetrievalPlan):
    """Fragment-name retrieval without a structural address."""

    retrieval_mode: Literal["name"]
    pattern: None = None
    name_query: NonEmptyStr


RetrievalPlan: TypeAlias = Annotated[
    SemanticRetrievalPlan | StructureRetrievalPlan | NameRetrievalPlan,
    Field(discriminator="retrieval_mode"),
]

_RETRIEVAL_PLAN_ADAPTER = TypeAdapter(RetrievalPlan)


def validate_retrieval_plan(value) -> RetrievalPlan:
    """Validate raw or updated data and return the matching plan variant."""

    return _RETRIEVAL_PLAN_ADAPTER.validate_python(value)


class CriticVerdict(BaseModel):
    """
    LLM critic's verdict on a drafted answer.
    Attributes:
        satisfied (bool): Whether the answer is accepted as-is.
        critique (str): Short explanation of what is wrong (empty when satisfied).
        refined_search_query (str | None): A better search query to use on the next round.
    """

    satisfied: bool
    critique: str = ""
    refined_search_query: str | None = None
