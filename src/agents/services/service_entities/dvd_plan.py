from enum import StrEnum

from pydantic import BaseModel


class SearchKind(StrEnum):
    """Which IDU_DVD search surface to query."""

    TEXT = "text"
    TABLE = "table"
    ALL = "all"


class RetrievalPlan(BaseModel):
    """
    LLM-produced retrieval parameters for a single RAG search round.
    Attributes:
        search_query (str): Reformulated query for the vector search.
        kind (SearchKind): Search surface — text fragments, tables, or both.
        limit (int): Number of fragments to retrieve (clamped to 1..20 by the planner).
        context_height (int): Neighbour fragments to attach per hit (clamped to 0..5).
        document_names (list[str] | None): Restrict the search to these document names
            (any of); ``None`` searches across the whole base.
        block (str | None): Restrict to ``main`` (base text) or ``amendment`` (changes/
            поправки); ``None`` searches both.
        types (list[str] | None): Restrict to these structural levels (``chapter`` /
            ``section`` / ``clause`` / ``subclause`` / ``table`` / ``definition`` / ...);
            ``None`` searches all levels.
    """

    search_query: str = ""
    kind: SearchKind = SearchKind.ALL
    limit: int = 10
    context_height: int = 1
    document_names: list[str] | None = None
    block: str | None = None
    types: list[str] | None = None


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
