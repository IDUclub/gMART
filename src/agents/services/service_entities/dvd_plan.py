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
    """

    search_query: str = ""
    kind: SearchKind = SearchKind.ALL
    limit: int = 10
    context_height: int = 1


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
