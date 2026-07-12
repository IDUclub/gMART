from enum import StrEnum

from pydantic import BaseModel


class PrimaryTool(StrEnum):
    """Which NormGraph MCP tool drives a retrieval round."""

    SEARCH = "search"  # search_restrictions — open/free-text questions
    APPLICABLE = "applicable"  # restrictions_applicable — "what applies to X" questions


class NormGraphPlan(BaseModel):
    """
    LLM-produced retrieval parameters for a single NormGraph QA round.
    Attributes:
        primary_tool (PrimaryTool): ``search`` calls ``search_restrictions``; ``applicable``
            calls ``restrictions_applicable`` (requires ``object``).
        search_query (str): Free-text query for ``search_restrictions``.
        object (str | None): The object/entity to check restrictions against — required for
            ``applicable``, optional filter for ``search``.
        subject (str | None): Restriction subject entity filter.
        kind (str | None): Restriction kind (controlled vocabulary) filter.
        document_names (list[str] | None): Restrict to these document names (any of).
        doc_type (str | None): Document type filter.
        corpus (str | None): Corpus filter.
        lang (str | None): Language filter.
        tags (list[str] | None): Clause tags filter.
        limit (int): Number of restrictions to retrieve (clamped 1..20 by the planner).
        neighbors_depth (int): > 0 to also expand the graph neighbourhood of the hits
            (clamped 0..2 by the planner).
        check_conflicts (bool): Whether the question warrants a ``list_conflicts`` pass
            over the top hits (e.g. the user asks about contradictions, or several hits
            look like they could disagree).
    """

    primary_tool: PrimaryTool = PrimaryTool.SEARCH
    search_query: str = ""
    object: str | None = None
    subject: str | None = None
    kind: str | None = None
    document_names: list[str] | None = None
    doc_type: str | None = None
    corpus: str | None = None
    lang: str | None = None
    tags: list[str] | None = None
    limit: int = 10
    neighbors_depth: int = 0
    check_conflicts: bool = False


class NormGraphCriticVerdict(BaseModel):
    """
    LLM critic's verdict on a drafted answer.
    Attributes:
        satisfied (bool): Whether the answer is accepted as-is.
        critique (str): Short explanation of what is wrong (empty when satisfied).
        refined_search_query (str | None): A better search query to use on the next round.
        refined_object (str | None): A better ``object`` filter to use on the next round
            (for ``applicable``-mode rounds).
    """

    satisfied: bool
    critique: str = ""
    refined_search_query: str | None = None
    refined_object: str | None = None
