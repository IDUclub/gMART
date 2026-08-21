from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.agents.services.normgraph_reasoning import NormGraphRetrievalPlanner
from src.agents.services.service_entities.normgraph_plan import PrimaryTool

if TYPE_CHECKING:
    from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient


_METER_UNITS = {
    "m",
    "meter",
    "meters",
    "metre",
    "metres",
    "м",
    "метр",
    "метра",
    "метров",
}
_CANONICAL_KINDS = {
    "минимальное_расстояние",
    "запрет_размещения",
    "требование_размещения",
}
_ALL_RESTRICTIONS_INITIAL_LIMIT = 256
_ALL_RESTRICTIONS_MAX_LIMIT = 65_536


@dataclass(frozen=True)
class NormGraphRestrictionRetrieval:
    restrictions: list[dict[str, Any]]
    unsupported_count: int
    tool_call: dict[str, Any]


class NormGraphRestrictionRetriever:
    """Retrieve restrictions, retaining unsupported hits for compliance reporting."""

    def __init__(self, llm_client) -> None:
        self.planner = NormGraphRetrievalPlanner(llm_client)

    async def retrieve(
        self,
        client: "NormGraphMcpClient",
        model: str,
        user_query: str,
        history: list[dict] | None = None,
        retain_unsupported: bool = False,
        retrieve_all: bool = False,
    ) -> NormGraphRestrictionRetrieval:
        # Compliance is an audit, not a relevance-ranked QA answer.  The LLM may
        # choose a small result window for ordinary questions, but it must never
        # decide how many persisted norms an audit checks.
        if retrieve_all:
            hits, arguments = await self._retrieve_all(client)
            return self._result(
                hits, arguments, "search_restrictions", retain_unsupported
            )

        exhaustive_document_codes = self._exhaustive_document_codes(user_query)
        if exhaustive_document_codes is not None:
            arguments = {"limit": 100, "neighbors_depth": 0}
            result = await client.search_restrictions(**arguments)
            hits = [hit for hit in result.get("hits") or [] if isinstance(hit, dict)]
            if exhaustive_document_codes:
                hits = [
                    hit
                    for hit in hits
                    if self._matches_document_code(hit, exhaustive_document_codes)
                ]
            return self._result(
                hits,
                arguments,
                "search_restrictions",
                retain_unsupported,
            )

        plan = await self.planner.build_plan(model, user_query, history=history)
        if plan.primary_tool == PrimaryTool.APPLICABLE:
            arguments = self._active_arguments(
                object=plan.object,
                subject=plan.subject,
                kind=plan.kind,
                document_names=plan.document_names,
                limit=plan.limit,
            )
            result = await client.restrictions_applicable(**arguments)
            tool_name = "restrictions_applicable"
        else:
            arguments = self._active_arguments(
                query=plan.search_query,
                kind=plan.kind,
                document_names=plan.document_names,
                doc_type=plan.doc_type,
                corpus=plan.corpus,
                lang=plan.lang,
                tags=plan.tags,
                subject=plan.subject,
                object=plan.object,
                limit=plan.limit,
                neighbors_depth=0,
            )
            result = await client.search_restrictions(**arguments)
            tool_name = "search_restrictions"

        hits = [hit for hit in result.get("hits") or [] if isinstance(hit, dict)]
        explicit_hits = self._filter_explicit_references(hits, user_query)
        # LLM-generated structured filters are useful when they match the graph
        # vocabulary exactly, but a harmless wording difference (for example a
        # shortened document title) must not turn a real normative rule into an
        # empty result.  Retry once with semantic text search and no exact
        # filters; grounding below still accepts only canonical returned hits.
        if (not hits or explicit_hits == []) and (
            tool_name != "search_restrictions"
            or any(
                key not in {"query", "limit", "neighbors_depth"} for key in arguments
            )
        ):
            arguments = self._active_arguments(
                query=getattr(plan, "search_query", None) or user_query,
                # Clause numbers and document references are provenance fields,
                # not necessarily part of the embedded restriction text. Fetch
                # a wider candidate set, then narrow it deterministically below.
                limit=max(plan.limit, 100),
                neighbors_depth=0,
            )
            result = await client.search_restrictions(**arguments)
            tool_name = "search_restrictions"
            hits = [hit for hit in result.get("hits") or [] if isinstance(hit, dict)]

        explicit_hits = self._filter_explicit_references(hits, user_query)
        if explicit_hits is not None:
            hits = explicit_hits

        return self._result(hits, arguments, tool_name, retain_unsupported)

    async def _retrieve_all(
        self, client: "NormGraphMcpClient"
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Fetch the complete NormGraph corpus without a silent top-k window.

        NormGraph currently exposes a ``limit`` but no offset/cursor.  Grow the
        window until the server returns fewer rows than requested.  Hitting the
        explicit safety ceiling is an error: returning a partial audit as if it
        were complete would be worse than failing visibly.
        """

        limit = _ALL_RESTRICTIONS_INITIAL_LIMIT
        while True:
            arguments = {"limit": limit, "neighbors_depth": 0}
            response = await client.search_restrictions(**arguments)
            hits = [hit for hit in response.get("hits") or [] if isinstance(hit, dict)]
            if len(hits) < limit:
                # Preserve graph order while protecting the executor and summary
                # from duplicate restriction IDs.
                unique: dict[str, dict[str, Any]] = {}
                for index, hit in enumerate(hits):
                    unique.setdefault(str(hit.get("id") or index), hit)
                return list(unique.values()), arguments
            if limit >= _ALL_RESTRICTIONS_MAX_LIMIT:
                raise RuntimeError(
                    "NormGraph returned at least "
                    f"{_ALL_RESTRICTIONS_MAX_LIMIT} restrictions; complete "
                    "compliance retrieval requires server-side pagination"
                )
            limit = min(limit * 2, _ALL_RESTRICTIONS_MAX_LIMIT)

    @classmethod
    def _result(
        cls,
        hits: list[dict[str, Any]],
        arguments: dict[str, Any],
        tool_name: str,
        retain_unsupported: bool,
    ) -> NormGraphRestrictionRetrieval:
        supported = [hit for hit in hits if cls.is_canonical(hit)]
        restrictions = hits if retain_unsupported else supported
        unsupported_count = sum(
            1
            for hit in hits
            if not cls.is_canonical(hit)
            and (
                not isinstance(hit.get("check_plan"), dict)
                or hit["check_plan"].get("planner_status") == "unsupported"
            )
        )
        return NormGraphRestrictionRetrieval(
            restrictions=restrictions,
            unsupported_count=unsupported_count,
            tool_call={"function": {"name": tool_name, "arguments": arguments}},
        )

    @staticmethod
    def is_canonical(hit: dict[str, Any]) -> bool:
        """Phase one supports only metre-based spatial restrictions with a distance."""

        value = hit.get("value") or {}
        unit = str(value.get("unit") or "").strip().casefold().rstrip(".")
        number = value.get("number")
        kind = str(hit.get("kind") or "").strip().casefold()
        return (
            kind in _CANONICAL_KINDS
            and isinstance(number, (int, float))
            and not isinstance(number, bool)
            and number > 0
            and unit in _METER_UNITS
        )

    @staticmethod
    def _active_arguments(**values: Any) -> dict[str, Any]:
        return {
            key: value
            for key, value in values.items()
            if value is not None and value != []
        }

    @staticmethod
    def _exhaustive_document_codes(user_query: str) -> list[str] | None:
        query = user_query.casefold()
        asks_for_every_item = bool(
            re.search(r"\b(?:все\w*|кажд\w*)\b", query)
            and re.search(r"\b(?:огранич|норм|требован)", query)
        )
        if not asks_for_every_item:
            return None
        return list(dict.fromkeys(re.findall(r"\bсп\s*(\d+(?:\.\d+)+)", query)))

    @staticmethod
    def _matches_document_code(hit: dict[str, Any], codes: list[str]) -> bool:
        name = re.sub(
            r"\s+",
            "",
            str((hit.get("provenance") or {}).get("name") or "").casefold(),
        )
        return any(f"сп{code}" in name for code in codes)

    @staticmethod
    def _filter_explicit_references(
        hits: list[dict[str, Any]], user_query: str
    ) -> list[dict[str, Any]] | None:
        """Narrow broad vector results by explicit document/clause/distance references."""

        query = user_query.casefold()
        clause_match = re.search(r"(?:пункт(?:а|е|у)?|п\.)\s*(\d+(?:\.\d+)*)", query)
        document_match = re.search(r"\bсп\s*(\d+(?:\.\d+)+)", query)
        distances = {
            float(value.replace(",", "."))
            for value in re.findall(
                r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(?:[-‐‑–—]\s*)?м(?:\b|етр)",
                query,
            )
        }
        if not clause_match and not document_match and not distances:
            return None

        filtered = hits
        if clause_match:
            clause = clause_match.group(1).rstrip(".")
            filtered = [
                hit
                for hit in filtered
                if str((hit.get("provenance") or {}).get("numbering") or "")
                .strip()
                .rstrip(".")
                == clause
            ]
        if document_match:
            document_code = "сп" + document_match.group(1)
            filtered = [
                hit
                for hit in filtered
                if document_code
                in re.sub(
                    r"\s+",
                    "",
                    str((hit.get("provenance") or {}).get("name") or "").casefold(),
                )
            ]
        if distances:
            filtered = [
                hit
                for hit in filtered
                if isinstance((hit.get("value") or {}).get("number"), (int, float))
                and float((hit.get("value") or {})["number"]) in distances
            ]
        return filtered
