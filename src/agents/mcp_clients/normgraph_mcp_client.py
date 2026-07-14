from __future__ import annotations

from typing import Any

from fastmcp import Client as McpClient

from src.agents.mcp_clients.base_mcp_client import BaseMcpClient


class NormGraphMcpClient(BaseMcpClient):
    """
    Client for the NormGraph MCP server (graph-RAG of normative restrictions, СП/СНиП/ГОСТ/СанПиН).

    Like DvdMcpClient, the NormGraph MCP server is **unauthenticated** (no JWT verification —
    see NormGraph/src/main.py), so this client carries no bearer token and does not implement
    ``update_token``.

    Exposed NormGraph MCP tools (see NormGraph/src/mcp_server/server.py): ``search_restrictions``,
    ``restrictions_applicable``, ``get_restriction``, ``traverse_restrictions``, ``list_entities``,
    ``list_restriction_kinds``, ``list_conflicts``, ``health``.
    """

    def __init__(self, mcp_client: McpClient, mcp_url: str = "") -> None:
        super().__init__(mcp_client)
        self._mcp_url = mcp_url

    async def search_restrictions(
        self,
        query: str | None = None,
        kind: str | None = None,
        document_names: list[str] | None = None,
        version: str | None = None,
        doc_type: str | None = None,
        corpus: str | None = None,
        lang: str | None = None,
        tags: list[str] | None = None,
        subject: str | None = None,
        object: str | None = None,
        limit: int = 10,
        neighbors_depth: int = 0,
    ) -> dict[str, Any]:
        """
        Free-text and/or filtered search over normative restrictions.
        Args:
            query (str | None): Free-text query; omit for a purely filtered listing.
            kind (str | None): Restriction kind from the controlled vocabulary.
            document_names (list[str] | None): Restrict to any of these document names.
            version (str | None): Document version/redaction filter.
            doc_type (str | None): Document type filter.
            corpus (str | None): Corpus filter.
            lang (str | None): Language filter.
            tags (list[str] | None): Clause tags filter.
            subject (str | None): Restriction subject entity filter.
            object (str | None): Restriction object entity filter.
            limit (int): Maximum number of hits.
            neighbors_depth (int): > 0 to also return the graph neighbourhood of the hits.
        Returns:
            dict[str, Any]: ``{"count", "hits", "neighbors", "dvd_fallback"}`` (SearchResponse).
        """

        arguments = self._filters(
            query=query,
            kind=kind,
            document_names=document_names,
            version=version,
            doc_type=doc_type,
            corpus=corpus,
            lang=lang,
            tags=tags,
            subject=subject,
            object=object,
            limit=int(limit),
            neighbors_depth=int(neighbors_depth),
        )
        result = await self.execute_tool("search_restrictions", arguments)
        return self._normalize_search(result)

    async def restrictions_applicable(
        self,
        object: str,
        subject: str | None = None,
        kind: str | None = None,
        document_names: list[str] | None = None,
        version: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Restrictions applying to a given object/entity (compliance-style check).
        Returns:
            dict[str, Any]: ``{"count", "hits", "neighbors", "dvd_fallback"}`` (SearchResponse).
        """

        arguments = self._filters(
            object=object,
            subject=subject,
            kind=kind,
            document_names=document_names,
            version=version,
            limit=int(limit),
        )
        result = await self.execute_tool("restrictions_applicable", arguments)
        return self._normalize_search(result)

    async def get_restriction(self, restriction_id: str) -> dict[str, Any] | None:
        """One restriction with full provenance and its direct graph neighbours."""

        result = await self.execute_tool(
            "get_restriction", {"restriction_id": restriction_id}
        )
        return self._to_dict(result) if result is not None else None

    async def traverse_restrictions(
        self, restriction_id: str, depth: int = 1
    ) -> dict[str, Any] | None:
        """Traverse the restriction graph from a restriction up to ``depth`` hops."""

        result = await self.execute_tool(
            "traverse_restrictions",
            {"restriction_id": restriction_id, "depth": int(depth)},
        )
        if result is None:
            return None
        result = self._to_dict(result)
        result["nodes"] = [self._to_dict(node) for node in result.get("nodes") or []]
        return result

    async def list_entities(
        self, query: str | None = None, limit: int = 50
    ) -> list[dict]:
        """Canonical entities (subjects/objects), most-referenced first."""

        arguments: dict[str, Any] = {"limit": int(limit)}
        if query:
            arguments["query"] = query
        result = await self.execute_tool("list_entities", arguments)
        return [self._to_dict(item) for item in (result or [])]

    async def list_restriction_kinds(self) -> list[dict]:
        """The restriction-kind vocabulary, including auto-added pending kinds."""

        result = await self.execute_tool("list_restriction_kinds", {})
        return [self._to_dict(item) for item in (result or [])]

    async def list_conflicts(
        self,
        user_id: str | None = None,
        scenario_id: str | None = None,
        restriction_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Possible conflicts (contradicting restriction values) between restrictions.
        Returns:
            dict[str, Any]: ``{"count", "conflicts"}`` (ConflictListResponse); each conflict is
            ``{"restriction", "other", "reason", "severity"}``.
        """

        arguments: dict[str, Any] = {"limit": int(limit)}
        if user_id:
            arguments["user_id"] = user_id
        if scenario_id:
            arguments["scenario_id"] = scenario_id
        if restriction_id:
            arguments["restriction_id"] = restriction_id
        result = await self.execute_tool("list_conflicts", arguments)
        result = self._to_dict(result) if result is not None else {}
        conflicts = [self._to_dict(item) for item in result.get("conflicts") or []]
        for conflict in conflicts:
            conflict["restriction"] = self._to_dict(conflict.get("restriction"))
            conflict["other"] = self._to_dict(conflict.get("other"))
        result["conflicts"] = conflicts
        result.setdefault("count", len(conflicts))
        return result

    @staticmethod
    def _filters(**kwargs: Any) -> dict[str, Any]:
        """Drop ``None``/empty-list values so the MCP call only carries active filters."""

        return {
            key: value
            for key, value in kwargs.items()
            if value is not None and value != []
        }

    @classmethod
    def _to_dict(cls, obj: Any) -> Any:
        """
        Best-effort recursive conversion of an MCP-returned object into plain data.

        FastMCP rehydrates a tool's structured output (via ``result.data``) into synthetic
        types built from its output schema, which are neither a ``dict`` nor a pydantic-v2
        model with ``model_dump``. Try the known converters in order and fall back to
        attribute inspection. The conversion is recursive: nested synthetic objects
        (e.g. ``provenance``/``value`` inside a search hit) must also become dicts —
        downstream consumers (NormGraphContextBuilder) treat the result as plain JSON.
        """
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {key: cls._to_dict(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls._to_dict(item) for item in obj]
        for attr in ("model_dump", "dict", "_asdict"):  # pydantic v2 / v1 / namedtuple
            converter = getattr(obj, attr, None)
            if callable(converter):
                try:
                    return cls._to_dict(converter())
                except TypeError:
                    continue
        if hasattr(obj, "__dict__"):
            return {
                key: cls._to_dict(value)
                for key, value in vars(obj).items()
                if not key.startswith("_")
            }
        return obj

    @classmethod
    def _normalize_search(cls, result: Any) -> dict[str, Any]:
        """Normalize a SearchResponse-shaped MCP result into a plain dict."""

        if result is None:
            return {"count": 0, "hits": [], "neighbors": [], "dvd_fallback": []}
        result = cls._to_dict(result)
        if not isinstance(result, dict):
            result = {
                "count": getattr(result, "count", 0),
                "hits": getattr(result, "hits", []),
                "neighbors": getattr(result, "neighbors", []),
                "dvd_fallback": getattr(result, "dvd_fallback", []),
            }
        hits = [
            cls._to_dict(hit) for hit in (result.get("hits") or []) if hit is not None
        ]
        neighbors = [
            cls._to_dict(neighbor)
            for neighbor in (result.get("neighbors") or [])
            if neighbor is not None
        ]
        for neighbor in neighbors:
            neighbor["restriction"] = cls._to_dict(neighbor.get("restriction"))
        result["hits"] = hits
        result["neighbors"] = neighbors
        result["dvd_fallback"] = [
            cls._to_dict(hit)
            for hit in (result.get("dvd_fallback") or [])
            if hit is not None
        ]
        result.setdefault("count", len(hits))
        return result
