from __future__ import annotations

from typing import Any

from fastmcp import Client as McpClient

from src.agents.mcp_clients.base_mcp_client import BaseMcpClient

# kind -> IDU_DVD MCP tool name (see IDU_DVD/src/mcp_server/server.py)
_KIND_TO_TOOL = {
    "text": "search_texts",
    "table": "search_tables",
    "all": "search_all",
}


class DvdMcpClient(BaseMcpClient):
    """
    Client for the IDU_DVD document vector-DB MCP server (regulatory documents search).

    Unlike IduMcpClient / EffectsMcpClient, the IDU_DVD MCP server is **unauthenticated**
    (it mounts the MCP app without JWT verification — see IDU_DVD/src/main.py), so this
    client carries no bearer token and does not implement ``update_token``.

    Exposed IDU_DVD MCP tools (see IDU_DVD/src/mcp_server/server.py):
    ``search_texts`` / ``search_tables`` / ``search_all`` (vector search), ``list_documents``,
    ``document_versions``, ``find_document``, ``get_document``, ``get_tags``,
    ``pending_references`` and ``job_status``. Only the search tools are used by the RAG
    pipeline.
    """

    def __init__(self, mcp_client: McpClient, mcp_url: str = "") -> None:
        super().__init__(mcp_client)
        self._mcp_url = mcp_url

    @staticmethod
    def tool_name_for_kind(kind: str) -> str:
        """Map a retrieval ``kind`` (text|table|all) to the IDU_DVD MCP tool name."""
        return _KIND_TO_TOOL.get(str(kind), "search_all")

    async def search(
        self,
        query: str,
        kind: str = "all",
        limit: int = 10,
        context_height: int = 0,
        name: str | None = None,
        version: str | None = None,
        tags: list[str] | None = None,
        document_names: list[str] | None = None,
        block: str | None = None,
        types: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run a vector search against the IDU_DVD MCP server.
        Args:
            query (str): Search query.
            kind (str): One of ``text`` / ``table`` / ``all`` — selects the search tool.
            limit (int): Maximum number of fragments to return.
            context_height (int): Number of neighbour fragments to attach before/after each hit.
            name (str | None): Optional single document name filter.
            version (str | None): Optional document version filter.
            tags (list[str] | None): Optional tags filter (any of).
            document_names (list[str] | None): Optional filter to any of these document names.
            block (str | None): Optional structural block filter (``main`` / ``amendment``).
            types (list[str] | None): Optional structural level filter (chapter/clause/table/...).
        Returns:
            dict[str, Any]: ``{"count": int, "hits": list[dict]}`` (see IDU_DVD SearchResponse).
        """

        tool_name = self.tool_name_for_kind(kind)
        arguments: dict[str, Any] = {
            "query": query,
            "limit": int(limit),
            "context_height": int(context_height),
        }
        if name:
            arguments["name"] = name
        if version:
            arguments["version"] = version
        if tags:
            arguments["tags"] = tags
        if document_names:
            arguments["document_names"] = document_names
        if block:
            arguments["block"] = block
        if types:
            arguments["types"] = types
        result = await self.execute_tool(tool_name, arguments)
        return self._normalize(result)

    @staticmethod
    def _normalize(result: Any) -> dict[str, Any]:
        """Normalize the MCP tool result into a ``{"count", "hits": [dict, ...]}`` dict."""
        if result is None:
            return {"count": 0, "hits": []}
        if not isinstance(result, dict):
            result = (
                result.model_dump()
                if hasattr(result, "model_dump")
                else {
                    "count": getattr(result, "count", 0),
                    "hits": getattr(result, "hits", []),
                }
            )
        hits = result.get("hits") or []
        result["hits"] = [
            hit if isinstance(hit, dict) else hit.model_dump()
            for hit in hits
            if hit is not None
        ]
        result.setdefault("count", len(result["hits"]))
        return result
