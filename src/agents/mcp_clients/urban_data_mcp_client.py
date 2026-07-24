from __future__ import annotations

import asyncio
from typing import Any

from fastmcp import Client as McpClient
from loguru import logger

from src.agents.common.exceptions.token_exceptions import TokenExpiredError
from src.agents.mcp_clients.base_mcp_client import _is_token_expired

# Groups confirmed live against the deployed urban-mcp server (same host, one path per
# group). ``soc_groups`` is exposed at a hyphenated path, unlike its Python-style key.
_GROUP_PATHS: dict[str, str] = {
    "dictionaries": "dictionaries",
    "territories": "territories",
    "indicators": "indicators",
    "physical_objects": "physical_objects",
    "projects": "projects",
    "soc_groups": "soc-groups",
}
# Only the ``projects`` group touches private-project data and requires the caller's
# Urban API token; the other five groups are open reference/spatial data (confirmed live —
# ``list_tools`` succeeds against them with no ``auth`` set).
_AUTH_REQUIRED_GROUPS = {"projects"}
# The only mutating tool across all groups (creates a project in Urban API) — excluded so
# the LLM cannot trigger it from a Q&A conversation; this agent is strictly read-only.
_EXCLUDED_TOOLS = {"CreateProject"}


class UrbanDataMcpClient:
    """
    Client for the external, grouped Urban MCP server (urban-mcp): territorial/urban data
    tools backing the Urban API, split into groups exposed at distinct paths under one
    base URL (e.g. ``https://urban-mcp.example.ru/mcp/territories``).

    Unlike ``NormGraphMcpClient``, this client does not hardcode individual tool names —
    the tool set of every group is discovered dynamically via ``get_tools()`` and driven by
    the LLM's own tool-calling (see ``UrbanDataQaService``), since the full tool catalogue
    is not a stable, known-in-advance contract. The **set of groups** (``_GROUP_PATHS``) is
    a stable, live-verified fact about the deployed server, though, so it is kept as a small
    constant here rather than re-discovered — there is no endpoint that enumerates groups.

    Each group gets its own underlying ``fastmcp.Client`` because only the ``projects``
    group is authenticated (the caller's Urban API bearer token); the other five accept no
    ``auth`` at all — matching how the server actually behaves.
    """

    def __init__(self, base_url: str, token: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._group_clients: dict[str, McpClient] = {
            group: self._build_client(group) for group in _GROUP_PATHS
        }
        # Populated by get_tools(); maps a tool name to the group that exposes it, so
        # execute_tool() knows which underlying client to dispatch a call to.
        self._tool_group: dict[str, str] = {}

    def _build_client(self, group: str) -> McpClient:
        url = f"{self._base_url}/{_GROUP_PATHS[group]}"
        if group in _AUTH_REQUIRED_GROUPS and self._token:
            return McpClient(url, auth=self._token)
        return McpClient(url)

    def update_token(self, new_token: str) -> None:
        """Replace the bearer token and rebuild only the group client(s) that need it."""
        self._token = new_token
        for group in _AUTH_REQUIRED_GROUPS:
            self._group_clients[group] = self._build_client(group)

    async def get_tools(self) -> list[dict]:
        """
        Function retrieves the aggregated set of tools across every Urban MCP group.
        Returns:
            list[dict]: Available tools (minus ``_EXCLUDED_TOOLS``) in ollama-computable
                format. A group that fails to respond is skipped, not fatal.
        """

        results = await asyncio.gather(
            *(self._list_group_tools(group) for group in _GROUP_PATHS),
            return_exceptions=True,
        )

        tool_group: dict[str, str] = {}
        aggregated: list[dict] = []
        for group, result in zip(_GROUP_PATHS, results):
            if isinstance(result, BaseException):
                logger.warning(
                    f"Urban MCP group '{group}' unreachable, skipping: {result}"
                )
                continue
            for tool in result:
                if tool.name in _EXCLUDED_TOOLS:
                    continue
                if tool.name in tool_group:
                    logger.warning(
                        f"Duplicate Urban MCP tool name '{tool.name}' in groups "
                        f"'{tool_group[tool.name]}' and '{group}' — keeping the first"
                    )
                    continue
                tool_group[tool.name] = group
                aggregated.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
        self._tool_group = tool_group
        return aggregated

    async def _list_group_tools(self, group: str) -> list[Any]:
        client = self._group_clients[group]
        async with client:
            return await client.list_tools()

    async def execute_tool(
        self, tool_name: str, arguments: dict, meta: dict | None = None
    ):
        group = self._tool_group.get(tool_name)
        if group is None:
            raise ValueError(f"Unknown Urban MCP tool: {tool_name}")
        client = self._group_clients[group]
        try:
            async with client as mcp:
                result = await mcp.call_tool(tool_name, arguments, meta=meta or {})
                return result.data
        except Exception as e:
            if _is_token_expired(e):
                raise TokenExpiredError(str(e)) from e
            logger.exception(e)
            raise
