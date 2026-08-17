from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable

from fastmcp import Client as McpClient
from pydantic import BaseModel

from src.agents.common.exceptions.token_exceptions import TokenExpiredError
from src.agents.mcp_clients.base_mcp_client import _is_token_expired

URBAN_MCP_GROUPS = (
    "projects",
    "territories",
    "physical_objects",
    "dictionaries",
    "indicators",
    "soc_groups",
)

URBAN_MCP_GROUP_DESCRIPTIONS = {
    "projects": "Проекты, сценарии и объекты внутри пользовательских сценариев.",
    "territories": "Территории и расположенные на них базовые городские данные.",
    "physical_objects": "Физические объекты, их сервисы и геометрии.",
    "dictionaries": "Справочники типов объектов, сервисов, территорий и показателей.",
    "indicators": "Показатели территорий, проектов и сценариев.",
    "soc_groups": "Социальные группы, ценности и связанные данные.",
}


@dataclass(frozen=True)
class UrbanMcpTool:
    group: str
    name: str
    title: str
    description: str
    input_schema: dict[str, Any]
    tags: tuple[str, ...]

    def compact_prompt_entry(self) -> dict[str, Any]:
        properties = self.input_schema.get("properties") or {}
        required = set(self.input_schema.get("required") or [])
        parameters = []
        for name, schema in properties.items():
            parameters.append(
                {
                    "name": name,
                    "required": name in required,
                    "type": _schema_type(schema),
                    "description": str(schema.get("description") or "")[:180],
                }
            )
        return {
            "group": self.group,
            "name": self.name,
            "title": self.title,
            "description": self.description[:320],
            "parameters": parameters,
            "tags": list(self.tags),
        }


def _schema_type(schema: dict[str, Any]) -> str:
    if schema.get("type"):
        return str(schema["type"])
    variants = schema.get("anyOf") or schema.get("oneOf") or []
    types = [str(item.get("type")) for item in variants if item.get("type")]
    return " | ".join(types) or "any"


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, BaseModel):
        return _plain(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class UrbanMcpClient:
    """One logical client over the six thematic Urban MCP endpoints."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        clients: dict[str, Any] | None = None,
        client_factory: Callable[..., Any] = McpClient,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._client_factory = client_factory
        self._clients = clients or self._build_clients(token)
        missing = set(URBAN_MCP_GROUPS) - set(self._clients)
        if missing:
            raise ValueError(f"Urban MCP clients missing groups: {sorted(missing)}")
        self._tools: dict[tuple[str, str], UrbanMcpTool] = {}

    def endpoint_url(self, group: str) -> str:
        if group not in URBAN_MCP_GROUPS:
            raise ValueError(f"Unknown Urban MCP group: {group}")
        return f"{self.base_url}/mcp/{group}/"

    def _build_clients(self, token: str) -> dict[str, Any]:
        auth = {"auth": token} if token else {}
        return {
            group: self._client_factory(self.endpoint_url(group), **auth)
            for group in URBAN_MCP_GROUPS
        }

    def update_token(self, new_token: str) -> None:
        self._token = new_token
        self._clients = self._build_clients(new_token)

    async def load_tools(self) -> list[UrbanMcpTool]:
        async def load_group(group: str) -> tuple[str, list[Any]]:
            try:
                async with self._clients[group] as client:
                    return group, list(await client.list_tools())
            except Exception as exc:
                if _is_token_expired(exc):
                    raise TokenExpiredError(str(exc)) from exc
                raise

        loaded = await asyncio.gather(
            *(load_group(group) for group in URBAN_MCP_GROUPS)
        )
        registry: dict[tuple[str, str], UrbanMcpTool] = {}
        global_names: dict[str, str] = {}
        for group, tools in loaded:
            for tool in tools:
                if not self._is_read_only(tool):
                    continue
                normalized = self._normalize_tool(group, tool)
                previous_group = global_names.get(normalized.name)
                if previous_group and previous_group != group:
                    raise ValueError(
                        f"Duplicate Urban MCP tool name {normalized.name!r} in "
                        f"{previous_group!r} and {group!r}"
                    )
                global_names[normalized.name] = group
                registry[(group, normalized.name)] = normalized
        self._tools = registry
        return list(registry.values())

    @staticmethod
    def _is_read_only(tool: Any) -> bool:
        annotations = getattr(tool, "annotations", None)
        if isinstance(annotations, BaseModel):
            annotations = annotations.model_dump()
        if isinstance(annotations, dict) and "readOnlyHint" in annotations:
            return bool(annotations["readOnlyHint"])
        read_only = getattr(annotations, "readOnlyHint", None)
        if read_only is not None:
            return bool(read_only)
        # Urban MCP follows stable Get* naming for its data-retrieval tools.
        return str(getattr(tool, "name", "")).startswith("Get")

    @staticmethod
    def _normalize_tool(group: str, tool: Any) -> UrbanMcpTool:
        meta = getattr(tool, "meta", None) or getattr(tool, "_meta", None) or {}
        if isinstance(meta, BaseModel):
            meta = meta.model_dump(mode="json")
        tags = ((meta or {}).get("fastmcp") or {}).get("tags") or []
        return UrbanMcpTool(
            group=group,
            name=str(tool.name),
            title=str(getattr(tool, "title", None) or tool.name),
            description=str(getattr(tool, "description", None) or ""),
            input_schema=_plain(getattr(tool, "inputSchema", None) or {}),
            tags=tuple(str(tag) for tag in tags),
        )

    def get_tool(self, group: str, tool_name: str) -> UrbanMcpTool:
        try:
            return self._tools[(group, tool_name)]
        except KeyError as exc:
            raise ValueError(
                f"Unknown read-only Urban MCP tool {group}.{tool_name}"
            ) from exc

    async def execute_tool(
        self,
        group: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        self.get_tool(group, tool_name)
        try:
            async with self._clients[group] as client:
                result = await client.call_tool(tool_name, arguments, meta=meta or {})
                return _plain(result.data)
        except Exception as exc:
            if _is_token_expired(exc):
                raise TokenExpiredError(str(exc)) from exc
            raise
