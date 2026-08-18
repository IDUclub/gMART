from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from fastmcp import Client
from idu_service_auth import KeycloakTokenClient
from pydantic import BaseModel

from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.common.exceptions.base_exceptions import AgentsInputException
from src.agents.mcp_clients.urban_mcp_client import UrbanMcpClient
from src.common.service_auth import ServiceTokenAuth, user_id_from_jwt

_SAFE_NAME_PREFIXES = (
    "get",
    "list",
    "search",
    "find",
    "health",
    "calculate",
    "restrictions",
    "traverse",
    "pending",
    "job",
    "document",
)


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, BaseModel):
        return _plain(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _plain(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _annotation_read_only(tool: Any) -> bool | None:
    annotations = getattr(tool, "annotations", None)
    if isinstance(annotations, BaseModel):
        annotations = annotations.model_dump()
    if isinstance(annotations, dict) and "readOnlyHint" in annotations:
        return bool(annotations["readOnlyHint"])
    value = getattr(annotations, "readOnlyHint", None)
    return bool(value) if value is not None else None


def _is_safe_tool(tool: Any) -> bool:
    annotated = _annotation_read_only(tool)
    if annotated is not None:
        return annotated
    return str(getattr(tool, "name", "")).lower().startswith(_SAFE_NAME_PREFIXES)


class McpDiagnosticsService:
    """Allowlisted, read-only access to MCP servers configured for gMART."""

    def __init__(
        self,
        config: AgentsAppConfig,
        token: str,
        service_auth: KeycloakTokenClient,
    ) -> None:
        self.config = config
        self.token = token
        self.transport_auth = ServiceTokenAuth(service_auth, user_id_from_jwt(token))

    def sources(self) -> list[dict[str, Any]]:
        definitions = (
            ("idu", "IDU MCP", "Геометрия и Urban API", self.config.IDU_MCP_URL),
            (
                "urban",
                "Urban API MCP",
                "Проекты, территории, объекты и справочники",
                self.config.URBAN_MCP_URL,
            ),
            (
                "effects",
                "ObjectEffects MCP",
                "Расчёт эффектов и обеспеченности",
                self.config.EFFECTS_MCP_URL,
            ),
            (
                "dvd",
                "IDU_DVD MCP",
                "Поиск по документам",
                self.config.DVD_MCP_URL,
            ),
            (
                "normgraph",
                "NormGraph MCP",
                "Поиск нормативных ограничений",
                self.config.NORM_GRAPH_MCP_URL,
            ),
        )
        return [
            {
                "id": source,
                "title": title,
                "description": description,
                "available": bool(url),
            }
            for source, title, description, url in definitions
        ]

    def _url(self, source: str) -> str:
        urls = {
            "idu": self.config.IDU_MCP_URL,
            "effects": self.config.EFFECTS_MCP_URL,
            "dvd": self.config.DVD_MCP_URL,
            "normgraph": self.config.NORM_GRAPH_MCP_URL,
        }
        if source not in urls:
            raise AgentsInputException("Неизвестный MCP-источник", source)
        url = urls[source]
        if not url:
            raise AgentsInputException("MCP-источник не настроен", source)
        return url

    def _client(self, source: str) -> Client:
        url = self._url(source)
        return Client(url, auth=self.transport_auth)

    async def list_tools(self, source: str) -> list[dict[str, Any]]:
        if source == "urban":
            if not self.config.URBAN_MCP_URL:
                raise AgentsInputException("MCP-источник не настроен", source)
            urban = UrbanMcpClient(self.config.URBAN_MCP_URL, self.transport_auth)
            return [
                {
                    "type": "function",
                    "source": source,
                    "group": tool.group,
                    "read_only": True,
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in await urban.load_tools()
            ]

        client = self._client(source)
        async with client:
            tools = await client.list_tools()
        return [
            {
                "type": "function",
                "source": source,
                "group": None,
                "read_only": True,
                "function": {
                    "name": str(tool.name),
                    "description": str(getattr(tool, "description", None) or ""),
                    "parameters": _plain(getattr(tool, "inputSchema", None) or {}),
                },
            }
            for tool in tools
            if _is_safe_tool(tool)
        ]

    async def list_prompts(self, source: str) -> list[dict[str, Any]]:
        if source == "urban":
            return []
        client = self._client(source)
        async with client:
            prompts = await client.list_prompts()
        return [_plain(prompt) for prompt in prompts]

    async def call_tool(
        self,
        source: str,
        name: str,
        arguments: dict[str, Any],
        *,
        group: str | None,
        meta: dict[str, Any],
    ) -> Any:
        tools = await self.list_tools(source)
        allowed = {(item.get("group"), item["function"]["name"]) for item in tools}
        selected = (group if source == "urban" else None, name)
        if selected not in allowed:
            raise AgentsInputException(
                "Инструмент недоступен в безопасном режиме",
                {"source": source, "group": group, "name": name},
            )

        if source == "urban":
            urban = UrbanMcpClient(self.config.URBAN_MCP_URL or "", self.transport_auth)
            await urban.load_tools()
            return await urban.execute_tool(group or "", name, arguments, meta=meta)

        client = self._client(source)
        async with client:
            result = await client.call_tool(name, arguments, meta=meta)
        return _plain(result.data)
