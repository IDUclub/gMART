"""In-process stand-in for ``IduMcpClient``: the IDU tools without the transport.

The restrictions pipeline touches a very small part of the MCP surface — four
tools and two catalog prompts — and every one of them is, underneath, a plain
Python object: ``UrbanApiTool`` and ``GeometryTools`` take the bearer token as an
ordinary argument. Only ``extract_token`` ties them to HTTP, by reading the
request headers.

So for an experiment run the transport can be dropped entirely. That removes the
class of failures that has nothing to do with the model under test and that
dominated the previous run's error rate:

* ``CreateBuffers`` and ``CreateRestrictions`` carry whole layers as arguments,
  which on the large scenarios is tens of megabytes serialised per call — against
  a request body limit the server had to be patched to raise;
* every call opens an MCP session, and a client that opens one per call leaves a
  live transport behind for each;
* the same GeoJSON is encoded and decoded once per hop.

In-process the layers are passed as Python objects and none of that happens.

**The tool boundary is kept, deliberately.** Argument validation
(``GeometryToolValidator``) runs here exactly as it does inside the MCP tool
wrappers, and failures surface as the same ``ToolError``. Without that the local
arm would quietly accept malformed plans the HTTP arm rejects, and the two
transports would no longer be comparable — which is the point of running both.

Usage mirrors ``IduMcpClient``::

    client = LocalIduMcpClient(token, urban_api_url="http://urban-api")
    layers = await client.execute_tool("GetServices", {...})
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.idu_mcp.api_clients.urban_api_client import UrbanApiClient
from src.idu_mcp.common.api_handlers.json_api_handler import JsonApiHandler
from src.idu_mcp.tools_descriptions import geometry_validation_messages as messages
from src.idu_mcp.tools_services.entites.object_type_enum import ObjectTypeEnum
from src.idu_mcp.tools_services.geometry_tools import GeometryTools
from src.idu_mcp.tools_services.geometry_validator import GeometryToolValidator
from src.idu_mcp.tools_services.urb_api_tools import UrbanApiTool

try:  # pragma: no cover - import shape only
    from fastmcp.exceptions import ToolError
except ImportError:  # pragma: no cover
    ToolError = RuntimeError  # type: ignore[assignment,misc]


class LocalIduMcpClient:
    """The IDU MCP tools, called in this process.

    Attributes:
        token (str): bearer token handed to every Urban API call.
        urban_api_tools (UrbanApiTool): entity lookup and layer retrieval.
        geom_tools (GeometryTools): buffer and restriction geometry.
        calls (list[dict]): every tool invocation, in order, for the run record.
    """

    #: What ``mcp_source`` the pipeline's tool_call events should carry. The
    #: HTTP client reports "IDU_MCP_URL"; a local run says so plainly.
    MCP_SOURCE = "LOCAL_IDU_TOOLS"

    def __init__(
        self,
        token: str,
        urban_api_url: str,
        urban_api_tools: UrbanApiTool | None = None,
        geom_tools: GeometryTools | None = None,
    ) -> None:
        self._token = token
        self.urban_api_url = urban_api_url
        self.urban_api_client = UrbanApiClient(JsonApiHandler(urban_api_url))
        self.urban_api_tools = urban_api_tools or UrbanApiTool(self.urban_api_client)
        self.geom_tools = geom_tools or GeometryTools()
        self.calls: list[dict] = []

    # ------------------------------------------------------------- token ----
    def current_token(self) -> str:
        """The bearer token in use.

        ``IduMcpClient`` grows the same method; the pipeline asks the client for
        its token rather than reaching into an HTTP transport that a local client
        does not have.
        """

        return self._token

    def update_token(self, new_token: str) -> None:
        """Replace the bearer token used for all subsequent calls."""

        self._token = new_token

    # -------------------------------------------------------------- tools ---
    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict,
        meta: dict | None = None,  # noqa: ARG002 — MCP-only, no local meaning
        log: bool = False,
    ) -> Any:
        """Dispatch a tool by the name the pipeline knows it by."""

        handler = {
            "GetServices": self._get_services,
            "GetPhysicalObjects": self._get_physical_objects,
            "CreateBuffers": self._create_buffers,
            "CreateRestrictions": self._create_restrictions,
        }.get(tool_name)
        if handler is None:
            raise ToolError(f"Unknown tool {tool_name!r} for the local IDU client")
        self.calls.append({"tool": tool_name, "arguments_keys": sorted(arguments)})
        result = await handler(arguments)
        if log:
            logger.info(f"Executed local tool {tool_name}")
        return result

    async def _get_services(self, arguments: dict) -> dict[str, Any]:
        names = arguments.get("services_names") or []
        if not names:
            return {}
        return await self.urban_api_tools.get_entity_by_names(
            arguments["scenario_id"], names, ObjectTypeEnum.SERVICE, self._token
        )

    async def _get_physical_objects(self, arguments: dict) -> dict[str, Any]:
        names = arguments.get("physical_objects_names") or []
        if not names:
            return {}
        return await self.urban_api_tools.get_entity_by_names(
            arguments["scenario_id"],
            names,
            ObjectTypeEnum.PHYSICAL_OBJECT,
            self._token,
        )

    async def _create_buffers(self, arguments: dict) -> dict[str, Any]:
        buffer_info = arguments["buffer_info"]
        objects = arguments["objects"]
        # Same guard, same error, as the MCP tool wrapper.
        GeometryToolValidator.validate_buffers(buffer_info, objects)
        try:
            return await self.geom_tools.async_generate_geometry_buffers(
                buffer_info, objects
            )
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(messages.BUFFERS_RUNTIME_ERROR.format(error=exc)) from exc

    async def _create_restrictions(self, arguments: dict) -> dict[str, Any]:
        generators = arguments["generators"]
        objects = arguments["objects"]
        restrictions = arguments["restrictions"]
        layers = arguments["layers"]
        GeometryToolValidator.validate_restrictions(
            generators, objects, restrictions, layers
        )
        try:
            return await self.geom_tools.async_create_restrictions(
                layers, generators, objects, restrictions
            )
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(
                messages.RESTRICTIONS_RUNTIME_ERROR.format(error=exc)
            ) from exc

    # ------------------------------------------------------------ prompts ---
    async def get_available_services_prompt(self, scenario_id: int) -> str:
        """The scenario's service catalog, in the wording the prompt uses.

        The string shape matters: ``parse_catalog_prompt`` splits on the first
        colon and then on commas, so it must match the MCP prompt exactly.
        """

        names = await self.urban_api_client.get_available_scenario_services(
            scenario_id, self._token
        )
        return f"Список сервисов: {', '.join(name.lower() for name in names)}"

    async def get_available_physical_objects_prompt(self, scenario_id: int) -> str:
        """The scenario's physical-object catalog, in the prompt's wording."""

        names = await self.urban_api_client.get_available_physical_objects(
            scenario_id, self._token
        )
        return (
            "Список физических объектов: "
            f"{', '.join(name.lower() for name in names)}"
        )
