from typing import Any

from fastapi import APIRouter, Depends

from src.agents.dependencies.dependencies import get_mcp_diagnostics_service
from src.agents.schema.mcp_diagnostics import McpToolCallRequest
from src.agents.services.mcp_diagnostics_service import McpDiagnosticsService

mcp_diagnostics_router = APIRouter(prefix="/mcp-diagnostics", tags=["mcp-diagnostics"])


@mcp_diagnostics_router.get("/sources")
async def list_sources(
    service: McpDiagnosticsService = Depends(get_mcp_diagnostics_service),
) -> list[dict[str, Any]]:
    """List the fixed MCP sources configured for this Agents instance."""

    return service.sources()


@mcp_diagnostics_router.get("/tools")
async def list_tools(
    source: str = "idu",
    service: McpDiagnosticsService = Depends(get_mcp_diagnostics_service),
) -> list[dict[str, Any]]:
    """List safe tools exposed by one configured MCP source."""

    return await service.list_tools(source)


@mcp_diagnostics_router.get("/prompts")
async def list_prompts(
    source: str = "idu",
    service: McpDiagnosticsService = Depends(get_mcp_diagnostics_service),
) -> list[dict[str, Any]]:
    """List prompts exposed by one configured MCP source."""

    return await service.list_prompts(source)


@mcp_diagnostics_router.post("/tools/call")
async def call_tool(
    request: McpToolCallRequest,
    service: McpDiagnosticsService = Depends(get_mcp_diagnostics_service),
) -> dict[str, Any]:
    """Execute one allowlisted read-only MCP tool and return its data payload."""

    result = await service.call_tool(
        request.source,
        request.name,
        request.arguments,
        group=request.group,
        meta=request.meta,
    )
    return {"result": result}
