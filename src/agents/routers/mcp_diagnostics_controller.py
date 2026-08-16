from typing import Any

from fastapi import APIRouter, Depends

from src.agents.dependencies.dependencies import get_idu_mcp_client
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.schema.mcp_diagnostics import McpToolCallRequest

mcp_diagnostics_router = APIRouter(prefix="/mcp-diagnostics", tags=["mcp-diagnostics"])


@mcp_diagnostics_router.get("/tools")
async def list_tools(
    mcp_client: IduMcpClient = Depends(get_idu_mcp_client),
) -> list[dict[str, Any]]:
    """List tools exposed by the IDU MCP configured for this Agents instance."""

    return await mcp_client.load_ollama_tools()


@mcp_diagnostics_router.get("/prompts")
async def list_prompts(
    mcp_client: IduMcpClient = Depends(get_idu_mcp_client),
) -> list[dict[str, Any]]:
    """List prompts exposed by the configured IDU MCP server."""

    prompts = await mcp_client.get_prompts()
    return [
        prompt.model_dump(mode="json") if hasattr(prompt, "model_dump") else prompt
        for prompt in prompts
    ]


@mcp_diagnostics_router.post("/tools/call")
async def call_tool(
    request: McpToolCallRequest,
    mcp_client: IduMcpClient = Depends(get_idu_mcp_client),
) -> dict[str, Any]:
    """Execute one configured IDU MCP tool and return its unmodified data payload."""

    result = await mcp_client.execute_tool(
        request.name,
        request.arguments,
        meta=request.meta,
        log=True,
    )
    return {"result": result}
