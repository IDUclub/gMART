from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan
from loguru import logger

from src.idu_mcp.dependencies.dependencies import mcp_deps
from src.idu_mcp.tools_interfaces import geometry_mcp, urban_api_mcp


@lifespan
async def main_app_lifespan(server: FastMCP):
    logger.info(f"Loaded dependencies {mcp_deps}")
    try:
        yield {"started_at": "2024-01-01"}
    finally:
        logger.info("Shutting down...")


main_mcp = FastMCP("IDU Fast MCP Server", lifespan=main_app_lifespan)
main_mcp.mount(urban_api_mcp)
main_mcp.mount(geometry_mcp)

mcp_app = main_mcp.http_app("/mcp")
