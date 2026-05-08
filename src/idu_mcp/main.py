from fastapi.responses import RedirectResponse
from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan
from fastmcp_docs import FastMCPDocs
from loguru import logger
from starlette.routing import Route

from src.idu_mcp.dependencies.dependencies import mcp_deps
from src.idu_mcp.prompts.restriction_prompts import mcp as restrictions_prompts_mcp
from src.idu_mcp.tools_interfaces.geom_interface import geometry_mcp
from src.idu_mcp.tools_interfaces.urb_api_interface import urban_api_mcp


async def setup_docs(server: FastMCP):

    docs = FastMCPDocs(
        mcp=server,
        title="IDU Fast MCP Server",
        version="1.0.0",
        description="Documentation for IDU MCP tools",
        base_url="http://localhost:8000",
    )

    await docs.setup()


@lifespan
async def main_app_lifespan(server: FastMCP):
    logger.info(f"Loaded dependencies {mcp_deps}")
    await setup_docs(server)
    try:
        yield {"started_at": "2024-01-01"}
    finally:
        logger.info("Shutting down...")


main_mcp = FastMCP("IDU Fast MCP Server", lifespan=main_app_lifespan)
main_mcp.mount(urban_api_mcp)
main_mcp.mount(geometry_mcp)
main_mcp.mount(restrictions_prompts_mcp)

docs = FastMCPDocs(
    mcp=main_mcp,
    title="IDU Fast MCP Server",
    version="1.0.0",
)

mcp_app = main_mcp.http_app()


async def redirect_to_docs(request):
    return RedirectResponse(url="/docs")


mcp_app.routes.insert(0, Route("/", endpoint=redirect_to_docs))
