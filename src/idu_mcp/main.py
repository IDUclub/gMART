import asyncio
import os
import shutil
import sys
import tempfile
from datetime import datetime

import anyio
from fastapi.responses import RedirectResponse
from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan
from fastmcp_docs import FastMCPDocs
from loguru import logger
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from src.idu_mcp.common.logging.log_config import config_logger
from src.idu_mcp.common.middlewares.logging_middleware import RequestLoggingMiddleware
from src.idu_mcp.dependencies.dependencies import mcp_deps
from src.idu_mcp.prompts.restriction_prompts import mcp as restrictions_prompts_mcp
from src.idu_mcp.tools_interfaces.geom_interface import geometry_mcp
from src.idu_mcp.tools_interfaces.urb_api_interface import urban_api_mcp
from src.__version__ import __VERSION__ as MCP_VERSION

# FastMCPDocs.setup() prints a "✓" via print(); make stdout/stderr UTF-8 so it
# does not raise UnicodeEncodeError on a Windows (cp1252) console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

log_path = config_logger()


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
main_mcp.mount(restrictions_prompts_mcp)

docs = FastMCPDocs(
    mcp=main_mcp,
    title="IDU Fast MCP Server",
    version=MCP_VERSION,
    description="Documentation for IDU MCP tools",
    base_url="http://localhost:8000",
)
# Register the documentation routes (/docs, /openapi.json, /api/tools, ...) on
# the FastMCP server BEFORE building the HTTP app. http_app() snapshots the
# routes at call time, so docs registered later (e.g. in the lifespan) would
# never reach the served app.
asyncio.run(docs.setup())

mcp_app = main_mcp.http_app()
mcp_app.add_middleware(RequestLoggingMiddleware)


async def redirect_to_docs(request):
    return RedirectResponse(url="/docs")


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def get_logs(request: Request) -> FileResponse:
    """Download a stable snapshot of the idu_mcp log file.

    The live log file keeps growing while the response streams (request
    logging appends on every request, loguru flushes asynchronously), which
    would make the streamed body exceed the Content-Length computed from the
    initial ``stat``. Serving an immutable copy avoids that race.
    """
    snapshot = tempfile.NamedTemporaryFile(
        prefix="idu-mcp-", suffix=".log", delete=False
    )
    snapshot.close()
    await anyio.to_thread.run_sync(shutil.copyfile, log_path, snapshot.name)
    return FileResponse(
        path=snapshot.name,
        filename=f"idu-mcp-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log",
        media_type="text/plain",
        background=BackgroundTask(os.unlink, snapshot.name),
    )


mcp_app.routes.insert(0, Route("/", endpoint=redirect_to_docs))
mcp_app.routes.insert(0, Route("/health", endpoint=health))
mcp_app.routes.insert(0, Route("/logs", endpoint=get_logs))
