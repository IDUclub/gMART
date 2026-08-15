import asyncio
import os
import shutil
import sys
import tempfile
from datetime import datetime

import anyio
import fastmcp.server.http as fastmcp_http
from fastapi.responses import RedirectResponse
from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan
from fastmcp_docs import FastMCPDocs
from loguru import logger
from mcp.server.streamable_http_manager import RequestBodyLimitMiddleware
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from src.__version__ import __VERSION__ as MCP_VERSION
from src.idu_mcp.common.logging.log_config import config_logger
from src.idu_mcp.common.memory import (
    count_types,
    memory_snapshot,
    release_memory,
    retention_report,
    rss_bytes,
)
from src.idu_mcp.common.middlewares.logging_middleware import RequestLoggingMiddleware
from src.idu_mcp.dependencies.dependencies import mcp_deps
from src.idu_mcp.prompts.restriction_prompts import mcp as restrictions_prompts_mcp
from src.idu_mcp.tools_interfaces.geom_interface import geometry_mcp
from src.idu_mcp.tools_interfaces.urb_api_interface import urban_api_mcp

# FastMCPDocs.setup() prints a "✓" via print(); make stdout/stderr UTF-8 so it
# does not raise UnicodeEncodeError on a Windows (cp1252) console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

log_path = config_logger()


def _max_request_body_size() -> int:
    """POST body limit for the MCP transport, in bytes."""

    try:
        return max(1, int(os.getenv("MCP_MAX_REQUEST_BODY_MB", "64"))) * 1024 * 1024
    except ValueError:
        return 64 * 1024 * 1024


def _session_idle_timeout() -> float | None:
    """Seconds a session may sit idle before it is reaped; 0/off disables it."""

    raw = os.getenv("MCP_SESSION_IDLE_TIMEOUT", "600").strip().lower()
    if raw in {"0", "off", "none", ""}:
        return None
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 600.0


class _BodyLimitSessionManager(fastmcp_http.FastMCPStreamableHTTPSessionManager):
    """Session manager with a raised request body limit and idle sessions reaped.

    The mcp SDK caps Streamable HTTP POST bodies at 4 MiB and answers 413 before
    parsing, which the geometry tools hit on large scenarios: ``CreateBuffers``
    and ``CreateRestrictions`` carry whole layers as arguments. FastMCP builds
    the manager itself and does not forward ``max_request_body_size``, so the
    limit is re-applied here after construction.

    Each session also starts a task that sits inside ``app.run()``, and the SDK
    only ever ends it on the idle timeout — which defaults to off. A client that
    opens a session per tool call therefore leaves one live transport behind per
    call: 800 of them within half an hour of benchmark load, and the process
    grew until the host started killing other containers' workers.

    The deadline only moves forward on incoming HTTP requests, and a tool call in
    progress sends none, so the timeout must stay comfortably above the longest
    call (``MCP_HTTP_TIMEOUT``) or it would cancel the work it is waiting for.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        limit = _max_request_body_size()
        self.max_request_body_size = limit
        self.asgi_app = RequestBodyLimitMiddleware(self._handle_request, limit)
        idle_timeout = _session_idle_timeout()
        if idle_timeout is not None and not getattr(self, "stateless", False):
            self.session_idle_timeout = idle_timeout
            logger.info(f"MCP sessions idle for {idle_timeout}s will be reaped")

    @staticmethod
    def _session_id(scope: dict) -> str | None:
        for name, value in scope.get("headers") or []:
            if name.lower() == b"mcp-session-id":
                return value.decode("ascii", "replace")
        return None

    def _drop_session(self, session_id: str | None) -> None:
        """Forget a session the client has closed, and end the task behind it.

        Dropping the instance alone is not enough: the session's ``app.run()``
        task keeps the transport, its streams and whatever they still hold. The
        idle scope is the SDK's own handle on that task, so cancelling it is what
        actually frees the session — the timeout then only covers clients that
        disconnect without saying so.
        """

        if not session_id:
            return
        transport = self._server_instances.pop(session_id, None)
        self._session_owners.pop(session_id, None)
        if transport is None:
            return
        idle_scope = getattr(transport, "idle_scope", None)
        if idle_scope is not None:
            idle_scope.cancel()

    async def handle_request(self, scope: dict, receive, send) -> None:
        await super().handle_request(scope, receive, send)
        # The client opens a session per tool call and closes it with DELETE;
        # without this the transport outlives the call and the process grows by
        # one session for every call it serves.
        if scope.get("method") == "DELETE":
            self._drop_session(self._session_id(scope))


fastmcp_http.FastMCPStreamableHTTPSessionManager = _BodyLimitSessionManager


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

asyncio.run(docs.setup())

mcp_app = main_mcp.http_app(host_origin_protection=False)
mcp_app.add_middleware(RequestLoggingMiddleware)


async def redirect_to_docs(request):
    return RedirectResponse(url="/docs")


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def memory(request: Request) -> JSONResponse:
    """RSS and the most numerous live objects, for tracking a growing process.

    Off unless ``MCP_DEBUG_MEMORY`` is set: walking every object costs a second
    on a large heap, which is not something a served deployment should expose.
    Compare two snapshots — a type whose count keeps climbing is a leak, while
    flat counts under a high RSS mean the allocator has not returned the arenas.
    """

    if os.getenv("MCP_DEBUG_MEMORY", "").strip().lower() not in {"1", "true", "yes"}:
        return JSONResponse(
            {"error": "set MCP_DEBUG_MEMORY=1 to enable"}, status_code=404
        )
    if request.query_params.get("retainers"):
        return JSONResponse(retention_report())
    types = request.query_params.get("types")
    if types:
        return JSONResponse(
            {
                "rss_mb": round(rss_bytes() / 1024 / 1024, 1),
                "counts": count_types(
                    [t.strip() for t in types.split(",") if t.strip()]
                ),
            }
        )
    snapshot = memory_snapshot()
    if request.query_params.get("release"):
        release_memory()
        snapshot["rss_mb_after_release"] = round(rss_bytes() / 1024 / 1024, 1)
    return JSONResponse(snapshot)


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
mcp_app.routes.insert(0, Route("/debug/memory", endpoint=memory))
