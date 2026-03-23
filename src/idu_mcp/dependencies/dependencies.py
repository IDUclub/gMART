from fastmcp import Context
from loguru import logger
from starlette.requests import Request

from src.idu_mcp.contexts.geom_contexts.create_buffers_context import BufferContext
from src.idu_mcp.contexts.geom_contexts.create_restrictions_context import RestrictionsContext

from .init_dependencies import init_dependencies
from .tool_deps import BaseDep, UrbanApiToolsDeps
from .tool_deps.geom_tools_deps import GeomToolsDeps

mcp_deps: dict[str, BaseDep] = init_dependencies()


async def __get_runtime_from_request__(request: Request) -> dict | list:
    """
    Function extracts runtime context from request as row dict
    Args:
        request (Request): tool call request
    Returns:
        dict | list: row representation of runtime request
    """

    body = await request.json()
    try:
        return body.get("_runtime")
    except Exception as e:
        logger.exception(e)
        raise


def get_urban_api_tools() -> UrbanApiToolsDeps:
    return mcp_deps["urban_api_tools"].urban_api_tools


def get_geom_tools() -> GeomToolsDeps:
    return mcp_deps["geom_tools"].geom_tools


def get_scenario_id(ctx: Context) -> int:
    """
    Function retrieves scenario_id from tool call context.
    Args:
        ctx (Context): Context for mcp tool call.
    Returns:
        int: Scenario ID value from headers
    """

    meta = ctx.request_context.meta if ctx.request_context else None
    if not meta or not hasattr(meta, "scenario_id"):
        raise ValueError("scenario_id is missing in meta")
    return int(meta.scenario_id)


async def get_buffers_context(ctx: Context) -> BufferContext:
    """
    Function extracts and forms tools params and context from request for create buffers tool.
    Args:
        ctx (Context): Context for mcp tool call.
    Returns:
        BufferContext: pydantic model of context for request
    """

    meta = ctx.request_context.meta if ctx.request_context else None
    if not meta or not hasattr(meta, "objects"):
        raise ValueError("objects are missing in meta")
    return BufferContext(objects=meta.objects)


async def get_restrictions_context(ctx: Context) -> RestrictionsContext:
    """
    Function extracts and forms tools params and context from request for create restriction tool.
    Args:
        ctx (Context): Context for mcp tool call.
    Returns:
        RestrictionsContext: pydantic model of context for tool call.
    """

    meta = ctx.request_context.meta if ctx.request_context else None
    if not meta or not hasattr(meta, "layers"):
        raise ValueError("layers are missing in meta")
    return RestrictionsContext(layers=meta.layers)
