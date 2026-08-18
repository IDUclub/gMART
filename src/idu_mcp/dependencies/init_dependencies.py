import redis.asyncio as aioredis

from src.common.service_auth import build_service_auth
from src.idu_mcp.common.config.mcp_config_loader import load_config
from src.idu_mcp.dependencies.tool_deps.base_tool_dep import BaseDep
from src.idu_mcp.dependencies.tool_deps.server_deps import ServerDeps
from src.idu_mcp.dependencies.tool_deps.urban_api_tools_deps import (
    UrbanApiToolsDeps,
)
from src.idu_mcp.tools_services.geometry_tools import GeometryTools
from src.idu_mcp.tools_services.workspace_store import WorkspaceStore


def init_dependencies() -> dict[str, BaseDep]:

    mcp_config = load_config()
    service_auth = build_service_auth()
    workspace_store = None
    if mcp_config.WORKSPACE_ENABLED:
        workspace_store = WorkspaceStore(
            aioredis.from_url(mcp_config.REDIS_URL, decode_responses=True),
            root=mcp_config.WORKSPACE_DIR,
            ttl_seconds=mcp_config.WORKSPACE_TTL_SECONDS,
            max_dataset_bytes=mcp_config.WORKSPACE_MAX_DATASET_BYTES,
            max_total_bytes=mcp_config.WORKSPACE_MAX_TOTAL_BYTES,
        )
    return {
        "mcp_config": mcp_config,
        "service_auth": service_auth,
        "urban_api_tools": UrbanApiToolsDeps(mcp_config.URBAN_API_URL, service_auth),
        "geom_tools": GeometryTools(),
        "server_deps": ServerDeps(mcp_config.APP_WORKERS),
        "workspace_store": workspace_store,
    }
