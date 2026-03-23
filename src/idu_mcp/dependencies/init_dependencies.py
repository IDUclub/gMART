from src.idu_mcp.common.config.mcp_config_loader import load_config
from src.idu_mcp.tools_services.geometry_tools import GeometryTools

from .tool_deps import BaseDep, ServerDeps, UrbanApiToolsDeps


def init_dependencies() -> dict[str, BaseDep]:

    mcp_config = load_config()
    return {
        "mcp_config": mcp_config,
        "urban_api_tools": UrbanApiToolsDeps(mcp_config.URBAN_API_URL),
        "geom_tools": GeometryTools(),
        "server_deps": ServerDeps(mcp_config.APP_WORKERS),
    }
