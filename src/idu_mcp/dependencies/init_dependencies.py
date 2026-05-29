from src.common.auth.auth_client import AuthenticationClient
from src.idu_mcp.common.config.mcp_config_loader import load_config
from src.idu_mcp.dependencies.tool_deps.base_tool_dep import BaseDep
from src.idu_mcp.dependencies.tool_deps.server_deps import ServerDeps
from src.idu_mcp.dependencies.tool_deps.urban_api_tools_deps import UrbanApiToolsDeps
from src.idu_mcp.tools_services.geometry_tools import GeometryTools


def init_dependencies() -> dict[str, BaseDep]:

    mcp_config = load_config()
    auth_client = AuthenticationClient(mcp_config.AUTH_CONFIG)
    return {
        "mcp_config": mcp_config,
        "auth_client": auth_client,
        "urban_api_tools": UrbanApiToolsDeps(mcp_config.URBAN_API_URL),
        "geom_tools": GeometryTools(),
        "server_deps": ServerDeps(mcp_config.APP_WORKERS),
    }
