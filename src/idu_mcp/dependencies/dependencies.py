from src.idu_mcp.api_clients.urban_api_client import UrbanApiClient
from src.idu_mcp.dependencies.init_dependencies import init_dependencies
from src.idu_mcp.dependencies.tool_deps.base_tool_dep import BaseDep
from src.idu_mcp.dependencies.tool_deps.geom_tools_deps import GeomToolsDeps
from src.idu_mcp.dependencies.tool_deps.urban_api_tools_deps import (
    UrbanApiToolsDeps,
)
from src.idu_mcp.tools_services.geometry_tools import GeometryTools
from src.idu_mcp.tools_services.urb_api_tools import UrbanApiTool

mcp_deps: dict[str, BaseDep | UrbanApiToolsDeps | GeomToolsDeps] = init_dependencies()


def get_urban_api_tools() -> UrbanApiTool:
    return mcp_deps["urban_api_tools"].urban_api_tools


def get_geom_tools() -> GeometryTools:
    return mcp_deps["geom_tools"]


def get_urban_api_client() -> UrbanApiClient:
    return mcp_deps["urban_api_tools"].urban_api_client
