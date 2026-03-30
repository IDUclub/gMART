from src.idu_mcp.tools_services.geometry_tools import GeometryTools
from src.idu_mcp.api_clients.urban_api_client import UrbanApiClient
from src.idu_mcp.tools_services.urb_api_tools import UrbanApiTool
from .init_dependencies import init_dependencies
from .tool_deps import BaseDep, UrbanApiToolsDeps, GeomToolsDeps

mcp_deps: dict[
    str, BaseDep | UrbanApiToolsDeps | GeomToolsDeps
] = init_dependencies()


def get_urban_api_tools() -> UrbanApiTool:
    return mcp_deps["urban_api_tools"].urban_api_tools


def get_geom_tools() -> GeometryTools:
    return mcp_deps["geom_tools"]

def get_urban_api_client() -> UrbanApiClient:
    return mcp_deps["urban_api_tools"].urban_api_client
