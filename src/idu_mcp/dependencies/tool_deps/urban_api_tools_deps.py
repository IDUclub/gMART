from idu_service_auth import KeycloakTokenClient

from src.idu_mcp.api_clients.urban_api_client import UrbanApiClient
from src.idu_mcp.common.api_handlers.json_api_handler import JsonApiHandler
from src.idu_mcp.dependencies.tool_deps.base_tool_dep import BaseDep
from src.idu_mcp.tools_services.urb_api_tools import UrbanApiTool


class UrbanApiToolsDeps(BaseDep):
    """
    Class for managing urban api tools
    Attributes:
        urban_api_client: UrbanApiClient object
        urban_api_tools: UrbanApiTool object
    """

    def __init__(self, urban_api_url: str, service_auth: KeycloakTokenClient):
        """
        Constructor for UrbanApiToolsDeps class
        Args:
            urban_api_url (str): urban api url
        """

        super().__init__()
        self.urban_api_client: UrbanApiClient = UrbanApiClient(
            JsonApiHandler(urban_api_url, service_auth=service_auth)
        )
        self.urban_api_tools: UrbanApiTool = UrbanApiTool(self.urban_api_client)
