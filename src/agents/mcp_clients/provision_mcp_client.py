from base_mcp_client import BaseMcpClient
from fastmcp import Client as McpClient


class RestrictionMcpClient(BaseMcpClient):

    def __init__(self, mcp_client: McpClient, mcp_url: str = ""):

        super().__init__(mcp_client)
        self._mcp_url = mcp_url
