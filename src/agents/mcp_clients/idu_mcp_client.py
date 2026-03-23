from fastmcp import Client as McpClient
from mcp import ListToolsResult, Tool

from .base_mcp_client import BaseMcpClient


class IduMcpClient(BaseMcpClient):

    def __init__(self, mcp_client: McpClient):
        super().__init__(mcp_client)

    async def get_urban_api_tools(self) -> ListToolsResult | list[Tool]:
        """
        Function retrieves urban_api tools from IDU MCP server.
        Returns:
            ListToolResult | list[Tool]: list of available Urban API Tools.
        """

        return await self.load_tools(tags=["urban_api"])

    async def get_geometry_tools(self) -> ListToolsResult | list[Tool]:
        """
        Function retrieves geometry tools from IDU MCP server.
        Returns:
            ListToolResult | list[Tool]: list of available Urban API Tools.
        """

        return await self.load_tools(tags=["geometry"])
