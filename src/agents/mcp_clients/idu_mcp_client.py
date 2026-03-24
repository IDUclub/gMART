from fastmcp import Client as McpClient

from .base_mcp_client import BaseMcpClient


#TODO add prompts cache
class IduMcpClient(BaseMcpClient):

    def __init__(self, mcp_client: McpClient):
        super().__init__(mcp_client)

    async def get_urban_api_tools(self) ->list[dict]:
        """
        Function retrieves urban_api tools from IDU MCP server.
        Returns:
            list[dict]: list of available Urban API Tools in ollama computable format.
        """

        return await self.load_ollama_tools(tags=["urban_api"])

    async def filter_tool_by_name(self, name: str) -> list[dict]:
        """
        Function filters tool by name.
        Args:
            name (str): tool name.
        Returns:
            list[dict]: Filtered tool in ollama computable format.
        """

        tools = await self.load_ollama_tools()
        return [tool for tool in tools if tool if tool["function"]["name"] == name]

    async def get_urban_api_service_tool(self) -> list[dict]:
        """
        Function returns Urban API GetServices tool.
        Returns:
            list[dict]: GetServices Urban API tool in ollama computable service.
        """

        return await self.filter_tool_by_name("GetServices")

    async def get_urban_api_physical_objects_tool(self) -> list[dict]:
        """
        Function returns Urban API GetPhysicalObjects tool.
        Returns:
            list[dict]: GetPhysicalObjects Urban API tool in ollama computable service.
        """

        return await self.filter_tool_by_name("GetPhysicalObjects")

    async def get_geometry_tools(self) -> list[dict]:
        """
        Function retrieves geometry tools from IDU MCP server.
        Returns:
            ListToolResult | list[Tool]: list of available Urban API Tools in ollama computable format.
        """

        return await self.load_ollama_tools(tags=["geometry"])

    async def get_create_buffer_tool(self) -> list[dict]:
        """
        Function retrieves geometry tool
        :return:
        """

    async def get_prompts_by_name(self, prompts_names: list[str]) -> list[dict]:
        """
        Function filters prompts from mcp server by name.
        Args:
            prompts_names (list[str]): prompts names to filter by
        Returns:
            list[dict]: filtered prompts from mcp server.
        """

        prompts = await self.get_prompts()
        return [
            prompt for prompt in prompts if prompt["name"] in prompts_names
        ]

    async def get_services_example_prompts(self) -> list[dict]:
        """
        Function returns  services example prompts
        Returns:
            list[dict]: list of service examples street
        """

        return await self.get_prompts_by_name(["GetServicesExample", "NoGetServicesExample"])


    async def get_physical_objects_example_prompts(self):
        """
        Function returns physical objects prompts examples.
        Returns:
             list[dict]: list of physical objects examples.
        """

        return await self.get_prompts_by_name(["GetPhysicalObjectsExample", "NoGetPhysicalObjectsExample"])
