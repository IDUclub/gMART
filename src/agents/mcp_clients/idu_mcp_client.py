from fastmcp import Client as McpClient
from mcp import GetPromptResult

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
        Function retrieves geometry tool create_buffer_tool.
        Returns:
            list[dict]: list of create buffer tool
        """

        return await self.load_ollama_tools(tags=["buffers"])

    async def get_create_restriction_tool(self) -> list[dict]:
        """
        Function retrieves geometry tool create_restrictions_tool.
        Returns:
            list[dict]: list of create buffer tool
        """

        return await self.load_ollama_tools(tags=["restrictions"])

    async def get_prompts_by_name(self, prompts_names: list[str], arguments: dict | None = None) -> list[GetPromptResult]:
        """
        Function filters prompts from mcp server by name.
        Args:
            prompts_names (list[str]): Prompts names to filter by
            arguments (dict | None): Key arguments as dict to pass to prompt. Default to None.
        Returns:
            list[dict]: filtered prompts from mcp server.
        """

        result = []
        async with self.mcp_client as client:
            for prompt_name in prompts_names:
                prompt = await client.get_prompt(prompt_name, arguments)
                result.append(prompt)
        return result

    async def get_services_example_prompts(self) -> list[dict]:
        """
        Function returns  services example prompts
        Returns:
            list[dict]: list of service examples street
        """

        result = await self.get_prompts_by_name(["GetServicesExample", "NoGetServicesExample"])
        messages_list = [res.model_dump()["messages"] for res in result]
        return [
            {"role": message["role"], "content": message["content"]["text"]} for messages in messages_list for message in messages
        ]

    async def get_physical_objects_example_prompts(self) -> list[dict]:
        """
        Function returns physical objects prompts examples.
        Returns:
             list[dict]: list of physical objects examples.
        """

        result = await self.get_prompts_by_name(["GetPhysicalObjectsExample", "NoGetPhysicalObjectsExample"])
        messages_list = [res.model_dump()["messages"] for res in result]
        return [
            {"role": message["role"], "content": message["content"]["text"]} for messages in messages_list for message in messages
        ]

    async def get_available_services_prompt(self, scenario_id: int) -> str:
        """
        Function returns available services prompt.
        Returns:
            str: prompt with available services.
        """

        result = await self.get_prompts_by_name(["GetAvailableServices"], {"scenario_id": scenario_id})
        return result[0].model_dump()["messages"][0]["content"]["text"]

    async def get_available_physical_objects_prompt(self, scenario_id: int) -> str:
        """
        Function returns available physical_objects for scenario.
        Returns:
            str: prompt with available services.
        """

        result = await self.get_prompts_by_name(["GetAvailablePhysicalObjects"], {"scenario_id": scenario_id})
        return result[0].model_dump()["messages"][0]["content"]["text"]
