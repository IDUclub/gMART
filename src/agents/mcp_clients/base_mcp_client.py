from dataclasses import asdict, is_dataclass

from fastmcp import Client as MCPClient
from loguru import logger
from mcp import ListToolsResult, Tool
from mcp.server.fastmcp.prompts import Prompt


class BaseMcpClient:

    def __init__(self, mcp_client: MCPClient):

        self.mcp_client: MCPClient = mcp_client

    @staticmethod
    async def __filter_tools_by_tag__(
        tools: ListToolsResult, tags: list[str]
    ) -> list[Tool]:
        """
        Function filters tools by
        Args:
            tools (ListToolsResult): List of available tools.
            tags (list[str]): Tags filter list.
        Returns:
            list[Tool]: Filtered tools list.
        """

        result_list = []
        for tag in tags:
            result_list.extend(
                [
                    tool
                    for tool in tools
                    if hasattr(tool, "meta")
                    and tool.meta
                    and tool.meta.get("fastmcp", {})
                    and tag in tool.meta.get("fastmcp", {}).get("tags", [])
                    and tool not in result_list
                ]
            )
        return list(result_list)

    async def load_ollama_tools(self, tags: list[str] | None = None) -> list[dict]:
        """
        Function returns available tools.
        Args:
            tags (list[str] | None): List of tags to filter tools by. Default to None.
        Returns:
            list[dict]: List of available tools filtered by tags in ollama computable format.
        """

        async with self.mcp_client as client:
            tools = await client.list_tools()
            if tags:
                tools = await self.__filter_tools_by_tag__(tools, tags)
            return [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                } for tool in tools
            ]

    async def get_prompts(self) -> list[Prompt] | list[dict]:
        """
        Function returns available for mcp server prompts templates.
        Returns:
            list[Prompt] | list[dict]: list of available prompts.
        """

        async with self.mcp_client:
            return await self.mcp_client.list_prompts()

    # TODO enhance exception handling
    async def execute_tool(self, tool_name: str, arguments: dict, meta: dict):

        try:
            async with self.mcp_client as mcp:
                result = await mcp.call_tool(tool_name, arguments, meta=meta)
                logger.info(f"Executed tool with meta {result.meta}")
                if is_dataclass(result.data):
                    return asdict(result.data)
                return result.data
        except Exception as e:
            logger.exception(e)
            raise
