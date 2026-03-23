from fastmcp import Client as MCPClient
from loguru import logger
from mcp import ListToolsResult, Tool


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
                    and tool.meta.get("_fastmcp", {})
                    and tag in tool.meta.get("_fastmcp", {}).get("tags", [])
                ]
            )
        return list(set(result_list))

    async def load_tools(self, tags: list[str]) -> list[Tool] | ListToolsResult:
        """
        Function returns available tools.
        Args:
            tags (list[str]): List of tags to filter tools by.
        Returns:
            ListToolsResult: List of available tools filtered by tags.
        """

        async with self.mcp_client as client:
            tools = await client.list_tools()
            if tags:
                return await self.__filter_tools_by_tag__(tools, tags)
            return tools

    # TODO enhance exception handling
    async def execute_tool(self, tool_name: str, arguments: dict, meta: dict):

        try:
            async with self.mcp_client as mcp:
                result = await mcp.call_tool(tool_name, arguments, meta=meta)
                logger.info(f"Executed tool with meta {result.meta}")
                return result.data
        except Exception as e:
            logger.exception(e)
            raise
