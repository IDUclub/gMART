from fastmcp import Client as MCPClient
from loguru import logger
from mcp import ListToolsResult, Tool
from mcp.server.fastmcp.prompts import Prompt

from src.agents.common.exceptions.token_exceptions import TokenExpiredError


def _is_token_expired(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "401" in msg
        or "unauthorized" in msg
        or "token expired" in msg
        or "token_expired" in msg
        or "authentication" in msg
    )


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
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in tools
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
    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict,
        meta: dict | None = None,
        log: bool = False,
    ):
        try:
            async with self.mcp_client as mcp:
                result = await mcp.call_tool(tool_name, arguments, meta=meta or {})
                if log:
                    logger.info(
                        f"Executed tool with meta {result.meta} and data {result.data}"
                    )
                return result.data
        except Exception as e:
            if _is_token_expired(e):
                raise TokenExpiredError(str(e)) from e
            logger.exception(e)
            raise
