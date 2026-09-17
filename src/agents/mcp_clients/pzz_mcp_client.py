from typing import Any

from src.agents.common.exceptions.token_exceptions import TokenExpiredError
from src.agents.mcp_clients.base_mcp_client import BaseMcpClient


class PzzMcpClient(BaseMcpClient):
    """PZZ task tools use the same authenticated FastMCP transport as other agents."""

    api_client = None

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        for key, value in arguments.items():
            if key.endswith("_upload_id") and value:
                if self.api_client is None:
                    raise ValueError("PZZ_API_URL is required for uploaded files")
                await self.api_client.validate_upload(value)
        result = await self.execute_tool(name, arguments)
        if not isinstance(result, dict):
            raise ValueError(f"PZZ tool {name} returned an invalid response")
        if "AUTH_TOKEN_EXPIRED" in str(result.get("error", "")):
            raise TokenExpiredError("PZZ authentication expired")
        return result
