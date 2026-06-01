from fastmcp import Client as McpClient
from pydantic import SecretStr

from src.agents.common.exceptions.token_exceptions import TokenExpiredError
from src.agents.mcp_clients.base_mcp_client import BaseMcpClient, _is_token_expired


class EffectsMcpClient(BaseMcpClient):
    def __init__(self, mcp_client: McpClient, mcp_url: str = "") -> None:
        super().__init__(mcp_client)
        self._mcp_url = mcp_url

    def update_token(self, new_token: str) -> None:
        """Replace the bearer token used for all subsequent MCP calls."""
        if self._mcp_url:
            self.mcp_client = McpClient(self._mcp_url, auth=new_token)
        else:
            try:
                self.mcp_client.transport.auth.token = SecretStr(new_token)
            except AttributeError:
                pass

    async def calculate_object_effects(
        self,
        service_type_id: int,
        scenario_id: int,
        target_population: int | None = None,
    ) -> dict:
        """
        Call CalculateObjectEffects on the effects MCP server.
        Args:
            service_type_id (int): Service type identifier.
            scenario_id (int): Scenario ID passed as tool argument.
            target_population (int | None): Optional population override.
        Returns:
            dict: Effects result with before_prove_data, after_prove_data, effects, pivot.
        """
        arguments: dict = {
            "service_type_id": service_type_id,
            "scenario_id": scenario_id,
        }
        if target_population is not None:
            arguments["target_population"] = target_population
        try:
            return await self.execute_tool("CalculateObjectEffects", arguments)
        except Exception as exc:
            if _is_token_expired(exc):
                raise TokenExpiredError(str(exc)) from exc
            raise
