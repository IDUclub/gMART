"""
Module aimed to handle requests to Urban API REST service.
"""

from src.agents.common.api_handlers.json_api_handler import JsonApiHandler


class UrbanApiClient:
    def __init__(self, json_handler: JsonApiHandler) -> None:
        self.json_handler = json_handler
        self.__name__ = "UrbanAPIClient"

    async def get_project_by_scenario(self, token: str, scenario_id: int) -> int:
        """
        Function retrieves project_id by scenario_id.
        Args:
            token (str): Authorization token.
            scenario_id (int): Scenario ID from Urban API.
        Returns:
            Int: Project ID from Urban API.
        """

        result = await self.json_handler.get(
            f"/v1/scenarios/{scenario_id}", auth_token=token
        )
        return result["project"]["project_id"]
