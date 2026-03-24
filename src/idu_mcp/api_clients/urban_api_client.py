import asyncio

from watchfiles import awatch

from src.idu_mcp.common.api_handlers.json_api_handler import JsonApiHandler

LIVING_BUILDINGS_ID = 4


class UrbanApiClient:

    def __init__(self, json_handler: JsonApiHandler) -> None:
        self.json_handler = json_handler
        self.__name__ = "UrbanAPIClient"

    async def get_name_id(
        self, endpoint: str, names: list[str], token: str
    ) -> dict[str, int]:
        """
        Function retrieves name_id from endpoint asynchronously
        If name is not included in scenario it is skipped.
        Args:
            endpoint (str): endpoint url
            names (list[str]): list of names to retrieve
            token (str): auth bearer token
        Returns:
            dict[str, int]: dictionary with name as key and id as value
        """

        tasks = [
            self.json_handler.get(endpoint, params={"name": name}, auth_token=token)
            for name in names
        ]
        results = await asyncio.gather(*tasks)
        final_results = [res[0] for res in results if res]
        if "service" in endpoint:
            return {
                i["name"]: i["service_type_id"]
                for i in final_results
                if i and i["name"] in names
            }
        else:
            return {
                i["name"]: i["physical_object_type_id"]
                for i in final_results
                if i and i["name"] in names
            }

    async def get_service_name_id(self, names: list[str], token: str) -> dict[str, int]:
        """
        Function retrieves services name_id for scenario asynchronously.
        If service name is not included in scenario it is skipped.
        Args:
            names (list[str]): list of service names to retrieve
            token (str): auth bearer token
        Returns:
            dict[str, int]: dictionary with service name as key and id as value
        """

        return await self.get_name_id(f"api/v1/service_types", names, token)

    async def get_physical_objects_name_id(
        self, names: list[str], token: str
    ) -> dict[str, int]:
        """
        Function retrieves physical objects name_id for scenario asynchronously.
        If physical object name is not included in scenario it is skipped.
        Args:
            scenario_id (int): scenario id
            names (list[str]): list of physical objects names to retrieve
            token (str): auth bearer token
        Returns:
            dict[str, int]: dictionary with physical objects name as key and id as value
        """

        return await self.get_name_id(f"api/v1/physical_object_types", names, token)

    async def get_services(
        self, scenario_id: int, services: list[int], token: str
    ) -> list[dict]:
        """
        Function retrieves services for scenario asynchronously
        Args:
            scenario_id (int): scenario id
            services (list[int]): list of service ids to retrieve
            token (str): auth bearer token
        """

        tasks = [
            self.json_handler.get(
                f"api/v1/scenarios/{scenario_id}/services_with_geometry",
                params={"service_type_id": service_id},
                auth_token=token,
            )
            for service_id in services
        ]
        return await asyncio.gather(*tasks)

    async def get_physical_objects(
        self, scenario_id: int, physical_objects: list[int], token: str
    ):
        """
        Function retrieves services for scenario asynchronously
        Args:
            scenario_id (int): scenario id
            physical_objects (list[int]): list of service ids to retrieve
            token (str): auth bearer token
        """

        tasks = [
            self.json_handler.get(
                f"api/v1/scenarios/{scenario_id}/physical_objects_with_geometry",
                params={"physical_object_type_id": physical_object_id},
                auth_token=token,
            )
            for physical_object_id in physical_objects
        ]
        return await asyncio.gather(*tasks)

    async def get_available_scenario_services(self, scenario_id: int, token: str) -> list[str]:
        """
        Function returns list of available service types names.
        Args:
            scenario_id (int): Scenario ID from Urban API.
            token (str): Auth token.
        Returns:
            list[str]: List of available for scenario service type names.
        """

        service_types = await self.json_handler.get(
            f"api/v1/scenarios/{scenario_id}/service_types",
            auth_token=token
        )
        return [service_type["name"] for service_type in service_types]

    async def get_available_physical_objects(self, scenario_id: int, token: str) -> list[str]:
        """
        Function returns list of available physical objects types names.
        Args:
            scenario_id (int): Scenario ID from Urban API.
            token (str): Auth token.
        Returns:
            list[str]: List of available for scenario physical objects type names.
        """

        physical_objects_types = await self.json_handler.get(
            f"api/v1/scenarios/{scenario_id}/physical_object_types",
            auth_token=token
        )
        return [physical_object_type["name"] for physical_object_type in physical_objects_types]
