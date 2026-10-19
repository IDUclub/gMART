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

    async def get_scenario_service_types(
        self, token: str, scenario_id: int
    ) -> dict[str, int]:
        """
        Function retrieves service types available for a scenario.
        Args:
            token (str): Authorization token.
            scenario_id (int): Scenario ID from Urban API.
        Returns:
            dict[str, int]: Mapping of service type name to service_type_id.
        """

        service_types = await self.json_handler.get(
            f"/v1/scenarios/{scenario_id}/service_types", auth_token=token
        )
        result: dict[str, int] = {}
        for service_type in service_types or []:
            name = service_type.get("name")
            type_id = service_type.get("service_type_id", service_type.get("id"))
            if name and type_id is not None:
                result[name] = int(type_id)
        return result

    # --- dictionaries and territories ------------------------------------------

    async def get_physical_object_types(self, token: str) -> list[dict]:
        return await self.json_handler.get(
            "/v1/physical_object_types", auth_token=token
        )

    async def get_service_types(self, token: str) -> list[dict]:
        data = await self.json_handler.get("/v1/service_types", auth_token=token)
        return (data.get("results") or []) if isinstance(data, dict) else data

    async def get_functional_zone_types(self, token: str) -> list[dict]:
        return await self.json_handler.get(
            "/v1/functional_zones_types", auth_token=token
        )

    async def get_root_territories(self, token: str) -> list[dict]:
        """Top-level territories (the country) without geometry."""

        data = await self.json_handler.get(
            "/v1/territories_without_geometry",
            params={"page_size": 100},
            auth_token=token,
        )
        return (data.get("results") or []) if isinstance(data, dict) else data

    async def get_child_territories(self, token: str, parent_id: int) -> list[dict]:
        """Territories one level below ``parent_id``, without geometry."""

        return await self.json_handler.get(
            "/v1/all_territories_without_geometry",
            params={"parent_id": int(parent_id), "get_all_levels": False},
            auth_token=token,
        )

    async def get_intersecting_territories(
        self, token: str, parent_id: int, geometry: dict
    ) -> list[dict]:
        """Children of ``parent_id`` (one level down) intersecting ``geometry``."""

        return await self.json_handler.post(
            f"/v1/territory/{int(parent_id)}/intersecting_territories",
            auth_token=token,
            data=geometry,
        )

    async def get_physical_objects_around(
        self, token: str, geometry: dict
    ) -> list[dict]:
        """Physical objects of the city data within ``geometry`` plus 50 m."""

        return await self.json_handler.post(
            "/v1/physical_objects/around", auth_token=token, data=geometry
        )

    # --- projects and scenarios -----------------------------------------------
    # Writes are sent once: a repeated POST after a lost response would create a
    # second project or object.

    async def create_project(
        self, token: str, project: dict, user_id: str | None = None
    ) -> dict:
        """Create a project with its territory; Urban API adds the base scenario."""

        return await self.json_handler.post(
            "/v1/projects",
            auth_token=token,
            params={"user_id": user_id} if user_id else None,
            data=project,
            attempts=1,
        )

    async def get_project(self, token: str, project_id: int) -> dict:
        return await self.json_handler.get(
            f"/v1/projects/{int(project_id)}", auth_token=token
        )

    async def get_project_scenarios(self, token: str, project_id: int) -> list[dict]:
        return await self.json_handler.get(
            f"/v1/projects/{int(project_id)}/scenarios", auth_token=token
        )

    async def copy_scenario(self, token: str, scenario_id: int, scenario: dict) -> dict:
        """A new scenario of the project as a copy of ``scenario_id``."""

        return await self.json_handler.post(
            f"/v1/scenarios/{int(scenario_id)}",
            auth_token=token,
            data=scenario,
            attempts=1,
        )

    async def get_scenario_physical_objects(self, token: str, scenario_id: int) -> dict:
        """The scenario's physical objects as a GeoJSON FeatureCollection."""

        return await self.json_handler.get(
            f"/v1/scenarios/{int(scenario_id)}/physical_objects_with_geometry",
            auth_token=token,
        )

    async def delete_scenario_physical_object(
        self,
        token: str,
        scenario_id: int,
        physical_object_id: int,
        is_scenario_object: bool,
    ) -> None:
        await self.json_handler.delete(
            f"/v1/scenarios/{int(scenario_id)}/physical_objects/"
            f"{int(physical_object_id)}",
            auth_token=token,
            params={"is_scenario_object": is_scenario_object},
        )

    async def delete_scenario_functional_zones(
        self, token: str, scenario_id: int
    ) -> None:
        await self.json_handler.delete(
            f"/v1/scenarios/{int(scenario_id)}/functional_zones", auth_token=token
        )

    async def add_scenario_functional_zones(
        self, token: str, scenario_id: int, zones: list[dict]
    ) -> list[dict]:
        """Add zones; Urban API first deletes the scenario's user zones."""

        return await self.json_handler.post(
            f"/v1/scenarios/{int(scenario_id)}/functional_zones",
            auth_token=token,
            data=zones,
            attempts=1,
        )

    async def add_scenario_physical_object(
        self, token: str, scenario_id: int, physical_object: dict
    ) -> dict:
        return await self.json_handler.post(
            f"/v1/scenarios/{int(scenario_id)}/physical_objects",
            auth_token=token,
            data=physical_object,
            attempts=1,
        )

    async def add_scenario_building(
        self, token: str, scenario_id: int, building: dict
    ) -> dict:
        return await self.json_handler.post(
            f"/v1/scenarios/{int(scenario_id)}/buildings",
            auth_token=token,
            data=building,
            attempts=1,
        )

    async def patch_scenario_building(
        self, token: str, scenario_id: int, building_id: int, building: dict
    ) -> dict:
        return await self.json_handler.patch(
            f"/v1/scenarios/{int(scenario_id)}/buildings/{int(building_id)}",
            auth_token=token,
            params={"is_scenario_object": True},
            data=building,
        )

    async def add_scenario_service(
        self, token: str, scenario_id: int, service: dict
    ) -> dict:
        return await self.json_handler.post(
            f"/v1/scenarios/{int(scenario_id)}/services",
            auth_token=token,
            data=service,
            attempts=1,
        )
