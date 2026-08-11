from geojson_pydantic import FeatureCollection
from loguru import logger

from src.idu_mcp.api_clients.urban_api_client import UrbanApiClient
from src.idu_mcp.tools_services.entites.object_type_enum import ObjectTypeEnum
from src.idu_mcp.tools_services.scenario_cache import ScenarioCache


class UrbanApiTool:
    """
    Class for communication with Urban API amd processing urban db data.
    Attributes:
        client (UrbanApiClient): client with async methods for resolving urban api requests
        cache (ScenarioCache): per-scenario response cache, disabled unless
            the SCENARIO_CACHE_SIZE env var is set
    """

    def __init__(self, urban_client: UrbanApiClient):
        """
        Initialization method for UrbanApiTool
        Args:
             urban_client (UrbanApiClient): urban api client with async methods for resolving urban api requests
        """
        self.client: UrbanApiClient = urban_client
        self.cache = ScenarioCache()

    async def get_entity_by_names(
        self,
        scenario_id: int,
        names: list[str],
        object_type: ObjectTypeEnum | str,
        token: str,
    ) -> dict[str, FeatureCollection]:
        """
        Method for getting all services with given names
        Args:
            scenario_id (int): scenario id
            names (list[str]): list of service names
            object_type (ObjectTypeEnum): object type. Possible values "SERVICE" ot "PHYSICAL_OBJECT"
            token (str): Auth Bearer token
        Returns:
            dict[str, FeatureCollection]: dict of all services for give scenario layers in 4326 crs with given names as keys
        """

        names = [i.capitalize() for i in names]
        # cache the raw payload rather than the models, so callers never share
        # (and cannot mutate) one another's FeatureCollection instances
        cache_key = (str(object_type), tuple(sorted(names)))
        cached = self.cache.get(scenario_id, cache_key)
        if cached is not None:
            object_name_id, objects = cached
        else:
            match object_type:
                case ObjectTypeEnum.SERVICE:
                    object_name_id = await self.client.get_service_name_id(names, token)
                    objects = await self.client.get_services(
                        scenario_id, list(object_name_id.values()), token
                    )
                case ObjectTypeEnum.PHYSICAL_OBJECT:
                    object_name_id = await self.client.get_physical_objects_name_id(
                        names, token
                    )
                    objects = await self.client.get_physical_objects(
                        scenario_id, list(object_name_id.values()), token
                    )
                case _:
                    logger.info(
                        f"Unknown object type {object_type}\nfor scenario {scenario_id}\nand names {names}"
                    )
                    raise ValueError("Unsupported object type")
            self.cache.set(scenario_id, cache_key, (object_name_id, objects))
        existing_names = list(object_name_id)
        return {
            existing_names[i]: FeatureCollection(**objects[i])
            for i in range(len(existing_names))
        }

    async def get_entity_id_by_name(
        self,
        service_name: str,
        token: str,
    ) -> int | None:
        """
        Method for retrieving service type id by name.
        Args:
            service_name (str): Service name.
            token (str): Auth Bearer token.
        Returns:
            int | None: Service type id. None if service not found.
        """

        service_name_id = await self.client.get_service_name_id(
            [service_name.capitalize()], token
        )
        return service_name_id.get(service_name.capitalize())
