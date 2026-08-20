from geojson_pydantic import FeatureCollection
from loguru import logger

from src.idu_mcp.api_clients.urban_api_client import UrbanApiClient
from src.idu_mcp.tools_services.entites.object_type_enum import ObjectTypeEnum


class UrbanApiTool:
    """
    Class for communication with Urban API amd processing urban db data.
    Attributes:
        client (UrbanApiClient): client with async methods for resolving urban api requests
    """

    def __init__(self, urban_client: UrbanApiClient):
        """
        Initialization method for UrbanApiTool
        Args:
             urban_client (UrbanApiClient): urban api client with async methods for resolving urban api requests
        """
        self.client: UrbanApiClient = urban_client

    async def get_entity_by_names(
        self,
        scenario_id: int,
        names: list[str],
        object_type: ObjectTypeEnum | str,
        token: str,
    ) -> dict[str, dict]:
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
        existing_names = list(object_name_id)
        result: dict[str, dict] = {}
        for index, name in enumerate(existing_names):
            collection = FeatureCollection(**objects[index]).model_dump(mode="json")
            collection["meta"] = {
                "complete": True,
                "truncated": False,
                "revision": (
                    f"scenario:{scenario_id}:{str(object_type).lower()}:{object_name_id[name]}"
                ),
            }
            result[name] = collection
        return result

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

    async def get_functional_zones(
        self,
        scenario_id: int,
        token: str,
        *,
        source: str | None = None,
        year: int | None = None,
        zone_type_names: list[str] | None = None,
    ) -> dict[str, dict]:
        """Resolve a source/year and return a bounded functional-zone layer."""

        sources = await self.client.get_functional_zone_sources(scenario_id, token)
        candidates = [
            item
            for item in sources
            if (
                source is None
                or str(item.get("source") or item.get("name", "")).casefold()
                == source.casefold()
            )
            and (year is None or int(item.get("year", -1)) == year)
        ]
        if not candidates:
            return {}
        selected = max(candidates, key=lambda item: int(item.get("year", 0)))
        selected_source = str(selected.get("source") or selected.get("name") or "")
        selected_year = int(selected["year"])
        collection = await self.client.get_functional_zones(
            scenario_id,
            source=selected_source,
            year=selected_year,
            token=token,
        )
        requested = {name.casefold() for name in (zone_type_names or [])}
        if requested:

            def zone_type_name(feature: dict) -> str:
                properties = feature.get("properties") or {}
                value = properties.get("functional_zone_type")
                if isinstance(value, dict):
                    value = value.get("name")
                return str(value or properties.get("functional_zone_type_name") or "")

            collection = {
                **collection,
                "features": [
                    feature
                    for feature in collection.get("features", [])
                    if zone_type_name(feature).casefold() in requested
                ],
            }
        collection["meta"] = {
            "complete": True,
            "truncated": False,
            "source": selected_source,
            "year": selected_year,
            "revision": f"scenario:{scenario_id}:functional_zones:{selected_source}:{selected_year}",
        }
        return {"functional_zones": collection}
