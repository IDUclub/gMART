"""Urban API writes of the project creation: endpoints, owner, single attempts."""

from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient


class Handler:
    def __init__(self, response=None):
        self.calls: list[tuple] = []
        self.response = response if response is not None else {}

    async def get(self, endpoint, **kwargs):
        self.calls.append(("get", endpoint, kwargs))
        return self.response

    async def post(self, endpoint, **kwargs):
        self.calls.append(("post", endpoint, kwargs))
        return self.response

    async def patch(self, endpoint, **kwargs):
        self.calls.append(("patch", endpoint, kwargs))
        return self.response

    async def delete(self, endpoint, **kwargs):
        self.calls.append(("delete", endpoint, kwargs))
        return None


async def test_project_is_created_once_for_its_owner():
    handler = Handler({"project_id": 1})

    await UrbanApiClient(handler).create_project("t", {"name": "Сад"}, user_id="alice")

    method, endpoint, kwargs = handler.calls[0]
    assert (method, endpoint) == ("post", "/v1/projects")
    assert kwargs["params"] == {"user_id": "alice"}
    assert kwargs["attempts"] == 1
    assert kwargs["data"] == {"name": "Сад"}


async def test_scenario_objects_are_written_once():
    handler = Handler()
    client = UrbanApiClient(handler)

    await client.copy_scenario("t", 11, {"name": "s"})
    await client.add_scenario_functional_zones("t", 12, [{"year": 2026}])
    await client.add_scenario_physical_object("t", 12, {"geometry": {}})
    await client.add_scenario_building("t", 12, {"physical_object_id": 1})
    await client.add_scenario_service("t", 12, {"service_type_id": 1})

    assert [(call[1], call[2]["attempts"]) for call in handler.calls] == [
        ("/v1/scenarios/11", 1),
        ("/v1/scenarios/12/functional_zones", 1),
        ("/v1/scenarios/12/physical_objects", 1),
        ("/v1/scenarios/12/buildings", 1),
        ("/v1/scenarios/12/services", 1),
    ]


async def test_deletes_and_building_patch_name_scenario_objects():
    handler = Handler()
    client = UrbanApiClient(handler)

    await client.delete_scenario_physical_object("t", 12, 5, False)
    await client.delete_scenario_functional_zones("t", 12)
    await client.patch_scenario_building("t", 12, 7, {"floors": 9})

    assert handler.calls[0][1:] == (
        "/v1/scenarios/12/physical_objects/5",
        {"auth_token": "t", "params": {"is_scenario_object": False}},
    )
    assert handler.calls[1][1] == "/v1/scenarios/12/functional_zones"
    assert handler.calls[2][2]["params"] == {"is_scenario_object": True}


async def test_paginated_dictionaries_are_unwrapped():
    handler = Handler({"results": [{"service_type_id": 1, "name": "Школа"}]})

    services = await UrbanApiClient(handler).get_service_types("t")

    assert services == [{"service_type_id": 1, "name": "Школа"}]
