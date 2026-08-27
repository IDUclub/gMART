"""Unit tests for the in-process IDU tool client.

The local client replaces the MCP transport for experiment runs, so what matters
is that it is the *same* tool boundary and not a looser one: identical dispatch,
identical validation, identical catalog prompt wording. If it were looser, the
local arm would score better than the HTTP arm for reasons that have nothing to
do with the model, and the transport comparison the paper draws would be void.
"""

from __future__ import annotations

import pytest
from fastmcp.exceptions import ToolError

from src.agents.mcp_clients.local_idu_mcp_client import LocalIduMcpClient
from src.agents.services.restriction_catalog import parse_catalog_prompt


class FakeUrbanApiTool:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result if result is not None else {"школа": {"features": []}}
        self.error = error
        self.calls: list[tuple] = []

    async def get_entity_by_names(self, scenario_id, names, object_type, token):
        self.calls.append((scenario_id, list(names), str(object_type), token))
        if self.error:
            raise self.error
        return self.result


class FakeGeomTools:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.buffer_calls: list[tuple] = []
        self.restriction_calls: list[tuple] = []

    async def async_generate_geometry_buffers(self, buffer_info, objects):
        self.buffer_calls.append((buffer_info, objects))
        if self.error:
            raise self.error
        return {name: {"type": "FeatureCollection", "features": []} for name in objects}

    async def async_create_restrictions(
        self, layers, generators, objects, restrictions
    ):
        self.restriction_calls.append((layers, generators, objects, restrictions))
        if self.error:
            raise self.error
        return {
            "objects": {"type": "FeatureCollection", "features": []},
            "generators": {"type": "FeatureCollection", "features": []},
        }


def _client(urban=None, geom=None) -> LocalIduMcpClient:
    return LocalIduMcpClient(
        "token-abc",
        urban_api_url="http://urban",
        urban_api_tools=urban or FakeUrbanApiTool(),
        geom_tools=geom or FakeGeomTools(),
    )


def _feature_collection(count: int = 1) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Point", "coordinates": [30.0, 60.0]},
            }
        ]
        * count,
    }


# --------------------------------------------------------------------------- #
# token
# --------------------------------------------------------------------------- #
def test_current_token_and_update():
    client = _client()

    assert client.current_token() == "token-abc"
    client.update_token("token-xyz")
    assert client.current_token() == "token-xyz"


async def test_token_is_passed_to_every_urban_api_call():
    urban = FakeUrbanApiTool()
    client = _client(urban=urban)
    client.update_token("fresh")

    await client.execute_tool(
        "GetServices", {"services_names": ["школа"], "scenario_id": 7}
    )

    assert urban.calls[0][3] == "fresh"


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
async def test_get_services_dispatches_with_the_service_object_type():
    urban = FakeUrbanApiTool()
    client = _client(urban=urban)

    await client.execute_tool(
        "GetServices", {"services_names": ["школа", "парк"], "scenario_id": 7}
    )

    scenario_id, names, object_type, _ = urban.calls[0]
    assert (scenario_id, names) == (7, ["школа", "парк"])
    assert "SERVICE" in object_type.upper()


async def test_get_physical_objects_dispatches_with_the_physical_object_type():
    urban = FakeUrbanApiTool()
    client = _client(urban=urban)

    await client.execute_tool(
        "GetPhysicalObjects",
        {"physical_objects_names": ["жилой дом"], "scenario_id": 7},
    )

    _, names, object_type, _ = urban.calls[0]
    assert names == ["жилой дом"]
    assert "PHYSICAL" in object_type.upper()


async def test_empty_name_list_short_circuits_without_calling_urban_api():
    """A plan with no services of its own must not fetch the whole scenario."""

    urban = FakeUrbanApiTool()
    client = _client(urban=urban)

    assert (
        await client.execute_tool(
            "GetServices", {"services_names": [], "scenario_id": 7}
        )
        == {}
    )
    assert urban.calls == []


async def test_unknown_tool_is_a_tool_error():
    client = _client()

    with pytest.raises(ToolError):
        await client.execute_tool("NoSuchTool", {})


async def test_every_call_is_recorded_for_the_run_record():
    client = _client()

    await client.execute_tool(
        "GetServices", {"services_names": ["школа"], "scenario_id": 7}
    )
    await client.execute_tool(
        "CreateBuffers",
        {
            "buffer_info": {
                "школа": {"buffer_size": 50, "buffer_type": "round", "title": "t"}
            },
            "objects": {"школа": _feature_collection()},
        },
    )

    assert [call["tool"] for call in client.calls] == ["GetServices", "CreateBuffers"]


# --------------------------------------------------------------------------- #
# the tool boundary is kept
# --------------------------------------------------------------------------- #
async def test_buffer_validation_runs_exactly_as_in_the_mcp_wrapper():
    """buffer_info naming a layer that objects does not carry must be rejected."""

    geom = FakeGeomTools()
    client = _client(geom=geom)

    with pytest.raises(ToolError):
        await client.execute_tool(
            "CreateBuffers",
            {
                "buffer_info": {
                    "школа": {"buffer_size": 50, "buffer_type": "round", "title": "t"}
                },
                "objects": {"парк": _feature_collection()},
            },
        )
    assert geom.buffer_calls == []


async def test_restriction_validation_runs_before_the_geometry():
    geom = FakeGeomTools()
    client = _client(geom=geom)

    with pytest.raises(ToolError):
        await client.execute_tool(
            "CreateRestrictions",
            {
                "generators": ["буфер школа"],
                "objects": ["жилой дом"],
                "restrictions": {},
                "layers": {},  # neither generator nor object layer present
            },
        )
    assert geom.restriction_calls == []


async def test_geometry_runtime_failure_becomes_a_tool_error():
    geom = FakeGeomTools(error=ValueError("bad geometry"))
    client = _client(geom=geom)

    with pytest.raises(ToolError) as excinfo:
        await client.execute_tool(
            "CreateBuffers",
            {
                "buffer_info": {
                    "школа": {"buffer_size": 50, "buffer_type": "round", "title": "t"}
                },
                "objects": {"школа": _feature_collection()},
            },
        )
    assert "bad geometry" in str(excinfo.value)


async def test_urban_api_failures_propagate_unchanged():
    """A data-layer failure must keep its own type so the runner can class it.

    Wrapping it in a ToolError here would make an Urban API outage or an offline
    gap indistinguishable from a geometry-tool failure in the taxonomy.
    """

    class Boom(RuntimeError):
        pass

    urban = FakeUrbanApiTool(error=Boom("urban down"))
    client = _client(urban=urban)

    with pytest.raises(Boom):
        await client.execute_tool(
            "GetServices", {"services_names": ["школа"], "scenario_id": 7}
        )


# --------------------------------------------------------------------------- #
# catalog prompts
# --------------------------------------------------------------------------- #
class FakeUrbanApiClient:
    def __init__(self, services: list[str], physical: list[str]) -> None:
        self.services = services
        self.physical = physical

    async def get_available_scenario_services(self, scenario_id, token):
        return self.services

    async def get_available_physical_objects(self, scenario_id, token):
        return self.physical


async def test_catalog_prompts_parse_back_to_the_catalog():
    """The wording matters: parse_catalog_prompt splits on ':' then on ','."""

    client = _client()
    client.urban_api_client = FakeUrbanApiClient(
        services=["Школа", "Поликлиника"], physical=["Жилой дом"]
    )

    services_prompt = await client.get_available_services_prompt(7)
    physical_prompt = await client.get_available_physical_objects_prompt(7)

    assert parse_catalog_prompt(services_prompt) == ["школа", "поликлиника"]
    assert parse_catalog_prompt(physical_prompt) == ["жилой дом"]
    assert services_prompt.startswith("Список сервисов:")
    assert physical_prompt.startswith("Список физических объектов:")


async def test_empty_catalog_yields_an_empty_list_not_a_blank_entry():
    client = _client()
    client.urban_api_client = FakeUrbanApiClient(services=[], physical=[])

    assert parse_catalog_prompt(await client.get_available_services_prompt(7)) == []
