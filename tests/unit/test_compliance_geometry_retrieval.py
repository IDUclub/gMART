"""Full-geometry requests survive the public tool -> Urban API client chain."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.idu_mcp.api_clients.urban_api_client import UrbanApiClient
from src.idu_mcp.tools_interfaces.urb_api_interface import (
    get_physical_objects_by_name,
    get_services_by_name,
)
from src.idu_mcp.tools_services.urb_api_tools import UrbanApiTool


@pytest.mark.parametrize("service", [True, False])
@pytest.mark.parametrize("centers_only", [False, True])
async def test_public_geometry_tool_forwards_flag_to_http_and_preserves_polygons(
    service, centers_only
):
    name = "Школа" if service else "Жилой дом"
    type_key = "service_type_id" if service else "physical_object_type_id"
    geometry = {
        "type": "Polygon",
        "coordinates": [[[30, 60], [30.01, 60], [30.01, 60.01], [30, 60]]],
    }

    async def get(endpoint, *, params=None, auth_token=None):
        assert auth_token == "user"
        if endpoint in ("v1/service_types", "v1/physical_object_types"):
            return [{"name": name, type_key: 4}]
        assert (
            endpoint
            == f"v1/scenarios/772/{'services' if service else 'physical_objects'}_with_geometry"
        )
        assert params == {type_key: 4, "centers_only": centers_only}
        return {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": geometry, "properties": {}}],
        }

    handler = SimpleNamespace(get=AsyncMock(side_effect=get))
    tools = UrbanApiTool(UrbanApiClient(handler))
    common = dict(
        scenario_id=772,
        centers_only=centers_only,
        user_id="user",
        urban_api_tools=tools,
    )
    if service:
        result = await get_services_by_name(services_names=[name], **common)
    else:
        result = await get_physical_objects_by_name(
            physical_objects_names=[name], **common
        )
    assert result[name]["features"][0]["geometry"] == geometry
    assert result[name]["meta"]["revision"].endswith(
        f"centers_only={str(centers_only).lower()}"
    )


@pytest.mark.parametrize("method", ["get_services", "get_physical_objects"])
async def test_client_explicitly_requests_full_geometry_when_flag_omitted(method):
    handler = SimpleNamespace(
        get=AsyncMock(return_value={"type": "FeatureCollection", "features": []})
    )
    await getattr(UrbanApiClient(handler), method)(772, [4], "user")
    assert handler.get.await_args.kwargs["params"]["centers_only"] is False
