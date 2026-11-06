"""Hermetic by_geometry queries: real HTTP adapter, mocked aiohttp boundary."""

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from fastmcp.exceptions import ToolError
from shapely.geometry import Point, Polygon, box, mapping, shape
from shapely.ops import unary_union

from src.idu_mcp.api_clients.urban_api_client import UrbanApiClient
from src.idu_mcp.common.api_handlers.json_api_handler import JsonApiHandler
from src.idu_mcp.tools_services.services_by_geometry import ServicesByGeometry
from src.idu_mcp.tools_services.spatial_predicates import filter_objects_by_zones


def fc(*geometries):
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": mapping(geometry), "properties": {}}
            for geometry in geometries
        ],
    }


def service(service_id, geometry, type_id=1):
    return {
        "type": "Feature",
        "geometry": mapping(geometry),
        "properties": {
            "service_id": service_id,
            "service_type_id": type_id,
            "service_type_name": f"Тип {type_id}",
            "name": f"Сервис {service_id}",
            "capacity": 100,
            "is_capacity_real": True,
            "object_geometry_id": 700 + service_id,
            "is_scenario_service": True,
            "is_scenario_geometry": False,
            "is_locked": False,
        },
    }


class Response:
    def __init__(self, status=200, data=None, text=""):
        self.status, self.data, self.body = status, data, text
        self.content_type = "application/json"

    async def json(self, **kwargs):
        return self.data

    async def text(self):
        return self.body


class Request:
    def __init__(self, session, outcome, track=True):
        self.session, self.outcome, self.track = session, outcome, track

    async def __aenter__(self):
        self.session.active += int(self.track)
        self.session.maximum = max(self.session.maximum, self.session.active)
        await asyncio.sleep(0)
        if isinstance(self.outcome, Exception):
            self.session.active -= int(self.track)
            raise self.outcome
        return self.outcome

    async def __aexit__(self, *args):
        self.session.active -= int(self.track)


class Session:
    def __init__(self, responder, names=None):
        self.responder = responder
        self.names = names or {"Школа": 1, "Театр": 2}
        self.calls = []
        self.active = self.maximum = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def get(self, *, url, headers, params):
        name = params["name"]
        return Request(
            self,
            Response(
                data=(
                    [{"name": name, "service_type_id": self.names[name]}]
                    if name in self.names
                    else []
                )
            ),
            track=False,
        )

    def post(self, *, url, headers, params, json):
        self.calls.append(
            {
                "url": url,
                "headers": deepcopy(headers),
                "params": deepcopy(params),
                "body": deepcopy(json),
            }
        )
        return Request(self, self.responder(self.calls[-1], len(self.calls)))


def search(monkeypatch, responder, names=None, service_auth=None):
    session = Session(responder, names)
    monkeypatch.setattr(
        "src.idu_mcp.common.api_handlers.json_api_handler.aiohttp.ClientSession",
        lambda: session,
    )
    handler = JsonApiHandler("http://urban/api", service_auth=service_auth)
    return ServicesByGeometry(UrbanApiClient(handler)), session


def arguments(**kwargs):
    return {
        "scenario_id": 42,
        "services_names": ["школа", "театр"],
        "zones_layers": ["a", "b"],
        "layer_name": "services",
        "token": "user-token",
        "layers": {
            "a": fc(box(30, 60, 30.01, 60.01)),
            "b": fc(box(30.02, 60, 30.03, 60.01)),
        },
        **kwargs,
    }


async def test_exact_union_one_post_per_type_and_auth_and_properties(monkeypatch):
    from src.idu_mcp.tools_interfaces.urb_api_interface import get_services_by_geometry
    from src.idu_mcp.tools_services.urb_api_tools import UrbanApiTool

    def respond(call, number):
        type_id = call["params"]["service_type_id"]
        return Response(
            data={
                "type": "FeatureCollection",
                "features": [service(type_id, Point(30.005, 60.005), type_id)],
            }
        )

    tool, session = search(monkeypatch, respond)
    args = arguments(physical_object_type_ids=[4])
    token = args.pop("token")
    result = (
        await get_services_by_geometry(
            **args, user_id=token, urban_api_tools=UrbanApiTool(tool.client)
        )
    )["services"]
    assert len(session.calls) == 2
    expected = unary_union([box(30, 60, 30.01, 60.01), box(30.02, 60, 30.03, 60.01)])
    for call in session.calls:
        assert shape(call["body"]).equals(expected)
        assert call["params"] == {
            "service_type_id": call["params"]["service_type_id"],
            "intersection_type": "intersects",
        }
        assert call["headers"]["Authorization"] == "Bearer user-token"
        assert (
            call["url"]
            == "http://urban/api/v1/scenarios/42/services/by_geometry/geojson"
        )
    props = result["features"][0]["properties"]
    assert props["object_ref"]["id"] == "service/1/geometry/701"
    assert props["object_ref"]["namespace"] == "service"
    assert props["capacity"] == 100
    assert props["service_type"] == {"id": 1, "name": "Тип 1"}
    assert result["meta"]["mode"] == "single"
    assert result["meta"]["calls_made"] == 2
    assert result["meta"]["component_count"] == 2
    assert result["meta"]["per_type_counts"] == {"1": 1, "2": 1}


async def test_arbitrary_polygon_holes_and_union_are_not_simplified(monkeypatch):
    tool, session = search(monkeypatch, lambda *_: Response(data=fc()))
    area = Polygon(
        [
            (30, 60),
            (30.04, 60),
            (30.04, 60.01),
            (30.02, 60.01),
            (30.02, 60.04),
            (30, 60.04),
        ],
        holes=[
            [(30.001, 60.001), (30.009, 60.001), (30.009, 60.009), (30.001, 60.009)]
        ],
    )
    args = arguments(layers={"a": fc(area), "b": fc()}, services_names=["Школа"])
    await tool.query(**args)
    sent = shape(session.calls[0]["body"])
    assert sent.equals(area)
    assert len(sent.exterior.coords) == len(area.exterior.coords)
    assert len(sent.interiors) == 1


async def test_concurrency_bound_is_shared_between_queries(monkeypatch):
    monkeypatch.setenv("COMPLIANCE_BY_GEOMETRY_CONCURRENCY", "3")
    names = {f"Тип {index}": index for index in range(12)}
    tool, session = search(monkeypatch, lambda *_: Response(data=fc()), names)
    args = arguments(services_names=list(names))
    await asyncio.gather(tool.query(**args), tool.query(**args))
    assert len(session.calls) == 24
    assert session.maximum <= 3
    assert session.maximum > 1


async def test_bounded_exponential_backoff_counts_http_attempts(monkeypatch):
    outcomes = [
        ConnectionResetError("reset"),
        Response(500, {"error": "reset by peer"}),
        Response(data=fc()),
    ]
    tool, session = search(monkeypatch, lambda *_: outcomes.pop(0))
    sleep = AsyncMock()
    # Request's scheduling yield also calls sleep(0), so inspect only backoff delays.
    monkeypatch.setattr(
        "src.idu_mcp.common.api_handlers.json_api_handler.asyncio.sleep", sleep
    )
    result = (await tool.query(**arguments(services_names=["Школа"])))["services"]
    assert [call.args[0] for call in sleep.await_args_list if call.args[0]] == [
        0.5,
        1.0,
    ]
    assert len(session.calls) == result["meta"]["calls_made"] == 3


async def test_retry_exhaustion_is_bounded(monkeypatch):
    tool, session = search(monkeypatch, lambda *_: ConnectionResetError("reset"))
    monkeypatch.setattr(
        "src.idu_mcp.common.api_handlers.json_api_handler.asyncio.sleep", AsyncMock()
    )
    with pytest.raises(ToolError, match="3 attempts"):
        await tool.query(**arguments(services_names=["Школа"]))
    assert len(session.calls) == 3


@pytest.mark.parametrize(
    "status,reason", [(413, "payload"), (422, "too many coordinates")]
)
async def test_split_components_deduplicates_and_records_meta(
    monkeypatch, status, reason
):
    def respond(call, number):
        if number == 1:
            return Response(status, text=reason)
        return Response(
            data={
                "type": "FeatureCollection",
                "features": [
                    service(1, Point(30.005, 60.005)),
                    service(1, Point(30.005, 60.005)),
                    service(2, Point(30.025, 60.005)),
                ],
            }
        )

    tool, session = search(monkeypatch, respond)
    result = (await tool.query(**arguments(services_names=["Школа"])))["services"]
    assert len(session.calls) == 3
    assert [f["properties"]["service_id"] for f in result["features"]] == [1, 2]
    assert result["meta"]["calls_made"] == 3
    assert result["meta"]["mode"] == "split"
    assert result["meta"]["part_count"] == 2
    assert result["meta"]["per_type_counts"] == {"1": 2}


async def test_single_component_quadrants_are_lossless_and_within_is_local(monkeypatch):
    area = box(30, 60, 30.04, 60.04)
    crosses_cut = box(30.01, 60.01, 30.03, 60.03)
    crosses_outer = box(29.99, 60.01, 30.01, 60.02)

    def respond(call, number):
        if number == 1:
            return Response(413, text="too large")
        assert call["params"]["intersection_type"] == "intersects"
        return Response(
            data={
                "type": "FeatureCollection",
                "features": [
                    service(1, crosses_cut),
                    service(2, crosses_outer),
                ],
            }
        )

    tool, session = search(monkeypatch, respond)
    result = (
        await tool.query(
            **arguments(
                services_names=["Школа"],
                intersection_type="within",
                layers={"a": fc(area), "b": fc()},
            )
        )
    )["services"]
    assert len(session.calls) == 5
    assert unary_union([shape(c["body"]) for c in session.calls[1:]]).equals(area)
    assert len(result["features"]) == 1
    assert result["features"][0]["properties"]["service_id"] == 1
    assert result["meta"]["part_count"] == 4
    assert result["meta"]["intersection_type"] == "within"


async def test_split_depth_limit_has_russian_error(monkeypatch):
    monkeypatch.setenv("COMPLIANCE_BY_GEOMETRY_MAX_SPLIT_DEPTH", "1")
    tool, session = search(monkeypatch, lambda *_: Response(413, text="too large"))
    with pytest.raises(ToolError, match="после 1 уровней деления"):
        await tool.query(**arguments(services_names=["Школа"], zones_layers=["a"]))
    assert len(session.calls) == 2


async def test_unrelated_422_is_not_split_or_retried(monkeypatch):
    tool, session = search(
        monkeypatch, lambda *_: Response(422, text="invalid geometry")
    )
    with pytest.raises(ToolError, match="422"):
        await tool.query(**arguments(services_names=["Школа"]))
    assert len(session.calls) == 1


async def test_metric_buffer_includes_sources_outside_project(monkeypatch):
    outside = Point(30.0105, 60.005)
    tool, session = search(
        monkeypatch,
        lambda *_: Response(
            data={"type": "FeatureCollection", "features": [service(1, outside)]}
        ),
    )
    result = (
        await tool.query(
            **arguments(services_names=["Школа"], zones_layers=["a"], buffer_m=100)
        )
    )["services"]
    assert len(result["features"]) == 1
    assert shape(session.calls[0]["body"]).contains(outside)


async def test_empty_area_and_unknown_type(monkeypatch):
    tool, session = search(monkeypatch, lambda *_: Response(data=fc()))
    result = (await tool.query(**arguments(layers={"a": fc(), "b": fc()})))["services"]
    assert result["features"] == []
    assert result["meta"]["calls_made"] == 0
    with pytest.raises(ToolError, match="не найдены"):
        await tool.query(**arguments(services_names=["Неизвестный тип"]))
    assert session.calls == []


async def test_service_auth_matches_get_services_identity_forwarding(monkeypatch):
    class ServiceAuth:
        async def get_authorization_headers(self):
            return {"Authorization": "Bearer service-token"}

    tool, session = search(
        monkeypatch, lambda *_: Response(data=fc()), service_auth=ServiceAuth()
    )
    await tool.query(**arguments(services_names=["Школа"], token="u1"))
    assert session.calls[0]["headers"]["Authorization"] == "Bearer service-token"
    assert session.calls[0]["headers"]["X-User-Id"] == "u1"


@pytest.mark.parametrize("buffer_m", [-1, float("nan"), float("inf")])
async def test_invalid_buffer_is_rejected_before_http(monkeypatch, buffer_m):
    tool, session = search(monkeypatch, lambda *_: Response(data=fc()))
    with pytest.raises(ValueError, match="buffer_m"):
        await tool.query(**arguments(buffer_m=buffer_m))
    assert session.calls == []


def test_pure_exact_predicates_keep_boundary_and_crossing_geometry():
    zones = fc(box(0, 0, 2, 2))
    objects = fc(box(1, 1, 3, 3), Point(0, 1), Point(1, 1))
    snapshot = deepcopy(objects)
    assert len(filter_objects_by_zones(zones, objects, "intersects")["features"]) == 3
    assert len(filter_objects_by_zones(zones, objects, "within")["features"]) == 1
    assert objects == snapshot
