"""Compact physical-object queries use the services lossless HTTP core."""

import asyncio

import pytest
from fastmcp.exceptions import ToolError
from shapely.geometry import LineString, Point, mapping

from src.idu_mcp.tools_services.services_by_geometry import PhysicalObjectsByGeometry
from src.idu_mcp.tools_services.urb_api_tools import UrbanApiTool
from tests.unit.test_services_by_geometry import Response, arguments, fc, search


def physical(entity_id, geometry_id, geometry, type_id=1):
    return {
        "type": "Feature",
        "geometry": mapping(geometry),
        "properties": {
            "physical_object_id": entity_id,
            "object_geometry_id": geometry_id,
            "physical_object_type_id": type_id,
            "physical_object_type_name": "Дом",
            "name": None,
            "is_scenario_physical_object": True,
            "is_scenario_geometry": False,
            "is_locked": False,
        },
    }


def physical_search(monkeypatch, responder, **kwargs):
    services, session = search(monkeypatch, responder, **kwargs)
    # The same HTTP catalogue shape with a physical type id.
    original_get = session.get

    def get(**args):
        request = original_get(**args)
        if "physical_object_types" in args["url"]:
            for row in request.outcome.data:
                row["physical_object_type_id"] = row.pop("service_type_id")
        return request

    monkeypatch.setattr(session, "get", get)
    args = arguments()
    args["physical_objects_names"] = args.pop("services_names")
    return PhysicalObjectsByGeometry(services.client), session, args


async def test_one_post_per_type_compact_properties_and_every_geometry(monkeypatch):
    def respond(call, number):
        type_id = call["params"]["physical_object_type_id"]
        features = [
            physical(type_id, 10, Point(30.005, 60.005), type_id),
            physical(
                type_id, 11, LineString([(30.001, 60.001), (30.009, 60.009)]), type_id
            ),
            physical(type_id, 10, Point(30.005, 60.005), type_id),
        ]
        return Response(data={"type": "FeatureCollection", "features": features})

    tool, session, args = physical_search(monkeypatch, respond)
    from src.idu_mcp.tools_interfaces.urb_api_interface import (
        get_physical_objects_by_geometry,
    )

    token = args.pop("token")
    urban = UrbanApiTool(tool.client)
    assert (
        urban.services_by_geometry.semaphore
        is urban.physical_objects_by_geometry.semaphore
    )
    result = (
        await get_physical_objects_by_geometry(
            **args, user_id=token, urban_api_tools=urban
        )
    )["services"]
    assert len(session.calls) == 2
    for call in session.calls:
        assert call["url"].endswith("/physical_objects/by_geometry/geojson")
        assert call["params"] == {
            "physical_object_type_id": call["params"]["physical_object_type_id"],
            "intersection_type": "intersects",
        }
        assert call["headers"]["Authorization"] == "Bearer user-token"
    assert len(result["features"]) == 4
    props = result["features"][0]["properties"]
    assert props["physical_object_type"] == {
        "physical_object_type_id": 1,
        "name": "Дом",
    }
    assert props["object_ref"]["id"] == "physical_object/1/geometry/10"
    assert "building" not in props and "properties" not in props
    assert result["meta"]["per_type_counts"] == {"1": 2, "2": 2}
    assert result["meta"]["calls_made"] == 2
    assert ":physical_objects_by_geometry:" in result["meta"]["revision"]


async def test_split_merges_pairs_and_applies_original_within(monkeypatch):
    inside = physical(1, 10, Point(30.005, 60.005))
    second_geometry = physical(1, 11, LineString([(30.001, 60.001), (30.009, 60.009)]))
    crossing = physical(2, 20, LineString([(29.99, 60.005), (30.005, 60.005)]))

    def respond(call, number):
        if number == 1:
            return Response(413, text="payload too large")
        assert call["params"]["intersection_type"] == "intersects"
        return Response(
            data={
                "type": "FeatureCollection",
                "features": [inside, second_geometry, crossing],
            }
        )

    tool, session, args = physical_search(monkeypatch, respond)
    args.update(
        physical_objects_names=["Школа"], zones_layers=["a"], intersection_type="within"
    )
    result = (await tool.query(**args))["services"]
    assert session.calls[0]["params"]["intersection_type"] == "within"
    assert result["meta"]["mode"] == "split"
    assert result["meta"]["calls_made"] == 5
    assert [f["properties"]["object_geometry_id"] for f in result["features"]] == [
        10,
        11,
    ]


async def test_empty_area_and_unknown_type(monkeypatch):
    tool, session, args = physical_search(monkeypatch, lambda *_: Response(data=fc()))
    args["layers"] = {"a": fc(), "b": fc()}
    result = (await tool.query(**args))["services"]
    assert result["features"] == [] and not session.calls
    args["physical_objects_names"] = ["Неизвестный тип"]
    with pytest.raises(ToolError, match="не найдены"):
        await tool.query(**args)
    assert not session.calls


async def test_physical_and_service_queries_share_concurrency_limit(monkeypatch):
    monkeypatch.setenv("COMPLIANCE_BY_GEOMETRY_CONCURRENCY", "1")
    tool, session, args = physical_search(monkeypatch, lambda *_: Response(data=fc()))
    urban = UrbanApiTool(tool.client)
    await asyncio.gather(
        urban.get_physical_objects_by_geometry(**args),
        urban.get_services_by_geometry(**arguments()),
    )
    assert session.maximum == 1
