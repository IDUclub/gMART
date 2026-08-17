import geopandas as gpd
import pandas as pd
import pytest
from fakeredis.aioredis import FakeRedis
from fastmcp.exceptions import ToolError
from shapely.geometry import Point

from src.idu_mcp.tools_services.workspace_store import (
    WorkspaceStore,
    frame_from_payload,
)


async def test_workspace_is_chat_scoped_and_does_not_store_payload_in_redis(tmp_path):
    redis = FakeRedis(decode_responses=True)
    store = WorkspaceStore(redis, root=str(tmp_path), ttl_seconds=3600)

    metadata = await store.create(
        pd.DataFrame([{"id": 1, "name": "Школа"}]),
        owner_id="user-a",
        chat_id="chat-a",
    )

    raw = await redis.get(store.KEY_PREFIX + metadata["handle"])
    assert '"records"' not in raw
    assert raw.count("Школа") == 1  # bounded profile metadata, not the table payload
    frame, _ = await store.load(metadata["handle"], owner_id="user-a", chat_id="chat-a")
    assert frame.to_dict(orient="records") == [{"id": 1, "name": "Школа"}]
    with pytest.raises(ToolError, match="другому пользователю"):
        await store.load(metadata["handle"], owner_id="user-a", chat_id="chat-b")
    with pytest.raises(ToolError, match="другому пользователю"):
        await store.load(metadata["handle"], owner_id="user-b", chat_id="chat-a")


async def test_workspace_preserves_geospatial_frame_as_geoparquet(tmp_path):
    store = WorkspaceStore(FakeRedis(decode_responses=True), root=str(tmp_path))
    frame = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(30.3, 59.9)], crs="EPSG:4326")

    metadata = await store.create(frame, owner_id="user", chat_id="chat")
    restored, _ = await store.load(metadata["handle"], owner_id="user", chat_id="chat")

    assert metadata["format"] == "geoparquet"
    assert isinstance(restored, gpd.GeoDataFrame)
    assert restored.crs.to_epsg() == 4326


async def test_workspace_flattens_nested_fields_and_returns_bounded_profile(tmp_path):
    store = WorkspaceStore(FakeRedis(decode_responses=True), root=str(tmp_path))
    frame = frame_from_payload(
        [
            {"id": 1, "type": {"id": 5, "name": "Школа"}},
            {"id": 2, "type": {"id": 5, "name": "Школа"}},
        ],
        None,
    )

    metadata = await store.create(
        frame, owner_id="user", chat_id="chat", lineage={"operation": "create"}
    )

    assert metadata["columns"] == ["id", "type.id", "type.name"]
    assert metadata["profile"]["bounded_unique_values"]["type.name"] == ["Школа"]


async def test_workspace_mcp_exposes_only_fixed_dsl_and_hides_dependencies():
    from src.idu_mcp.tools_interfaces.workspace_interface import workspace_mcp

    tools = await workspace_mcp.list_tools()
    by_name = {tool.name: tool for tool in tools}

    assert {
        "WorkspaceCreate",
        "WorkspaceFilter",
        "WorkspaceAggregate",
        "WorkspaceSpatialFilter",
        "WorkspaceToFeatureCollection",
    } <= set(by_name)
    assert "owner_id" not in by_name["WorkspaceCreate"].parameters["properties"]
    assert "store" not in by_name["WorkspaceCreate"].parameters["properties"]
    serialized = str([tool.parameters for tool in tools]).casefold()
    assert "query" not in serialized
    assert "eval" not in serialized
    assert "python" not in serialized
    assert "sql" not in serialized
    assert "url" not in serialized
    assert "path" not in serialized
