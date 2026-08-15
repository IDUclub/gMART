import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from src.agents.mcp_clients.urban_mcp_client import UrbanMcpTool
from src.agents.services.pipeline_state import PipelineStateStore
from src.agents.services.scenario_data_service import ScenarioDataService


def test_scenario_id_is_enforced_over_model_arguments():
    tool = UrbanMcpTool(
        group="projects",
        name="GetScenarioById",
        title="Scenario",
        description="",
        input_schema={
            "type": "object",
            "properties": {"scenario_id": {"type": "integer"}},
            "required": ["scenario_id"],
        },
        tags=(),
    )

    assert ScenarioDataService._prepare_arguments(
        tool, {"scenario_id": 999, "injected": "ignored"}, 42
    ) == {"scenario_id": 42}


def test_extracts_only_actual_feature_collections():
    layer = {"type": "FeatureCollection", "features": []}
    result = {
        "with_geometry_but_not_geojson": [{"geometry": {"type": "Point"}}],
        "nested": {"layer": layer},
    }

    assert list(ScenarioDataService._feature_collections(result)) == [
        ("nested.layer", layer)
    ]


def test_list_result_becomes_strict_table():
    table = ScenarioDataService._table_from_result(
        [{"id": 1, "name": "Школа"}], name="scenario objects", title="Объекты"
    )

    assert table == {
        "name": "scenario_objects",
        "title": "Объекты",
        "columns": [
            {"key": "id", "label": "id"},
            {"key": "name", "label": "name"},
        ],
        "rows": [{"id": 1, "name": "Школа"}],
    }


async def test_pipeline_replay_buffer_serializes_geojson_datetimes():
    redis = AsyncMock()
    store = PipelineStateStore(redis)
    event = {
        "type": "feature_collection",
        "content": {
            "feature_collection": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": None,
                        "properties": {
                            "updated_at": datetime(
                                2026, 8, 15, 10, 0, tzinfo=timezone.utc
                            )
                        },
                    }
                ],
            }
        },
    }

    await store.buffer_event("request-1", event)

    payload = redis.rpush.await_args.args[1]
    assert (
        json.loads(payload)["content"]["feature_collection"]["features"][0][
            "properties"
        ]["updated_at"]
        == "2026-08-15 10:00:00+00:00"
    )
