import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from src.agents.dto.scenario_data_request_dto import ScenarioDataRequestDTO
from src.agents.mcp_clients.urban_mcp_client import UrbanMcpTool
from src.agents.services.pipeline_state import PipelineStateStore
from src.agents.services.scenario_data_plan_builder import ScenarioDataPlanBuilder
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


def test_scenario_id_is_optional_in_rest_dto():
    dto = ScenarioDataRequestDTO(request="Какие типы сервисов доступны?")

    assert dto.scenario_id is None


def test_required_scenario_tool_is_hidden_without_scenario():
    scenario_tool = UrbanMcpTool(
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
    dictionary_tool = UrbanMcpTool(
        group="dictionaries",
        name="GetServiceTypes",
        title="Service types",
        description="",
        input_schema={"type": "object", "properties": {}},
        tags=(),
    )

    assert ScenarioDataService._tools_for_context(
        [scenario_tool, dictionary_tool], None
    ) == [dictionary_tool]
    assert ScenarioDataService._tools_for_context(
        [scenario_tool, dictionary_tool], 42
    ) == [scenario_tool, dictionary_tool]


def test_missing_scenario_id_is_not_injected_into_optional_tool():
    tool = UrbanMcpTool(
        group="projects",
        name="GetProjects",
        title="Projects",
        description="",
        input_schema={
            "type": "object",
            "properties": {"scenario_id": {"type": "integer"}},
        },
        tags=(),
    )

    assert ScenarioDataService._prepare_arguments(tool, {}, None) == {}


def test_planner_requests_scenario_clarification_when_context_is_missing():
    prompt = ScenarioDataPlanBuilder._build_prompt([], [], None)

    assert "Контекст сценария: не выбран" in prompt
    assert "попросить пользователя выбрать сценарий" in prompt


async def test_pipeline_without_scenario_skips_scenario_only_catalog(
    monkeypatch, fake_llm, fake_urban, state_store
):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *args, **kwargs: fake_llm,
    )
    service = ScenarioDataService("http://llm", AsyncMock(), fake_urban, state_store)
    scenario_tool = UrbanMcpTool(
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

    class FakeUrbanMcp:
        async def load_tools(self):
            return [scenario_tool]

        def update_token(self, token):
            raise AssertionError("token refresh is not expected")

    async def stream_answer(model, user_query, observations, temperature, history):
        assert any("выбрать сценарий" in item["summary"] for item in observations)
        yield {
            "type": "chunk",
            "content": {"text": "Выберите сценарий.", "done": True},
        }

    service._stream_answer = stream_answer
    events = [
        event
        async for event in service.run_scenario_data_pipeline(
            urban_mcp_client=FakeUrbanMcp(),
            token="token",
            model="model",
            temperature=0,
            user_query="Какие объекты есть в сценарии?",
            scenario_id=None,
            persist_history=False,
        )
    ]

    assert any(
        event.get("type") == "chunk"
        and event["content"]["text"] == "Выберите сценарий."
        for event in events
    )


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
