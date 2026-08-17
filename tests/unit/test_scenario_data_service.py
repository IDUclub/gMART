import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

from src.agents.dto.scenario_data_request_dto import ScenarioDataRequestDTO
from src.agents.mcp_clients.urban_mcp_client import UrbanMcpTool
from src.agents.services import scenario_data_service as scenario_data_service_module
from src.agents.services.pipeline_state import PipelineStateStore
from src.agents.services.scenario_data_plan_builder import ScenarioDataPlanBuilder
from src.agents.services.scenario_data_service import ScenarioDataService
from src.agents.services.service_entities.scenario_data_action import (
    ScenarioDataAction,
    ScenarioDataActionKind,
)


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

    async def draft_answer(model, user_query, observations, temperature, history):
        assert any("выбрать сценарий" in item["summary"] for item in observations)
        return "Выберите сценарий."

    service._draft_answer = draft_answer
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


async def test_a_rejected_answer_buys_a_second_pass_with_the_hint(
    monkeypatch, fake_llm, fake_urban, state_store
):
    """The evaluator must re-run the pipeline, not just annotate the answer.

    Reported case: the agent said "types are not specified" while the exact counts sat in the
    observations. A retry only helps if it is *steered*, so the rejection reason is asserted to
    reach the observations the second draft sees.
    """

    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *args, **kwargs: fake_llm,
    )
    service = ScenarioDataService("http://llm", AsyncMock(), fake_urban, state_store)

    tool = UrbanMcpTool(
        group="projects",
        name="GetScenarioPhysicalObjects",
        title="Physical objects",
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
            return [tool]

        def update_token(self, token):
            raise AssertionError("token refresh is not expected")

    # The planner finishes immediately; this test is about the answer loop, not tool choice.
    async def choose_action(*args, **kwargs):
        return ScenarioDataAction(action=ScenarioDataActionKind.FINAL_ANSWER)

    service.plan_builder.choose_action = choose_action

    # Counts are present from the start, so an "unknown types" draft trips a rule.
    aggregate_observation = {
        "tool": "projects.GetScenarioPhysicalObjects",
        "layer_count": 0,
        "aggregate": {
            "total_records": 924,
            "breakdown": {
                "physical_object_type.name": {
                    "distinct_values": 2,
                    "counts": {"Жилой дом": 900, "Банк": 24},
                }
            },
        },
    }

    drafts = ["Типы объектов неизвестны.", "Всего 924 объекта: домов 900, банков 24."]
    seen_observations: list[list[dict]] = []

    async def draft_answer(model, user_query, observations, temperature, history):
        if not seen_observations:
            observations.append(aggregate_observation)
        seen_observations.append([dict(item) for item in observations])
        return drafts[min(len(seen_observations) - 1, len(drafts) - 1)]

    service._draft_answer = draft_answer

    events = [
        event
        async for event in service.run_scenario_data_pipeline(
            urban_mcp_client=FakeUrbanMcp(),
            token="token",
            model="model",
            temperature=0,
            user_query="Какие объекты есть в сценарии?",
            scenario_id=7,
            persist_history=False,
        )
    ]

    # Two drafts means the pipeline genuinely ran a second pass.
    assert len(seen_observations) == 2
    assert any(
        event.get("type") == "status"
        and event["content"].get("status") == "answer_retry"
        for event in events
    )
    # The second pass was told why the first was rejected.
    assert any(
        "Что исправить" in (item.get("summary") or "") for item in seen_observations[1]
    )
    # Only the accepted answer reaches the user.
    text = "".join(
        event["content"]["text"] for event in events if event.get("type") == "chunk"
    )
    assert "924" in text and "неизвестны" not in text


def test_every_status_the_service_emits_is_in_the_sse_contract():
    """A status missing from the Literal kills the stream, it does not degrade gracefully.

    The response model validates each SSE payload, so an unlisted status raises mid-stream and
    the client waits forever for a terminal event. Adding a status to the service without
    adding it here is therefore a hang, which is how `answer_review` first shipped.
    """

    import re as _re
    from typing import get_args

    from src.agents.schema.scenario_data_response import ScenarioDataStatus

    source = Path(scenario_data_service_module.__file__).read_text(encoding="utf-8")
    emitted = set(_re.findall(r'self\._status\(\s*"([a-z_]+)"', source))
    declared = set(get_args(ScenarioDataStatus.model_fields["status"].annotation))

    assert emitted, "no statuses found — did _status change shape?"
    assert emitted <= declared, f"not in the SSE contract: {sorted(emitted - declared)}"
