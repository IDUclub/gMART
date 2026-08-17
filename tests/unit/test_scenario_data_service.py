import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

from src.agents.dto.scenario_data_request_dto import ScenarioDataRequestDTO
from src.agents.mcp_clients.urban_mcp_client import UrbanMcpTool
from src.agents.services import scenario_data_service as scenario_data_service_module
from src.agents.services.pipeline_state import PipelineStateStore
from src.agents.services.scenario_data_plan_builder import (
    ScenarioDataPlanBuilder,
    _off_topic_penalty,
)
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


def test_planner_prompt_uses_concrete_action_values():
    prompt = ScenarioDataPlanBuilder._build_prompt([], [], 772)

    assert '"action": "call_tool | final_answer"' not in prompt
    assert '"call_tool" или "final_answer"' in prompt
    assert '"action": "final_answer"' in prompt


async def test_pipeline_completion_awaits_chat_storage_persistence(
    monkeypatch, fake_llm, fake_urban, state_store
):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *args, **kwargs: fake_llm,
    )
    service = ScenarioDataService("http://llm", AsyncMock(), fake_urban, state_store)
    service.add_complex_message = AsyncMock()
    parts = [AsyncMock()]

    await service._complete_pipeline(
        "request",
        "token",
        "chat",
        parts,
        scenario_id=772,
        persist_history=True,
    )

    service.add_complex_message.assert_awaited_once()
    assert service.add_complex_message.await_args.args[1] == "chat"
    assert service.add_complex_message.await_args.args[3] == parts


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
        and "выберите сценарий" in event["content"]["text"].lower()
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


class TestPlannerToolSubject:
    """Restriction tools stay available, but stop outranking on-subject ones.

    "Получить зоны ограничений объектов на территории" shares "объекты" and "территория" with
    a plain objects question and scored level with the real objects tool, so the model was free
    to answer about restriction zones instead.
    """

    @staticmethod
    def _tool(group: str, name: str, title: str) -> UrbanMcpTool:
        return UrbanMcpTool(
            group=group,
            name=name,
            title=title,
            description=title,
            input_schema={"type": "object", "properties": {}},
            tags=(),
        )

    def _catalogue(self) -> list[UrbanMcpTool]:
        return [
            self._tool(
                "territories",
                "GetTerritoryPhysicalObjects",
                "Получить физические объекты на территории",
            ),
            self._tool(
                "projects",
                "GetContextBuffers",
                "Получить зоны ограничений объектов на территории контекста",
            ),
        ]

    def test_a_restriction_tool_is_penalised_for_an_objects_question(self):
        buffers, objects = self._catalogue()[1], self._catalogue()[0]
        question = "какие объекты есть на территории и сколько их?"

        assert _off_topic_penalty(buffers, question) > 0
        assert _off_topic_penalty(objects, question) == 0

    def test_no_penalty_once_restrictions_are_what_was_asked(self):
        buffers = self._catalogue()[1]

        assert _off_topic_penalty(buffers, "какие зоны ограничений есть?") == 0

    def test_a_restriction_question_still_reaches_them(self):
        """They are Urban API data and must stay answerable when actually asked for."""
        shortlist = ScenarioDataPlanBuilder._shortlist(
            self._catalogue(), "Какие зоны ограничений есть на территории?", []
        )

        assert "GetContextBuffers" in [tool.name for tool in shortlist]


class TestPlannerEmptyResponse:
    async def test_an_empty_reply_escalates_instead_of_failing(self):
        """An identical retry cannot help when the server returned nothing at all."""

        calls: list[dict] = []

        class FlakyLlm:
            async def chat(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    return {"message": {"content": ""}}
                return {
                    "message": {
                        "content": '{"action": "final_answer", "reason": "готово"}'
                    }
                }

        action = await ScenarioDataPlanBuilder(FlakyLlm()).choose_action(
            "model", "Какие объекты?", [], []
        )

        assert action.action == ScenarioDataActionKind.FINAL_ANSWER
        assert len(calls) == 2
        # The retry must actually differ: a bigger answer budget, and — the lever that
        # actually decides this on a Harmony-served gpt-oss — a higher reasoning effort.
        assert calls[1]["options"]["num_predict"] > calls[0]["options"]["num_predict"]
        assert "reasoning_effort" not in calls[0]
        assert calls[1]["reasoning_effort"] == "medium"

    async def test_the_last_attempt_drops_the_schema_constraint(self):
        """Structured output measurably shortens replies on this server; it goes last."""

        calls: list[dict] = []

        class SilentLlm:
            async def chat(self, **kwargs):
                calls.append(kwargs)
                if len(calls) <= 2:
                    return {"message": {"content": ""}}
                return {"message": {"content": '{"action": "final_answer"}'}}

        await ScenarioDataPlanBuilder(SilentLlm()).choose_action(
            "model", "Какие объекты?", [], []
        )

        assert "format" in calls[0] and "format" in calls[1]
        assert "format" not in calls[-1]


class TestPlannerAmbiguousActionRepair:
    @staticmethod
    def _tool() -> UrbanMcpTool:
        return UrbanMcpTool(
            group="physical_objects",
            name="GetPhysicalObjects",
            title="Физические объекты сценария",
            description="",
            input_schema={"type": "object", "properties": {}},
            tags=(),
        )

    async def test_exact_old_placeholder_is_repaired_to_a_real_tool_call(self):
        tool = self._tool()

        class PlaceholderLlm:
            async def chat(self, **kwargs):
                return {
                    "message": {
                        "content": json.dumps(
                            {
                                "action": "call_tool | final_answer",
                                "group": tool.group,
                                "tool_name": tool.name,
                                "arguments": {},
                            }
                        )
                    }
                }

        action = await ScenarioDataPlanBuilder(PlaceholderLlm()).choose_action(
            "model", "Какие объекты есть?", [tool], []
        )

        assert action.action == ScenarioDataActionKind.CALL_TOOL
        assert action.tool_name == tool.name

    async def test_exact_old_placeholder_is_repaired_to_final_answer(self):
        class PlaceholderLlm:
            async def chat(self, **kwargs):
                return {
                    "message": {
                        "content": json.dumps(
                            {
                                "action": "call_tool | final_answer",
                                "group": None,
                                "tool_name": None,
                                "reason": "готово",
                            }
                        )
                    }
                }

        action = await ScenarioDataPlanBuilder(PlaceholderLlm()).choose_action(
            "model", "Какие объекты есть?", [], []
        )

        assert action.action == ScenarioDataActionKind.FINAL_ANSWER

    async def test_an_ambiguous_partial_tool_is_not_guessed(self):
        calls = 0

        class InvalidLlm:
            async def chat(self, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return {
                        "message": {
                            "content": json.dumps(
                                {
                                    "action": "call_tool | final_answer",
                                    "group": "physical_objects",
                                    "tool_name": None,
                                }
                            )
                        }
                    }
                assert any(
                    "не объединяй варианты через символ |" in message["content"]
                    for message in kwargs["messages"]
                )
                return {"message": {"content": '{"action": "final_answer"}'}}

        action = await ScenarioDataPlanBuilder(InvalidLlm()).choose_action(
            "model", "Какие объекты есть?", [self._tool()], []
        )

        assert calls == 2
        assert action.action == ScenarioDataActionKind.FINAL_ANSWER
