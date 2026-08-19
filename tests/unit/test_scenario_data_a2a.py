from __future__ import annotations

import pytest
from python_a2a.models.task import TaskState

from src.agents.a2a.a2a_format import SCENARIO_CONTEXT_EXTENSION_URI
from src.agents.a2a.scenario_data_agent import ScenarioDataA2AAgent
from src.agents.a2a.scenario_data_executor import ScenarioDataAgentExecutor
from src.agents.a2a.task_store import A2ATaskStore
from src.agents.common.exceptions.a2a_exceptions import A2AInvalidParamsError
from src.agents.services.scenario_data_a2a_service import ScenarioDataA2AService

FEATURE_COLLECTION = {
    "type": "FeatureCollection",
    "features": [],
}
TABLE = {
    "name": "service_types",
    "title": "Типы сервисов",
    "columns": [{"key": "id", "label": "id"}],
    "rows": [{"id": 1}],
}


class FakeScenarioDataService:
    def __init__(self, events: list[dict] | None = None) -> None:
        self.events = events or []
        self.calls: list[dict] = []

    async def run_scenario_data_pipeline(self, **kwargs):
        self.calls.append(kwargs)
        for event in self.events:
            yield event


def message(text: str, **extra) -> dict:
    value = {"role": "user", "parts": [{"type": "text", "text": text}]}
    value.update(extra)
    return value


def events() -> list[dict]:
    return [
        {"type": "chunk", "content": {"text": "Ответ", "done": False}},
        {
            "type": "feature_collection",
            "content": {
                "name": "schools",
                "feature_collection": FEATURE_COLLECTION,
            },
        },
        {"type": "table", "content": TABLE},
        {"type": "chunk", "content": {"text": " готов", "done": True}},
    ]


def test_agent_card_uses_scenario_data_contract_with_optional_context():
    card = ScenarioDataA2AAgent().get_agent_card("http://host:80")

    assert card["name"] == "scenario-data-agent"
    assert card["url"] == "http://host:80/scenario-data/a2a"
    extension = next(
        item
        for item in card["capabilities"]["extensions"]
        if item["uri"] == SCENARIO_CONTEXT_EXTENSION_URI
    )
    assert extension["required"] is False
    assert extension["params"]["required"] == []
    assert "application/json" in card["defaultOutputModes"]


def test_execution_accepts_missing_scenario_id_and_chat_id():
    executor = ScenarioDataAgentExecutor(FakeScenarioDataService(), A2ATaskStore())

    execution = executor._prepare_execution(
        {
            "message": message(
                "Какие типы сервисов доступны?", metadata={"chat_id": "chat-1"}
            )
        }
    )

    assert execution["scenario_id"] is None
    assert execution["chat_id"] == "chat-1"


def test_execution_accepts_structured_and_inline_scenario_id():
    executor = ScenarioDataAgentExecutor(FakeScenarioDataService(), A2ATaskStore())

    structured = executor._prepare_execution(
        {"message": message("Объекты", metadata={"scenario_id": 42})}
    )
    inline = executor._prepare_execution(
        {"message": message("scenario_id=43 Покажи объекты")}
    )

    assert structured["scenario_id"] == 42
    assert inline["scenario_id"] == 43
    assert "scenario_id" not in inline["user_query"]


def test_invalid_scenario_id_is_rejected():
    executor = ScenarioDataAgentExecutor(FakeScenarioDataService(), A2ATaskStore())

    with pytest.raises(A2AInvalidParamsError):
        executor._prepare_execution(
            {"message": message("Объекты", metadata={"scenario_id": "bad"})}
        )


async def test_executor_disables_history_and_preserves_optional_context():
    pipeline = FakeScenarioDataService(events())
    executor = ScenarioDataAgentExecutor(pipeline, A2ATaskStore())

    await executor.execute(
        {"id": "task-1", "message": message("Справочники")}, object(), "token"
    )

    (call,) = pipeline.calls
    assert call["persist_history"] is False
    assert call["scenario_id"] is None
    assert call["request_id"] == "task-1"
    assert call["token"] == "token"


async def test_json_rpc_returns_text_geojson_and_table_artifacts():
    service = ScenarioDataA2AService(FakeScenarioDataService(events()))

    response = await service.handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": "rpc-1",
            "method": "message/send",
            "params": {"id": "task-1", "message": message("Справочники")},
        },
        object(),
        "token",
    )

    task = response["result"]
    assert task["status"]["state"] == TaskState.COMPLETED.value
    artifact_ids = {artifact["artifactId"] for artifact in task["artifacts"]}
    assert artifact_ids == {
        "scenario-data-agent-text",
        "geojson-schools",
        "table-service_types",
    }
    for artifact in task["artifacts"]:
        for part in artifact["parts"]:
            assert "kind" in part and "type" not in part


async def test_streaming_json_rpc_has_task_and_completed_terminal_event():
    service = ScenarioDataA2AService(FakeScenarioDataService(events()))

    frames = [
        frame
        async for frame in service.stream_json_rpc(
            {
                "jsonrpc": "2.0",
                "id": "rpc-1",
                "method": "message/stream",
                "params": {"message": message("Справочники")},
            },
            object(),
            "token",
        )
    ]

    assert frames[0]["result"]["kind"] == "task"
    assert any(
        frame.get("result", {}).get("kind") == "status-update"
        and frame["result"].get("final")
        and frame["result"]["status"]["state"] == TaskState.COMPLETED.value
        for frame in frames
    )


async def test_get_list_and_cancel_task_methods():
    service = ScenarioDataA2AService(FakeScenarioDataService())
    service.task_store.create_task("task-1", "context-1", message("Справочники"), {})

    cancel = await service.handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": "cancel",
            "method": "tasks/cancel",
            "params": {"id": "task-1"},
        },
        object(),
        "token",
    )
    get_task = await service.handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": "get",
            "method": "tasks/get",
            "params": {"id": "task-1"},
        },
        object(),
        "token",
    )
    listed = await service.handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": "list",
            "method": "tasks/list",
            "params": {},
        },
        object(),
        "token",
    )

    assert cancel["result"]["status"]["state"] == TaskState.CANCELED.value
    assert get_task["result"]["id"] == "task-1"
    assert [task["id"] for task in listed["result"]] == ["task-1"]
