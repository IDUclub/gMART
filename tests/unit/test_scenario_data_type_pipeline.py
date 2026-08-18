from unittest.mock import AsyncMock

from src.agents.mcp_clients.urban_mcp_client import UrbanMcpTool
from src.agents.services.scenario_data_service import ScenarioDataService


def _tool(group: str, name: str, title: str, *, scenario: bool) -> UrbanMcpTool:
    properties = {"scenario_id": {"type": "integer"}} if scenario else {}
    return UrbanMcpTool(
        group=group,
        name=name,
        title=title,
        description=title,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": ["scenario_id"] if scenario else [],
        },
        tags=(),
    )


PHYSICAL_OBJECTS = _tool(
    "projects",
    "GetScenarioPhysicalObjects",
    "Получить физические объекты сценария",
    scenario=True,
)
PHYSICAL_TYPES = _tool(
    "projects",
    "GetScenarioPhysicalObjectTypes",
    "Получить типы физических объектов сценария",
    scenario=True,
)
GLOBAL_PHYSICAL_TYPES = _tool(
    "dictionaries",
    "GetPhysicalObjectTypes",
    "Получить типы физических объектов",
    scenario=False,
)


class FakeUrbanMcp:
    def __init__(self, results):
        self.results = results
        self.calls = []
        self.load_calls = 0

    async def load_tools(self):
        self.load_calls += 1
        return [PHYSICAL_OBJECTS, PHYSICAL_TYPES, GLOBAL_PHYSICAL_TYPES]

    async def execute_tool(self, group, name, arguments, *, meta):
        self.calls.append((group, name, arguments, meta))
        return self.results[name]

    def update_token(self, token):
        raise AssertionError("token refresh is not expected")


def _type(type_id: int, name: str) -> dict:
    return {
        "physical_object_type_id": type_id,
        "name": name,
        "physical_object_function": {"id": 1, "name": "Здание"},
    }


async def test_ambiguous_objects_question_finishes_with_clarification(
    monkeypatch, fake_llm, fake_urban, state_store
):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *args, **kwargs: fake_llm,
    )
    service = ScenarioDataService("http://llm", AsyncMock(), fake_urban, state_store)
    mcp = FakeUrbanMcp({})

    events = [
        event
        async for event in service.run_scenario_data_pipeline(
            urban_mcp_client=mcp,
            token="token",
            model="model",
            temperature=0,
            user_query="Какие объекты есть в сценарии и сколько их по типам?",
            scenario_id=772,
            persist_history=False,
        )
    ]

    text = "".join(
        event["content"]["text"] for event in events if event.get("type") == "chunk"
    )
    assert "физические объекты" in text and "сервисы" in text
    assert mcp.load_calls == 0
    assert events[-1] == {"type": "chunk", "content": {"text": "", "done": True}}


async def test_available_service_types_clarifies_catalog_scope_before_planning(
    monkeypatch, fake_llm, fake_urban, state_store
):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *args, **kwargs: fake_llm,
    )
    service = ScenarioDataService("http://llm", AsyncMock(), fake_urban, state_store)
    service.plan_builder.build_acquisition_plan = AsyncMock(
        side_effect=AssertionError("ambiguous scope must not reach the LLM planner")
    )
    mcp = FakeUrbanMcp({})

    events = [
        event
        async for event in service.run_scenario_data_pipeline(
            urban_mcp_client=mcp,
            token="token",
            model="model",
            temperature=0,
            user_query="Какие типы городских сервисов доступны?",
            scenario_id=772,
            persist_history=False,
        )
    ]

    text = "".join(
        event["content"]["text"] for event in events if event.get("type") == "chunk"
    )
    assert "полный список" in text
    assert "проекте/сценарии" in text
    assert mcp.load_calls == 0
    assert not any(event["type"] == "plan_created" for event in events)
    assert events[-1] == {"type": "chunk", "content": {"text": "", "done": True}}


async def test_physical_type_count_bypasses_llm_and_returns_complete_table(
    monkeypatch, fake_llm, fake_urban, state_store
):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *args, **kwargs: fake_llm,
    )
    service = ScenarioDataService("http://llm", AsyncMock(), fake_urban, state_store)
    service.plan_builder.choose_action = AsyncMock(
        side_effect=AssertionError("the LLM planner must not count entities")
    )
    type_5 = _type(5, "Нежилое здание")
    type_6 = _type(6, "Жилое здание")
    mcp = FakeUrbanMcp(
        {
            "GetScenarioPhysicalObjects": [
                {"physical_object_id": 1, "physical_object_type": type_5},
                {"physical_object_id": 1, "physical_object_type": type_5},
                {"physical_object_id": 2, "physical_object_type": type_5},
            ],
            "GetScenarioPhysicalObjectTypes": [type_5, type_6],
        }
    )

    events = [
        event
        async for event in service.run_scenario_data_pipeline(
            urban_mcp_client=mcp,
            token="token",
            model="model",
            temperature=0,
            user_query="Сколько физических объектов в сценарии по типам?",
            scenario_id=772,
            persist_history=False,
        )
    ]

    table = next(event["content"] for event in events if event["type"] == "table")
    assert table["rows"] == [
        {
            "type_id": 5,
            "type_name": "Нежилое здание",
            "count": 2,
            "status": "точное соответствие",
            "possible_types": "—",
        },
    ]
    assert [call[1] for call in mcp.calls] == [
        "GetScenarioPhysicalObjects",
        "GetScenarioPhysicalObjectTypes",
    ]
    assert any(
        event["type"] == "status" and "Считаю уникальные" in event["content"]["text"]
        for event in events
    )
    text = "".join(
        event["content"]["text"] for event in events if event.get("type") == "chunk"
    )
    assert "2 уникальных физических объекта" in text


async def test_linear_type_count_emits_plan_steps_and_validation(
    monkeypatch, fake_llm, fake_urban, state_store
):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *args, **kwargs: fake_llm,
    )
    service = ScenarioDataService(
        "http://llm",
        AsyncMock(),
        fake_urban,
        state_store,
        linear_workflow_enabled=True,
    )
    type_5 = _type(5, "Нежилое здание")
    mcp = FakeUrbanMcp(
        {
            "GetScenarioPhysicalObjects": [
                {"physical_object_id": 1, "physical_object_type": type_5}
            ],
            "GetScenarioPhysicalObjectTypes": [type_5],
        }
    )

    events = [
        event
        async for event in service.run_scenario_data_pipeline(
            urban_mcp_client=mcp,
            token="token",
            model="model",
            temperature=0,
            user_query="Сколько физических объектов в сценарии по типам?",
            scenario_id=772,
            persist_history=False,
        )
    ]

    assert any(event["type"] == "plan_created" for event in events)
    assert len([event for event in events if event["type"] == "step_started"]) == 2
    assert any(event["type"] == "validation_completed" for event in events)


async def test_unknown_project_type_is_resolved_from_global_dictionary(
    monkeypatch, fake_llm, fake_urban, state_store
):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *args, **kwargs: fake_llm,
    )
    service = ScenarioDataService("http://llm", AsyncMock(), fake_urban, state_store)
    type_5 = _type(5, "Нежилое здание")
    type_7 = _type(7, "Площадка")
    mcp = FakeUrbanMcp(
        {
            "GetScenarioPhysicalObjects": [
                {"physical_object_id": 10, "physical_object_type": type_7},
            ],
            "GetScenarioPhysicalObjectTypes": [type_5],
            "GetPhysicalObjectTypes": [type_5, type_7],
        }
    )

    events = [
        event
        async for event in service.run_scenario_data_pipeline(
            urban_mcp_client=mcp,
            token="token",
            model="model",
            temperature=0,
            user_query="Сколько физических объектов в сценарии по типам?",
            scenario_id=772,
            persist_history=False,
        )
    ]

    table = next(event["content"] for event in events if event["type"] == "table")
    assert table["rows"] == [
        {
            "type_id": 7,
            "type_name": "Площадка",
            "count": 1,
            "status": "точное соответствие",
            "possible_types": "—",
        }
    ]
    assert [call[1] for call in mcp.calls] == [
        "GetScenarioPhysicalObjects",
        "GetScenarioPhysicalObjectTypes",
        "GetPhysicalObjectTypes",
    ]
