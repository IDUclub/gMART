"""Silent resolution of the project base scenario for indicator answers.

The base is never asked about: it is read from the project, or the answer degrades to a
summary with an honest note. Values and identifiers still come from authenticated reads.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.dependencies.dependencies import (
    get_scenario_data_service,
    get_urban_mcp_client,
)
from src.agents.mcp_clients.urban_mcp_client import UrbanMcpTool
from src.agents.routers.scenario_data_controller import scenario_data_router
from src.agents.services.scenario_data.scenario_data_service import ScenarioDataService


def _tool(group: str, name: str, title: str, *, keys: tuple[str, ...]) -> UrbanMcpTool:
    return UrbanMcpTool(
        group=group,
        name=name,
        title=title,
        description=title,
        input_schema={
            "type": "object",
            "properties": {key: {"type": "integer"} for key in keys},
            "required": list(keys),
        },
        tags=(),
    )


SCENARIO_BY_ID = _tool(
    "projects", "GetScenarioById", "Получить сценарий", keys=("scenario_id",)
)
PROJECT_BY_ID = _tool(
    "projects", "GetProjectById", "Получить проект", keys=("project_id",)
)
INDICATOR_VALUES = _tool(
    "indicators",
    "GetScenarioIndicatorsValues",
    "Получить показатели сценария",
    keys=("scenario_id",),
)


class FakeUrbanMcp:
    def __init__(self, results: dict, *, failing: tuple[str, ...] = ()) -> None:
        self.results = results
        self.failing = set(failing)
        self.calls: list[tuple[str, dict]] = []

    async def load_tools(self):
        return [SCENARIO_BY_ID, PROJECT_BY_ID, INDICATOR_VALUES]

    async def execute_tool(self, group, name, arguments, *, meta=None):
        self.calls.append((name, arguments))
        if name in self.failing:
            raise PermissionError("scenario is not accessible")
        target = arguments.get("scenario_id", arguments.get("project_id"))
        return self.results[(name, target)]

    def update_token(self, token):
        raise AssertionError("token refresh is not expected")

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def targets(self, name: str) -> list[int]:
        return [
            arguments.get("scenario_id", arguments.get("project_id"))
            for called, arguments in self.calls
            if called == name
        ]


NAMES = {772: "Застройка у реки", 700: "Исходный сценарий"}


def _scenario(sid: int, *, project_id: int | None = 10, is_based: bool = False) -> dict:
    info = {"scenario_id": sid, "name": NAMES[sid], "is_based": is_based}
    if project_id is not None:
        info["project"] = {"project_id": project_id}
    return info


def _values(sid: int, value: float) -> list[dict]:
    return [
        {
            "scenario": {"id": sid, "name": f"Сценарий {sid}"},
            "indicator": {
                "indicator_id": 1,
                "name_full": "Численность населения",
                "measurement_unit": {"name": "человек"},
            },
            "value": value,
        }
    ]


def _mcp(
    *, base_id: int | None = 700, is_based: bool = False, **kwargs
) -> FakeUrbanMcp:
    project: dict = {"project_id": 10, "name": "Проект"}
    if base_id is not None:
        project["base_scenario"] = {"id": base_id, "name": "Базовый сценарий"}
    results = {
        ("GetScenarioById", 772): _scenario(772, is_based=is_based),
        ("GetProjectById", 10): project,
        ("GetScenarioIndicatorsValues", 772): _values(772, 1500),
    }
    if base_id is not None:
        results[("GetScenarioById", base_id)] = _scenario(base_id, is_based=True)
        results[("GetScenarioIndicatorsValues", base_id)] = _values(base_id, 1000)
    return FakeUrbanMcp(results, **kwargs)


async def _run(
    monkeypatch,
    fake_llm,
    fake_urban,
    state_store,
    mcp,
    query,
    *,
    indicators_route: bool = True,
):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *args, **kwargs: fake_llm,
    )
    service = ScenarioDataService("http://llm", AsyncMock(), fake_urban, state_store)
    pipeline = (
        service.run_indicator_comparison_pipeline
        if indicators_route
        else service.run_scenario_data_pipeline
    )
    return [
        event
        async for event in pipeline(
            urban_mcp_client=mcp,
            token="token",
            model="model",
            temperature=0,
            user_query=query,
            scenario_id=772,
            persist_history=False,
        )
    ]


def _text(events) -> str:
    return "".join(
        event["content"]["text"] for event in events if event.get("type") == "chunk"
    )


COMPARISON_LABELS = [
    "Показатель",
    "Ед.",
    "Базовый сценарий «Исходный сценарий»",
    "Ваш сценарий «Застройка у реки»",
    "Разница",
    "Изменение, %",
]


def _with_area(mcp: FakeUrbanMcp) -> FakeUrbanMcp:
    for sid, area in ((772, 8.08), (700, 5.95)):
        mcp.results[("GetScenarioIndicatorsValues", sid)].append(
            {
                "scenario": {"id": sid, "name": f"Сценарий {sid}"},
                "indicator": {
                    "indicator_id": 2,
                    "name_full": "Площадь территории",
                    "measurement_unit": {"name": "км2"},
                },
                "value": area,
            }
        )
    return mcp


def _summary_model(fake_llm, summary: str) -> list[list[dict]]:
    """Answer only the summary prompt; other calls keep the fake's defaults."""
    prompts: list[list[dict]] = []
    default_chat = fake_llm.chat

    async def chat(*args, **kwargs):
        messages = kwargs.get("messages") or []
        if messages and messages[0]["content"].startswith("Напиши краткую сводку"):
            prompts.append(messages)
            return {"message": {"content": summary}, "done_reason": "stop"}
        return await default_chat(*args, **kwargs)

    fake_llm.chat = chat
    return prompts


async def test_base_scenario_is_read_first_so_the_difference_is_mine_minus_base(
    monkeypatch, fake_llm, fake_urban, state_store
):
    mcp = _mcp()

    events = await _run(
        monkeypatch,
        fake_llm,
        fake_urban,
        state_store,
        mcp,
        "Покажи «Численность населения»",
    )

    text = _text(events)
    assert mcp.targets("GetProjectById") == [10]
    assert mcp.targets("GetScenarioIndicatorsValues") == [700, 772]
    assert (
        "Сравниваются: базовый сценарий «Исходный сценарий» → "
        "ваш сценарий «Застройка у реки»." in text
    )
    assert "«Численность населения»: 1 000 → 1 500 человек (+50 %)." in text
    assert "772" not in text and "700" not in text
    assert fake_llm.chat_calls == []
    table = next(e["content"] for e in events if e.get("type") == "table")
    assert [column["label"] for column in table["columns"]] == COMPARISON_LABELS
    assert table["rows"] == [
        {
            "indicator": "Численность населения",
            "unit": "человек",
            "scenario_700": 1000,
            "scenario_772": 1500,
            "difference": 500,
            "change_percent": 50,
        }
    ]


@pytest.mark.parametrize(
    "query", ["Сравни с базовым сценарием", "Покажи показатели", "Что изменилось?"]
)
async def test_a_request_naming_no_indicator_gets_every_indicator_and_a_summary(
    query, monkeypatch, fake_llm, fake_urban, state_store
):
    mcp = _mcp()

    events = await _run(monkeypatch, fake_llm, fake_urban, state_store, mcp, query)

    text = _text(events)
    assert mcp.targets("GetScenarioIndicatorsValues") == [700, 772]
    assert (
        "Показателей: в базовом — 1, в вашем — 1. "
        "Изменились — 1, без изменений — 0." in text
    )
    assert "• Численность населения: 1 000 → 1 500 человек (+50 %)" in text
    assert text.endswith("Все значения — в таблице.")


async def test_a_base_scenario_compares_with_nothing_and_still_answers(
    monkeypatch, fake_llm, fake_urban, state_store
):
    mcp = _mcp(is_based=True)

    events = await _run(
        monkeypatch,
        fake_llm,
        fake_urban,
        state_store,
        mcp,
        "Покажи «Численность населения»",
    )

    text = _text(events)
    assert "GetProjectById" not in mcp.names()
    assert mcp.targets("GetScenarioIndicatorsValues") == [772]
    assert text.startswith(
        "Этот сценарий — базовый сценарий проекта, сравнивать не с чем.\n\n"
    )
    assert (
        "Базовый сценарий «Застройка у реки»: «Численность населения» — 1 500 человек."
        in text
    )
    assert "Ваш сценарий" not in text and "разница" not in text
    table = next(e["content"] for e in events if e.get("type") == "table")
    assert table["columns"][-1]["label"] == "Базовый сценарий «Застройка у реки»"


async def test_a_project_without_a_base_scenario_degrades_to_a_summary(
    monkeypatch, fake_llm, fake_urban, state_store
):
    mcp = _mcp(base_id=None)

    events = await _run(
        monkeypatch,
        fake_llm,
        fake_urban,
        state_store,
        mcp,
        "Покажи «Численность населения»",
    )

    text = _text(events)
    assert mcp.targets("GetProjectById") == [10]
    assert mcp.targets("GetScenarioIndicatorsValues") == [772]
    assert text.startswith("У проекта не задан базовый сценарий")
    assert (
        "Ваш сценарий «Застройка у реки»: «Численность населения» — 1 500 человек."
        in text
    )


async def test_an_unreadable_project_degrades_to_a_summary_instead_of_failing(
    monkeypatch, fake_llm, fake_urban, state_store
):
    mcp = _mcp(failing=("GetProjectById",))

    events = await _run(
        monkeypatch,
        fake_llm,
        fake_urban,
        state_store,
        mcp,
        "Покажи «Численность населения»",
    )

    text = _text(events)
    assert mcp.targets("GetScenarioIndicatorsValues") == [772]
    assert "Базовый сценарий недоступен" in text
    assert (
        "Ваш сценарий «Застройка у реки»: «Численность населения» — 1 500 человек."
        in text
    )


async def test_opting_out_in_the_prompt_never_touches_the_project(
    monkeypatch, fake_llm, fake_urban, state_store
):
    mcp = _mcp()

    events = await _run(
        monkeypatch,
        fake_llm,
        fake_urban,
        state_store,
        mcp,
        "Покажи «Численность населения» без сравнения",
    )

    text = _text(events)
    assert "GetProjectById" not in mcp.names()
    assert mcp.targets("GetScenarioIndicatorsValues") == [772]
    assert (
        "Ваш сценарий «Застройка у реки»: «Численность населения» — 1 500 человек."
        in text
    )
    assert "разница" not in text


async def test_the_old_route_keeps_its_behaviour_until_a_comparison_is_asked_for(
    monkeypatch, fake_llm, fake_urban, state_store
):
    mcp = _mcp()

    await _run(
        monkeypatch,
        fake_llm,
        fake_urban,
        state_store,
        mcp,
        "Покажи «Численность населения»",
        indicators_route=False,
    )

    assert "GetProjectById" not in mcp.names()
    assert mcp.targets("GetScenarioIndicatorsValues") == [772]


async def test_the_old_route_substitutes_the_base_when_no_second_id_is_named(
    monkeypatch, fake_llm, fake_urban, state_store
):
    mcp = _mcp()

    events = await _run(
        monkeypatch,
        fake_llm,
        fake_urban,
        state_store,
        mcp,
        "Сравни «Численность населения» с базовым сценарием",
        indicators_route=False,
    )

    assert mcp.targets("GetProjectById") == [10]
    assert mcp.targets("GetScenarioIndicatorsValues") == [700, 772]
    text = _text(events)
    assert "«Численность населения»: 1 000 → 1 500 человек (+50 %)." in text
    assert "772" not in text and "700" not in text
    table = next(e["content"] for e in events if e.get("type") == "table")
    assert [column["label"] for column in table["columns"]] == COMPARISON_LABELS


ORCHESTRATOR_TASK = (
    "Сравни все показатели выбранного сценария с базовым сценарием проекта"
)


async def test_the_orchestrator_task_gets_a_checked_model_summary_on_the_qa_route(
    monkeypatch, fake_llm, fake_urban, state_store
):
    mcp = _with_area(_mcp())
    prompts = _summary_model(
        fake_llm,
        "Численность населения выросла на 50 %, площадь территории увеличилась "
        "на 2,13 км². Все значения — в таблице.",
    )

    events = await _run(
        monkeypatch,
        fake_llm,
        fake_urban,
        state_store,
        mcp,
        ORCHESTRATOR_TASK,
        indicators_route=False,
    )

    assert mcp.targets("GetScenarioIndicatorsValues") == [700, 772]
    assert any(
        event.get("type") == "status" and event["content"]["text"] == "Готовлю сводку…"
        for event in events
    )
    assert len(prompts) == 1
    assert "772" not in prompts[0][0]["content"]
    assert "700" not in prompts[0][0]["content"]
    text = _text(events)
    assert text.startswith(
        "Сравниваются: базовый сценарий «Исходный сценарий» → "
        "ваш сценарий «Застройка у реки».\n\n"
        "Численность населения выросла на 50 %, площадь территории увеличилась "
        "на 2,13 км².\n\n"
    )
    assert text.endswith("Все значения — в таблице.")
    table = next(e["content"] for e in events if e.get("type") == "table")
    assert [column["label"] for column in table["columns"]] == COMPARISON_LABELS
    assert [row["indicator"] for row in table["rows"]] == [
        "Площадь территории",
        "Численность населения",
    ]


async def test_a_summary_with_an_invented_number_is_replaced_by_the_computed_one(
    monkeypatch, fake_llm, fake_urban, state_store
):
    mcp = _with_area(_mcp())
    _summary_model(fake_llm, "Численность населения выросла в 3 раза.")

    events = await _run(
        monkeypatch,
        fake_llm,
        fake_urban,
        state_store,
        mcp,
        ORCHESTRATOR_TASK,
        indicators_route=False,
    )

    text = _text(events)
    assert "в 3 раза" not in text
    assert "Изменились — 2, без изменений — 0." in text
    assert "• Численность населения: 1 000 → 1 500 человек (+50 %)" in text


async def test_a_project_pointing_at_the_selected_scenario_is_not_compared_with_itself(
    monkeypatch, fake_llm, fake_urban, state_store
):
    mcp = _mcp(base_id=None)
    mcp.results[("GetProjectById", 10)]["base_scenario"] = {
        "id": 772,
        "name": "Застройка у реки",
    }

    events = await _run(
        monkeypatch,
        fake_llm,
        fake_urban,
        state_store,
        mcp,
        "Покажи «Численность населения»",
    )

    text = _text(events)
    assert mcp.targets("GetScenarioIndicatorsValues") == [772]
    assert text.startswith("Этот сценарий — базовый сценарий проекта")
    assert "Базовый сценарий «Застройка у реки»" in text


class FakeScenarioDataService:
    """Records which pipeline the route picked."""

    def __init__(self) -> None:
        self.called: list[str] = []

    async def run_scenario_data_pipeline(self, model=None, **kwargs):
        self.called.append("qa")
        yield {"type": "chunk", "content": {"text": "qa", "done": True}}

    async def run_indicator_comparison_pipeline(self, model=None, **kwargs):
        self.called.append("indicators")
        yield {"type": "chunk", "content": {"text": "indicators", "done": True}}


@pytest.fixture
def routed_service():
    return FakeScenarioDataService()


@pytest.fixture
def client(routed_service):
    app = FastAPI()
    app.include_router(scenario_data_router)
    app.dependency_overrides[verify_bearer_token] = lambda: "test-token"
    app.dependency_overrides[get_urban_mcp_client] = lambda: object()
    app.dependency_overrides[get_scenario_data_service] = lambda: routed_service
    with TestClient(app) as test_client:
        yield test_client


def _parse_sse(text: str) -> list[dict]:
    return [
        json.loads(line[len("data:") :].strip())
        for line in text.splitlines()
        if line.startswith("data:")
    ]


def test_the_indicators_route_runs_the_comparison_pipeline(client, routed_service):
    response = client.get(
        "/scenario-data/indicators/stream",
        params={"request": "Покажи показатели", "scenario_id": 772},
    )

    assert response.status_code == 200
    assert routed_service.called == ["indicators"]
    assert _parse_sse(response.text)[0]["content"]["text"] == "indicators"


def test_the_qa_route_is_left_alone(client, routed_service):
    response = client.get(
        "/scenario-data/qa/stream",
        params={"request": "Покажи показатели", "scenario_id": 772},
    )

    assert response.status_code == 200
    assert routed_service.called == ["qa"]
