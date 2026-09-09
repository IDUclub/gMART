import json
from unittest.mock import AsyncMock

import pytest

from agents.services.scenario_data.scenario_data_read import (
    UrbanReadPlan,
    broad_data_query,
    data_layers,
    validate_read_plan,
)
from agents.services.scenario_data.scenario_data_service import ScenarioDataService
from src.agents.mcp_clients.urban_mcp_client import UrbanMcpTool


def make_tool(name="GetMeasurementUnits", group="dictionaries", properties=None):
    return UrbanMcpTool(
        group=group,
        name=name,
        title="Единицы измерения",
        description="Полный справочник",
        tags=(),
        input_schema={"type": "object", "properties": properties or {}},
    )


def read_plan(tool, arguments=None):
    return UrbanReadPlan(
        operation="list",
        calls=[{"tool_name": tool.name, "arguments_json": json.dumps(arguments or {})}],
    )


@pytest.mark.parametrize(
    "query",
    [
        "Все физические объекты сценария 772",
        "Сервисы территории 1955",
        "Группы показателей из справочника",
        "Гексагоны сценария с показателями",
        "Все типы сервисов, присутствующие в сценарии",
    ],
)
def test_broad_sources_are_not_entity_or_indicator_selection(query):
    assert broad_data_query(query)


@pytest.mark.parametrize(
    "query",
    [
        "Сколько физических объектов в сценарии по типам?",
        "Сравни показатель «Население» сценариев 772 и 848",
        "Посчитай сервисов типа «Библиотека» в выбранном сценарии",
    ],
)
def test_verified_specialized_paths_remain_available(query):
    assert not broad_data_query(query)


@pytest.mark.parametrize(
    "query",
    [
        "Показатель «Социальное обеспечение (базовое)» сценария 17",
        "Покажи значения двух показателей с единицами измерения",
    ],
)
def test_names_and_output_units_do_not_change_scope(query):
    assert not broad_data_query(query)


def test_context_substitution_rejected_before_mcp_call():
    tool = make_tool(
        "GetContextServices", "projects", {"scenario_id": {"type": "integer"}}
    )
    with pytest.raises(ValueError, match="Окружение"):
        validate_read_plan(
            read_plan(tool, {"scenario_id": 17}),
            [tool],
            "Все сервисы сценария 17",
            17,
            2,
        )


def test_fabricated_identifier_rejected_before_mcp_call():
    tool = make_tool(
        "GetTerritoryById", "territories", {"territory_id": {"type": "integer"}}
    )
    with pytest.raises(ValueError, match="ID"):
        validate_read_plan(
            read_plan(tool, {"territory_id": 88}),
            [tool],
            "Карточка территории 17",
            17,
            2,
        )


def test_year_cannot_be_used_as_territory_identifier():
    tool = make_tool(
        "GetTerritoryById", "territories", {"territory_id": {"type": "integer"}}
    )
    with pytest.raises(ValueError, match="ID"):
        validate_read_plan(
            read_plan(tool, {"territory_id": 2023}),
            [tool],
            "Данные территории 17 за 2023 год",
            3,
            2,
        )


def test_sorting_enum_is_validated_before_source_request():
    tool = make_tool(
        "GetServiceTypes",
        properties={"ordering": {"type": "string", "enum": ["asc", "desc"]}},
    )
    with pytest.raises(ValueError):
        validate_read_plan(
            read_plan(tool, {"ordering": "id ASC"}),
            [tool],
            "Типы сервисов по возрастанию",
            17,
            2,
        )


def test_scope_family_excludes_scenario_from_territory_normatives():
    from agents.services.scenario_data.scenario_data_read import scoped_tools

    scenario = make_tool("GetScenarioFunctionalZones", "projects")
    normatives = make_tool("GetTerritoryNormatives", "territories")
    allowed = scoped_tools([scenario, normatives], "Нормативы территории 17")
    assert allowed == [normatives]
    with pytest.raises(ValueError, match="каталога"):
        validate_read_plan(
            read_plan(scenario), allowed, "Нормативы территории 17", 3, 2
        )


def test_flat_catalogue_cannot_be_replaced_by_hierarchy():
    from agents.services.scenario_data.scenario_data_read import scoped_tools

    flat = make_tool("GetPhysicalObjectTypes")
    tree = make_tool("GetPhysicalObjectTypesHierarchy")
    assert scoped_tools(
        [flat, tree], "Полный общий справочник типов физических объектов"
    ) == [flat]


def test_indicator_definitions_cannot_be_replaced_by_groups():
    from agents.services.scenario_data.scenario_data_read import scoped_tools

    definitions = make_tool("GetIndicatorsByParent", "indicators")
    groups = make_tool("GetIndicatorsGroups")
    assert scoped_tools(
        [definitions, groups], "Корневые типы показателей из общего справочника"
    ) == [definitions]


def test_geometry_list_keeps_ids_and_coordinates():
    geometry = {"type": "Point", "coordinates": [28.1, 60.2]}
    layers = data_layers([{"object_geometry_id": 81, "geometry": geometry}])
    assert layers[0]["features"] == [
        {
            "type": "Feature",
            "geometry": geometry,
            "properties": {"object_geometry_id": 81},
        }
    ]


@pytest.mark.parametrize(
    "result",
    [
        [{"measurement_unit_id": 17, "name": "метр"}],
        [],
        {"investment": 0, "construction": 0, "actual_start_date": None},
    ],
)
async def test_real_pipeline_publishes_source_without_freeform_llm_facts(
    monkeypatch,
    fake_llm,
    fake_urban,
    state_store,
    result,
):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *a, **kw: fake_llm,
    )
    tool = make_tool()
    fake_llm.json_responses = [read_plan(tool).model_dump_json()]
    mcp = AsyncMock()
    mcp.load_tools.return_value = [tool]
    mcp.execute_tool.return_value = result
    service = ScenarioDataService("http://llm", None, fake_urban, state_store)
    service._draft_answer = AsyncMock(
        side_effect=AssertionError("Untrusted facts must not be drafted")
    )
    events = [
        e
        async for e in service.run_scenario_data_pipeline(
            mcp,
            "token",
            "model",
            0,
            "Покажи полный справочник единиц измерения",
            scenario_id=17,
            persist_history=False,
        )
    ]
    assert mcp.execute_tool.await_count == 1
    text = "".join(e["content"].get("text", "") for e in events if e["type"] == "chunk")
    assert text
    tables = [e["content"] for e in events if e["type"] == "table"]
    if result:
        assert tables[0]["rows"] == (result if isinstance(result, list) else [result])
        assert "отсутств" not in text
    else:
        assert not tables and "записи отсутствуют" in text
    service._draft_answer.assert_not_awaited()


async def test_all_records_follow_cursor_without_changing_scope(
    monkeypatch,
    fake_llm,
    fake_urban,
    state_store,
):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *a, **kw: fake_llm,
    )
    tool = make_tool(
        "GetTerritoryPhysicalObjects",
        "territories",
        {"territory_id": {"type": "integer"}, "cursor": {"type": "string"}},
    )
    fake_llm.json_responses = [read_plan(tool, {"territory_id": 17}).model_dump_json()]
    mcp = AsyncMock()
    mcp.load_tools.return_value = [tool]
    mcp.execute_tool.side_effect = [
        {"count": 2, "results": [{"physical_object_id": 81}], "nextCursor": "next"},
        {"count": 2, "results": [{"physical_object_id": 82}], "nextCursor": None},
    ]
    service = ScenarioDataService("http://llm", None, fake_urban, state_store)
    events = [
        e
        async for e in service.run_scenario_data_pipeline(
            mcp,
            "token",
            "model",
            0,
            "Все физические объекты территории 17",
            scenario_id=3,
            persist_history=False,
        )
    ]
    assert [c.args[2] for c in mcp.execute_tool.await_args_list] == [
        {"territory_id": 17},
        {"territory_id": 17, "cursor": "next"},
    ]
    tables = [e["content"] for e in events if e["type"] == "table"]
    assert tables[0]["complete"]
    assert tables[0]["rows"] == [{"physical_object_id": 81}, {"physical_object_id": 82}]


def test_complete_large_result_emits_every_record():
    from agents.services.scenario_data.scenario_data_read import output_tables

    rows = [{"id": i, "name": str(i)} for i in range(2301)]
    tables = output_tables(ScenarioDataService, rows, "Объекты", "objects")
    assert len(tables) == 3
    assert all(t["complete"] for t in tables)
    assert [r for t in tables for r in t["rows"]] == rows


async def test_single_unambiguous_source_is_read_when_model_abstains(
    monkeypatch,
    fake_llm,
    fake_urban,
    state_store,
):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *a, **kw: fake_llm,
    )
    tool = make_tool(
        "GetSocialGroupById", "soc_groups", {"soc_group_id": {"type": "integer"}}
    )
    tool.input_schema["required"] = ["soc_group_id"]
    fake_llm.json_responses = [
        UrbanReadPlan(calls=[], operation="unsupported").model_dump_json()
    ]
    mcp = AsyncMock()
    mcp.load_tools.return_value = [tool]
    mcp.execute_tool.return_value = {"soc_group_id": 1, "name": "Группа"}
    service = ScenarioDataService("http://llm", None, fake_urban, state_store)
    events = [
        e
        async for e in service.run_scenario_data_pipeline(
            mcp,
            "token",
            "model",
            0,
            "Покажи карточку социальной группы 1",
            scenario_id=3,
            persist_history=False,
        )
    ]
    mcp.execute_tool.assert_awaited_once()
    assert mcp.execute_tool.await_args.args[2] == {"soc_group_id": 1}
    assert any(e["type"] == "table" for e in events)


def test_permanent_regional_limitation_is_explained():
    from agents.services.scenario_data.scenario_data_read import source_error_answer

    answer = source_error_answer(
        ValueError(
            "Этот метод недоступен в сценарии ПРОЕКТА. Укажите идентификатор РЕГИОНАЛЬНОГО сценария."
        )
    )
    assert "региональн" in answer
    assert "позже" not in answer


async def test_abstained_page_plan_is_reconsidered_before_giving_up(
    monkeypatch, fake_llm, fake_urban, state_store
):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *a, **kw: fake_llm,
    )
    properties = {
        "territory_id": {"type": "integer"},
        "page_size": {"type": "integer"},
        "include_child_territories": {"type": "boolean"},
    }
    tool = make_tool(
        "GetTerritoryPhysicalObjectsWithGeometry", "territories", properties
    )
    other = make_tool("GetTerritoryPhysicalObjects", "territories", properties)
    arguments = {
        "territory_id": 17,
        "page_size": 10,
        "include_child_territories": False,
    }
    fake_llm.json_responses = [
        UrbanReadPlan(calls=[], operation="unsupported").model_dump_json(),
        read_plan(tool, arguments).model_dump_json(),
    ]
    mcp = AsyncMock()
    mcp.load_tools.return_value = [tool, other]
    mcp.execute_tool.return_value = {
        "count": 100,
        "results": [
            {
                "physical_object_id": i,
                "geometry": {"type": "Point", "coordinates": [30, 60]},
            }
            for i in range(10)
        ],
        "nextCursor": "next-page",
    }
    service = ScenarioDataService("http://llm", None, fake_urban, state_store)
    events = [
        e
        async for e in service.run_scenario_data_pipeline(
            mcp,
            "token",
            "model",
            0,
            "Первую страницу физических объектов территории 17 с геометрией, размер страницы 10, без дочерних территорий. Покажи на карте.",
            scenario_id=3,
            persist_history=False,
        )
    ]
    mcp.execute_tool.assert_awaited_once()
    assert mcp.execute_tool.await_args.args[1:3] == (tool.name, arguments)
    assert len(fake_llm.chat_calls) == 2
    assert "источник" in fake_llm.chat_calls[1].messages[-1]["content"].casefold()
    assert any(e["type"] == "feature_collection" for e in events)


@pytest.mark.parametrize("map_requested", [False, True])
def test_plain_entity_list_does_not_return_one_row_per_geometry(map_requested):
    properties = {
        "scenario_id": {"type": "integer"},
        "physical_object_type_id": {"type": "integer"},
    }
    plain = make_tool("GetScenarioPhysicalObjects", "projects", properties)
    geometry = make_tool(
        "GetScenarioPhysicalObjectsWithGeometry",
        "projects",
        {**properties, "centers_only": {"type": "boolean"}},
    )
    query = "Все физические объекты сценария 17 типа физического объекта 5"
    if map_requested:
        query += " на карте"
    plan = validate_read_plan(
        read_plan(
            geometry,
            {"scenario_id": 17, "physical_object_type_id": 5, "centers_only": False},
        ),
        [plain, geometry],
        query,
        17,
        None,
    )
    assert plan.calls[0].tool_name == (geometry.name if map_requested else plain.name)
    arguments = json.loads(plan.calls[0].arguments_json)
    assert arguments["scenario_id"] == 17 and arguments["physical_object_type_id"] == 5
    assert ("centers_only" in arguments) == map_requested
