import json

import pytest

from src.agents.mcp_clients.urban_mcp_client import UrbanMcpTool
from src.agents.services.scenario_data.scenario_data_selection import (
    ScenarioEntitySelection,
    exact_type_candidate,
    may_select_entities,
    selection_candidates,
    validate_selection,
    verified_entity_records,
)
from src.agents.services.scenario_data.scenario_data_service import ScenarioDataService
from src.agents.services.scenario_data.scenario_data_types import classify_type_query


def record(identifier, *, domain="service_type", type_id=92, name="Библиотека"):
    return {
        domain.removesuffix("_type") + "_id": identifier,
        domain: {f"{domain}_id": type_id, "name": name},
        "name": f"Учреждение {identifier}",
    }


class EntityMcp:
    def __init__(self, rows=None, geometry_rows=None):
        self.rows = rows if rows is not None else [record(11), record(12), record(11)]
        self.geometry_rows = geometry_rows if geometry_rows is not None else self.rows
        self.calls = []
        self.tools = []
        for noun, domain in (
            ("PhysicalObject", "physical_object_type"),
            ("Service", "service_type"),
        ):
            for suffix in ("Types", "s", "sWithGeometry"):
                self.tools.append(
                    UrbanMcpTool(
                        group="projects",
                        name=f"GetScenario{noun}{suffix}",
                        title=noun + suffix,
                        description="",
                        tags=(),
                        input_schema={
                            "type": "object",
                            "properties": {
                                "scenario_id": {"type": "integer"},
                                f"{domain}_id": {"type": "integer"},
                                "for_context": {"type": "boolean"},
                            },
                            "required": ["scenario_id"],
                        },
                    )
                )

    async def load_tools(self):
        return self.tools

    async def execute_tool(self, group, name, arguments, *, meta):
        self.calls.append((name, arguments))
        assert arguments["scenario_id"] == 17
        assert meta == {"scenario_id": 17}
        if name == "GetScenarioPhysicalObjectTypes":
            return [{"physical_object_type_id": 48, "name": "Жилой дом"}]
        if name == "GetScenarioServiceTypes":
            return [{"service_type_id": 92, "name": "Библиотека"}]
        assert arguments == {"scenario_id": 17, "service_type_id": 92}
        if name.endswith("WithGeometry"):
            return {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [30, 60]},
                        "properties": row,
                    }
                    for row in self.geometry_rows
                ],
            }
        return self.rows


async def run_query(
    monkeypatch,
    fake_llm,
    fake_urban,
    state_store,
    operation,
    *,
    mcp=None,
    selections=None,
):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *a, **kw: fake_llm,
    )
    fake_llm.json_responses = [
        json.dumps(
            {
                "candidate": (
                    None if operation in {"unmatched", "unsupported"} else "candidate_2"
                ),
            }
        )
    ]
    if selections is not None:
        fake_llm.json_responses = [json.dumps(selection) for selection in selections]
    fake_llm.json_responses.insert(
        0,
        json.dumps(
            {
                "operation": (
                    operation if operation in {"count", "list", "map"} else "count"
                ),
                "requested_type": "библиотеки",
            }
        ),
    )
    service = ScenarioDataService("http://llm", None, fake_urban, state_store)
    mcp = mcp or EntityMcp()
    events = [
        event
        async for event in service.run_scenario_data_pipeline(
            mcp,
            "token",
            "model",
            0,
            "Покажи библиотеки выбранного сценария",
            scenario_id=17,
            persist_history=False,
        )
    ]
    return events, mcp


@pytest.mark.parametrize("operation", ["count", "list", "map"])
async def test_filtered_outputs_share_unique_entities(
    monkeypatch, fake_llm, fake_urban, state_store, operation
):
    events, mcp = await run_query(
        monkeypatch, fake_llm, fake_urban, state_store, operation
    )
    text = "".join(e["content"].get("text", "") for e in events if e["type"] == "chunk")
    assert "«Библиотека»: 2" in text
    tables = [e["content"] for e in events if e["type"] == "table"]
    assert len(tables) == 1
    if operation == "count":
        assert tables[0]["rows"] == [{"type_name": "Библиотека", "count": 2}]
    else:
        assert len(tables[0]["rows"]) == 2
    layers = [e for e in events if e["type"] == "feature_collection"]
    assert len(layers) == int(operation == "map")
    assert all("Context" not in name for name, _ in mcp.calls)


async def test_real_empty_query_reports_zero(
    monkeypatch, fake_llm, fake_urban, state_store
):
    events, mcp = await run_query(
        monkeypatch, fake_llm, fake_urban, state_store, "count", mcp=EntityMcp(rows=[])
    )
    text = "".join(e["content"].get("text", "") for e in events if e["type"] == "chunk")
    assert "«Библиотека»: 0" in text
    assert mcp.calls[-1] == (
        "GetScenarioServices",
        {"scenario_id": 17, "service_type_id": 92},
    )


async def test_unknown_type_never_fetches_unfiltered_entities(
    monkeypatch, fake_llm, fake_urban, state_store
):
    events, mcp = await run_query(
        monkeypatch, fake_llm, fake_urban, state_store, "unmatched"
    )
    assert len(mcp.calls) == 2
    assert not any(e["type"] in {"table", "feature_collection"} for e in events)
    text = "".join(e["content"].get("text", "") for e in events if e["type"] == "chunk")
    assert "не найден тип" in text


@pytest.mark.parametrize(
    "mcp",
    [
        EntityMcp(rows=[record(11, type_id=99)]),
        EntityMcp(geometry_rows=[record(99)]),
        EntityMcp(rows={"items": [record(11)], "total": 2}),
    ],
)
async def test_unverified_outputs_are_not_published(
    monkeypatch, fake_llm, fake_urban, state_store, mcp
):
    events, _ = await run_query(
        monkeypatch, fake_llm, fake_urban, state_store, "map", mcp=mcp
    )
    assert not any(e["type"] in {"table", "feature_collection"} for e in events)
    text = "".join(e["content"].get("text", "") for e in events if e["type"] == "chunk")
    assert "не прошёл проверку" in text


@pytest.mark.parametrize(
    "query",
    [
        "Сколько сервисов типа «Школа» в сценарии?",
        "Сколько физических объектов типа Жилой дом?",
        "Сколько сервисов с вместимостью больше 100?",
        "Сколько физических объектов выше 5 этажей?",
    ],
)
def test_specific_filters_cannot_become_a_whole_catalogue_distribution(query):
    assert classify_type_query(query, scenario_selected=True) is None


def test_model_cannot_supply_a_type_id_or_invent_a_candidate():
    candidates = selection_candidates(
        {"service_type": [{"id": 92, "name": "Библиотека"}]}
    )
    with pytest.raises(ValueError):
        validate_selection(ScenarioEntitySelection(candidate="92"), candidates)
    with pytest.raises(ValueError):
        validate_selection(
            ScenarioEntitySelection(candidate="candidate_999"),
            candidates,
        )


def test_records_require_entity_and_type_identity():
    candidate = {"domain": "service_type", "type_id": 92}
    for result in ([{"name": "Школа"}], {"error": "not found"}, [None]):
        with pytest.raises(ValueError):
            verified_entity_records(result, candidate)


def test_exact_matching_requires_one_candidate_across_both_domains():
    candidates = {
        "a": {"name": "Школа", "domain": "service_type"},
        "b": {"name": "Жилой дом", "domain": "physical_object_type"},
    }
    assert exact_type_candidate(" «ШКОЛА» ", candidates) == "a"
    assert exact_type_candidate("школы", candidates) is None
    candidates["c"] = {"name": "Школа", "domain": "physical_object_type"}
    assert exact_type_candidate("Школа", candidates) is None


async def test_literal_type_mapping_cannot_be_overruled_by_a_bad_model():
    from unittest.mock import AsyncMock

    from src.agents.services.scenario_data.scenario_data_type_mapper import (
        UrbanTypeMapper,
    )

    llm = AsyncMock()
    candidates = {"a": {"name": "Школа", "domain": "service_type"}}
    selected = await UrbanTypeMapper(llm).select_scenario_entities(
        "model",
        "Сколько сервисов типа «Школа»?",
        candidates,
        requested_type="Школа",
    )
    assert selected.candidate == "a"
    llm.chat.assert_not_called()


@pytest.mark.parametrize(
    "query",
    [
        "Сколько жилых домов выше 10 этажей?",
        "Покажи школы в радиусе 500 метров",
        "Покажи библиотеки в контексте сценария",
    ],
)
def test_additional_predicates_require_general_planning(query):
    assert not may_select_entities(query)


async def test_general_pipeline_does_not_publish_rejected_draft_or_layers(
    monkeypatch,
    fake_llm,
    fake_urban,
    state_store,
):
    from src.agents.services.service_entities.scenario_data_action import (
        ScenarioDataAction,
        ScenarioDataActionKind,
    )

    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *a, **kw: fake_llm,
    )
    service = ScenarioDataService("http://llm", None, fake_urban, state_store)
    mcp = EntityMcp()
    tool = next(t for t in mcp.tools if t.name == "GetScenarioServicesWithGeometry")
    mcp.tools = [tool]
    mcp.get_tool = lambda group, name: tool
    called = False

    async def choose(*args, **kwargs):
        nonlocal called
        if called:
            return ScenarioDataAction(action=ScenarioDataActionKind.FINAL_ANSWER)
        called = True
        return ScenarioDataAction(
            action=ScenarioDataActionKind.CALL_TOOL,
            group="projects",
            tool_name=tool.name,
            arguments={"service_type_id": 92},
            layer_name="Школы",
        )

    async def draft(*args):
        return "Найдено 1 школа."

    service.plan_builder.choose_action = choose
    service._draft_answer = draft
    fake_llm.json_responses = [
        '{"sufficient": false, "missing_code": "answer_incomplete", "details": "wrong count and type"}'
    ] * 2
    events = [
        e
        async for e in service.run_scenario_data_pipeline(
            mcp,
            "token",
            "model",
            0,
            "Покажи школы на карте",
            scenario_id=17,
            persist_history=False,
        )
    ]
    assert mcp.calls
    assert not any(e["type"] in {"table", "feature_collection"} for e in events)
    text = "".join(e["content"].get("text", "") for e in events if e["type"] == "chunk")
    assert "1 школа" not in text
    assert "Не удалось подтвердить" in text


def test_entity_type_survives_many_constant_address_fields():
    from src.agents.services.scenario_data.scenario_data_aggregate import (
        aggregate_result,
    )

    rows = [
        record(i, domain="physical_object_type", type_id=48, name="Жилой дом")
        for i in range(70)
    ]
    for row in rows:
        row["building"] = {
            "properties": {f"address_{i}_name": "constant" for i in range(12)}
        }
    aggregate = aggregate_result(rows)
    assert aggregate["total_records"] == 70
    assert aggregate["breakdown"]["physical_object_type.name"]["counts"] == {
        "Жилой дом": 70
    }


class GlobalCatalogueMcp(EntityMcp):
    async def load_tools(self):
        return self.tools + [
            UrbanMcpTool(
                group="dictionaries",
                name=name,
                title=name,
                description="",
                tags=(),
                input_schema={"type": "object", "properties": {}},
            )
            for name in ("GetPhysicalObjectTypes", "GetServiceTypes")
        ]

    async def execute_tool(self, group, name, arguments, *, meta):
        if group == "dictionaries":
            assert arguments == {}
            self.calls.append((name, arguments))
            if name == "GetPhysicalObjectTypes":
                return [{"physical_object_type_id": 48, "name": "Жилой дом"}]
            return [{"service_type_id": 92, "name": "Библиотека"}]
        if name == "GetScenarioServiceTypes":
            self.calls.append((name, arguments))
            return []
        return await super().execute_tool(group, name, arguments, meta=meta)


@pytest.mark.parametrize("global_operation", ["count", "unsupported"])
async def test_global_catalogue_cannot_authorize_unfiltered_fallback(
    monkeypatch,
    fake_llm,
    fake_urban,
    state_store,
    global_operation,
):
    events, mcp = await run_query(
        monkeypatch,
        fake_llm,
        fake_urban,
        state_store,
        "unmatched",
        mcp=GlobalCatalogueMcp(rows=[]),
        selections=[
            {"candidate": None},
            {
                "candidate": "candidate_2" if global_operation == "count" else None,
            },
        ],
    )
    text = "".join(e["content"].get("text", "") for e in events if e["type"] == "chunk")
    if global_operation == "count":
        assert "«Библиотека»: 0" in text
        assert mcp.calls[-1] == (
            "GetScenarioServices",
            {"scenario_id": 17, "service_type_id": 92},
        )
    else:
        assert len(mcp.calls) == 4
        assert not any(e["type"] in {"table", "feature_collection"} for e in events)
        assert "не найден тип" in text
