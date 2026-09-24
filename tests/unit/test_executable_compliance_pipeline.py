import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.schema.restrictions_response import RestrictionsResponse
from src.agents.services.compilance.compliance_scope import ScopeOutcome
from src.agents.services.pipeline_state import PipelineStep
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.service_entities.compliance import (
    ComplianceResult,
    ComplianceSummary,
    VerificationCoverage,
)


def _plan():
    return {
        "schema_version": "1.0",
        "template": "distance_from_source",
        "template_version": 1,
        "params": {
            "source_layer": "source",
            "targets": ["targets"],
            "geometry_mode": "buffered",
            "distance_m": 50,
            "predicate": "intersects",
            "violation_when": "matched",
            "result_mode": "both",
        },
        "declared_requirements": {"layers": [], "attributes": []},
        "source": {"restriction_id": "r1"},
        "planner_status": "auto",
    }


async def test_pipeline_emits_and_checkpoints_structured_compliance_events():
    service = object.__new__(RestrictionParserService)
    service.state_store = SimpleNamespace(
        buffer_event=AsyncMock(),
        save_checkpoint=AsyncMock(),
        set_status=AsyncMock(),
    )
    result = ComplianceResult(
        restriction_id="r1",
        template="distance_from_source",
        template_version=1,
        verification_status="complete",
        compliance_status="passed",
        coverage=VerificationCoverage(
            applicable_objects=1,
            checked_objects=1,
            unchecked_objects=0,
            fill_rate=1,
        ),
        summary=ComplianceSummary(violated_objects=0, passed_objects=1),
        source={"restriction_id": "r1"},
    )
    service.compliance_executor = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(
                result=result,
                tool_calls=[
                    {"function": {"name": "CheckDistanceFromSource", "arguments": {}}}
                ],
                timings_ms={"template_execution": 1.0},
            )
        )
    )

    events = [
        event
        async for event in service._run_executable_compliance(
            mcp_client=object(),
            request_id="request-1",
            scenario_id=772,
            restrictions=[{"id": "r1", "check_plan": _plan()}],
            checkpoint={},
        )
    ]
    await asyncio.sleep(0)
    event_types = [event["type"] for event in events]
    for event in events:
        RestrictionsResponse.model_validate(event)
    assert "check_plan" in event_types
    assert "requirement_resolution" in event_types
    assert "compliance_result" in event_types
    assert event_types.count("compliance_progress") == 2
    assert "compliance_summary" in event_types
    progress = [
        event["content"] for event in events if event["type"] == "compliance_progress"
    ]
    assert progress[0]["completed_norms"] == 0
    assert progress[-1]["completed_norms"] == progress[-1]["total_norms"] == 1
    checkpoints = [
        call.args[1] for call in service.state_store.save_checkpoint.await_args_list
    ]
    assert checkpoints == [
        PipelineStep.CHECK_PLAN_VALIDATION,
        PipelineStep.REQUIREMENTS_RESOLUTION,
        PipelineStep.TEMPLATE_EXECUTION,
        PipelineStep.VERDICT_AGGREGATION,
    ]


async def test_completed_reconnect_does_not_reexecute_templates():
    service = object.__new__(RestrictionParserService)
    service.state_store = SimpleNamespace(set_status=AsyncMock())
    service.compliance_executor = SimpleNamespace(execute=AsyncMock())
    events = [
        event
        async for event in service._run_executable_compliance(
            mcp_client=object(),
            request_id="request-1",
            scenario_id=772,
            restrictions=[],
            checkpoint={PipelineStep.VERDICT_AGGREGATION: {}},
        )
    ]
    assert events == []
    service.compliance_executor.execute.assert_not_awaited()


@pytest.mark.parametrize("persist_history", [True, False])
async def test_compliance_history_keeps_tool_calls_and_text_without_large_results(
    persist_history,
):
    service = object.__new__(RestrictionParserService)
    service.resolve_model = AsyncMock(return_value="model")
    service.add_complex_message = AsyncMock()
    oversized_result = {"evidence": "x" * (17 * 1024 * 1024)}
    events = [
        {
            "type": "status",
            "content": {"status": "template_execution", "text": "Проверяю нормы"},
        },
        service._tool_call(
            "normgraph_search",
            [
                {
                    "function": {
                        "name": "search_restrictions",
                        "arguments": {"query": "школы"},
                    }
                }
            ],
            "NORM_GRAPH_MCP_URL",
        ),
        service._tool_call(
            "template_execution",
            [
                {
                    "function": {
                        "name": "CheckDistanceFromSource",
                        "arguments": {"distance_m": 50},
                    }
                }
            ],
            "IDU_MCP_URL",
        ),
        *[
            {"type": event_type, "content": oversized_result}
            for event_type in (
                "check_plan",
                "requirement_resolution",
                "compliance_result",
                "compliance_summary",
            )
        ],
        {"type": "compliance_progress", "content": {"completed_norms": 1}},
        service._chunk("Проверка ", done=False),
        service._chunk("завершена.", done=True),
    ]

    async def run_inner(**kwargs):
        for event in events:
            yield event

    service._run_restriction_execution_pipline = run_inner
    streamed = [
        event
        async for event in service.run_compliance_pipeline(
            mcp_client=object(),
            token="user-token",
            temperature=0,
            model="model",
            user_query="Проверь нормы",
            scenario_id=845,
            normgraph_mcp_client=object(),
            chat_id="chat-1",
            persist_history=persist_history,
        )
    ]
    await asyncio.sleep(0)

    # Full structured results still reach the caller, including oversized evidence.
    assert streamed == [event for event in events if event["type"] != "tool_call"]
    if not persist_history:
        service.add_complex_message.assert_not_awaited()
        return

    service.add_complex_message.assert_awaited_once()
    call = service.add_complex_message.await_args
    assert call.args[:2] == ("user-token", "chat-1")
    assert call.kwargs == {"scenario_id": 845}
    parts = [part.model_dump(mode="json") for part in call.args[3]]
    assert [part["kind"] for part in parts] == ["tool_call", "tool_call", "text"]
    assert [part["mcp_source"] for part in parts[:2]] == [
        "NORM_GRAPH_MCP_URL",
        "IDU_MCP_URL",
    ]
    assert [part["payload"]["calls"][0] for part in parts[:2]] == [
        {
            "step": 1,
            "tool_name": "search_restrictions",
            "arguments": {"query": "школы"},
        },
        {
            "step": 1,
            "tool_name": "CheckDistanceFromSource",
            "arguments": {"distance_m": 50},
        },
    ]
    assert parts[-1]["payload"]["text"] == "Проверка завершена."
    assert len(json.dumps(parts).encode()) < 4096


def test_status_history_is_preserved_outside_compliance():
    event = {"type": "status", "content": {"status": "planning", "text": "План"}}
    part = RestrictionParserService._pipeline_item_to_chat_part(event)
    assert part.kind == "status"
    assert part.payload.text == "План"


def test_compliance_history_keeps_clarification_text():
    part = RestrictionParserService._pipeline_item_to_chat_part(
        {"type": "clarification", "content": {"question": "Какой объект проверить?"}},
        text_only=True,
    )
    assert part.kind == "text"
    assert part.payload.text == "Какой объект проверить?"


@pytest.mark.parametrize("distance", ["20 м", "200 метров", "0,1 км"])
async def test_compliance_large_corpus_never_reaches_llm_and_executes_individually(
    distance,
):
    from copy import deepcopy

    from src.agents.services.normgraph.normgraph_restriction_retriever import (
        NormGraphRestrictionRetriever,
    )

    hits = [
        {"id": f"missing-{i}", "extraction_text": "discard-me" * 10000}
        for i in range(300)
    ]
    for rid in ["r1", "r2"]:
        plan = deepcopy(_plan())
        plan["source"]["restriction_id"] = rid
        hits.append({"id": rid, "check_plan": plan})
    llm = SimpleNamespace(
        chat=AsyncMock(side_effect=AssertionError("No corpus in LLM"))
    )
    service = object.__new__(RestrictionParserService)
    service.state_store = SimpleNamespace(
        new_request_id=lambda: "isolated",
        create=AsyncMock(),
        get_checkpoint=AsyncMock(return_value={}),
        save_checkpoint=AsyncMock(),
        set_status=AsyncMock(),
    )
    service._buf = AsyncMock(side_effect=lambda _id, event: event)
    service.compliance_result_harness = SimpleNamespace(
        prepare_follow_up=lambda *args: None
    )
    service.compliance_scope = SimpleNamespace(
        resolve=AsyncMock(return_value=ScopeOutcome(kind="scoped"))
    )
    service.normgraph_retriever = NormGraphRestrictionRetriever(llm)
    service._build_plan = AsyncMock(side_effect=AssertionError("No LLM replanning"))

    def page(limit, after_id=None):
        # NormGraph pages by id; the fake keeps list order, which the ids mirror.
        start = 0 if after_id is None else [h["id"] for h in hits].index(after_id) + 1
        rows = hits[start : start + limit + 1]
        more = len(rows) > limit
        return {
            "hits": rows[:limit],
            "next_after_id": rows[limit - 1]["id"] if more else None,
        }

    client = SimpleNamespace(
        list_restrictions=AsyncMock(side_effect=lambda **args: page(**args))
    )
    seen = []

    async def execute(_client, plan, scenario_id):
        rid = plan["source"]["restriction_id"]
        if rid == "r2":
            assert "result:r1" in seen
        seen.append("execute:" + rid)
        result = ComplianceResult(
            restriction_id=rid,
            template=plan["template"],
            template_version=1,
            verification_status="complete",
            compliance_status="passed",
            coverage=VerificationCoverage(
                applicable_objects=1,
                checked_objects=1,
                unchecked_objects=0,
                fill_rate=1,
            ),
            summary=ComplianceSummary(violated_objects=0, passed_objects=1),
        )
        return SimpleNamespace(result=result, tool_calls=[], timings_ms={})

    service.compliance_executor = SimpleNamespace(
        execute=AsyncMock(side_effect=execute)
    )
    events = []
    async for event in service._run_restriction_execution_pipline(
        mcp_client=object(),
        temperature=0,
        model="m",
        user_query=f"Проверь нормы с расстоянием {distance}",
        scenario_id=772,
        token_ref=["test"],
        persist_history=False,
        normgraph_mcp_client=client,
        history_agent="compliance",
    ):
        events.append(event)
        if event["type"] == "compliance_result":
            seen.append("result:" + event["content"]["restriction_id"])
    assert seen == ["execute:r1", "result:r1", "execute:r2", "result:r2"]
    llm.chat.assert_not_awaited()
    service._build_plan.assert_not_awaited()
    checkpoints = service.state_store.save_checkpoint.await_args_list
    graph = next(
        call.args[2] for call in checkpoints if call.args[1] == PipelineStep.NORMGRAPH
    )
    assert [hit["id"] for hit in graph["restrictions"]] == ["r1", "r2"]
    assert graph["unsupported_count"] == 300
    assert "discard-me" not in json.dumps(graph)
    summary = next(
        event["content"] for event in events if event["type"] == "compliance_summary"
    )
    assert summary["total_norms"] == 2
    assert any("Пропущено норм" in event["content"].get("text", "") for event in events)


async def test_old_checkpoint_missing_and_unsupported_plans_never_reach_executor():
    service = object.__new__(RestrictionParserService)
    service.state_store = SimpleNamespace(
        save_checkpoint=AsyncMock(), set_status=AsyncMock()
    )
    service._buf = AsyncMock(side_effect=lambda _id, event: event)
    service.compliance_executor = SimpleNamespace(execute=AsyncMock())
    unsupported = {**_plan(), "planner_status": "unsupported"}
    events = [
        event
        async for event in service._run_executable_compliance(
            mcp_client=object(),
            request_id="old",
            scenario_id=772,
            restrictions=[
                {"id": "absent"},
                {"check_plan": {}},
                {"check_plan": unsupported},
            ],
            checkpoint={},
        )
    ]
    service.compliance_executor.execute.assert_not_awaited()
    assert not any(
        event["type"] in {"check_plan", "compliance_result"} for event in events
    )
    assert any(
        "Пропущено норм без исполнимого плана: 3" in event["content"].get("text", "")
        for event in events
    )


async def test_compliance_without_normgraph_does_not_fall_back_to_llm():
    service = object.__new__(RestrictionParserService)
    service.state_store = SimpleNamespace(
        new_request_id=lambda: "none",
        create=AsyncMock(),
        get_checkpoint=AsyncMock(return_value={}),
        set_status=AsyncMock(),
    )
    service._buf = AsyncMock(side_effect=lambda _id, event: event)
    service.compliance_result_harness = SimpleNamespace(
        prepare_follow_up=lambda *args: None
    )
    service._build_plan = AsyncMock(side_effect=AssertionError("No LLM fallback"))
    events = [
        event
        async for event in service._run_restriction_execution_pipline(
            mcp_client=object(),
            temperature=0,
            model="m",
            user_query="Проверь 20 м",
            scenario_id=772,
            token_ref=["test"],
            persist_history=False,
            normgraph_mcp_client=None,
            history_agent="compliance",
        )
    ]
    service._build_plan.assert_not_awaited()
    assert any(
        "NormGraph не подключён" in event["content"].get("text", "") for event in events
    )


async def test_one_norm_failure_does_not_prevent_the_next_plan():
    from copy import deepcopy

    service = object.__new__(RestrictionParserService)
    service.state_store = SimpleNamespace(
        save_checkpoint=AsyncMock(), set_status=AsyncMock()
    )
    service._buf = AsyncMock(side_effect=lambda _id, event: event)
    second = deepcopy(_plan())
    second["source"]["restriction_id"] = "r2"
    success = ComplianceResult(
        restriction_id="r2",
        template=second["template"],
        template_version=1,
        verification_status="complete",
        compliance_status="passed",
        coverage=VerificationCoverage(
            applicable_objects=1, checked_objects=1, unchecked_objects=0, fill_rate=1
        ),
        summary=ComplianceSummary(violated_objects=0, passed_objects=1),
    )
    service.compliance_executor = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                RuntimeError("geometry unavailable"),
                SimpleNamespace(result=success, tool_calls=[], timings_ms={}),
            ]
        )
    )
    events = [
        event
        async for event in service._run_executable_compliance(
            mcp_client=object(),
            request_id="independent",
            scenario_id=772,
            restrictions=[{"check_plan": _plan()}, {"check_plan": second}],
            checkpoint={},
        )
    ]
    results = [
        event["content"] for event in events if event["type"] == "compliance_result"
    ]
    assert [(r["restriction_id"], r["verification_status"]) for r in results] == [
        ("r1", "unverifiable"),
        ("r2", "complete"),
    ]
    assert service.compliance_executor.execute.await_count == 2


@pytest.mark.parametrize(
    "status,verification,violated_count,has_geometry,expected_layers",
    [
        ("violated", "complete", 2, True, 1),
        ("violated", "partial", 2, True, 1),
        ("violated", "complete", 2, False, 0),
        ("passed", "complete", 0, True, 0),
        ("passed", "partial", 0, True, 0),
        ("unknown", "unverifiable", 0, False, 0),
    ],
)
async def test_compliance_ui_only_emits_nonempty_violation_layers(
    status, verification, violated_count, has_geometry, expected_layers
):
    service = object.__new__(RestrictionParserService)
    service.state_store = SimpleNamespace(
        buffer_event=AsyncMock(), save_checkpoint=AsyncMock(), set_status=AsyncMock()
    )
    geometry = {
        "type": "FeatureCollection",
        "features": (
            [
                {
                    "type": "Feature",
                    "properties": {"id": i},
                    "geometry": {"type": "Point", "coordinates": [30, 60]},
                }
                for i in range(2)
            ]
            if has_geometry
            else []
        ),
    }
    result = ComplianceResult(
        restriction_id="r1",
        template="distance_from_source",
        template_version=1,
        verification_status=verification,
        compliance_status=status,
        coverage=VerificationCoverage(
            applicable_objects=3,
            checked_objects=2,
            unchecked_objects=1,
            fill_rate=2 / 3,
        ),
        summary=ComplianceSummary(
            violated_objects=violated_count, passed_objects=2 - violated_count
        ),
        source={
            "document_name": "Тесты Норм",
            "clause_number": "2.1",
            "extraction_text": "Расстояние должно составлять не менее 100 м.",
        },
        violated_features=geometry,
        passed_features=geometry,
    )
    service.compliance_executor = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(result=result, tool_calls=[], timings_ms={})
        )
    )
    events = [
        event
        async for event in service._run_executable_compliance(
            mcp_client=object(),
            request_id="ui",
            scenario_id=772,
            restrictions=[
                {
                    "check_plan": {
                        **_plan(),
                        "source": {"restriction_id": "r1", **result.source},
                    }
                }
            ],
            checkpoint={},
        )
    ]
    layers = [e for e in events if e["type"] == "feature_collection"]
    assert len(layers) == expected_layers
    if layers:
        assert layers[0]["content"]["name"] == "Нарушение нормы — Тесты Норм, п. 2.1"
        assert layers[0]["content"]["feature_collection"] == geometry
    for event in events:
        if event["type"] != "feature_collection":
            payload = json.dumps(event, ensure_ascii=False)
            assert "FeatureCollection" not in payload
            assert "violated_features" not in payload
            assert "passed_features" not in payload
    final_text = next(e["content"]["text"] for e in events if e["type"] == "chunk")
    if status == "violated":
        assert "Тесты Норм, п. 2.1: нарушений на объектах — 2" in final_text
        assert "Расстояние должно составлять не менее 100 м." in final_text
        if verification == "partial":
            assert "Не проверено объектов: 1" in final_text
    else:
        assert "Нарушенные нормы:" not in final_text


def test_compliance_summary_identifies_every_violated_norm_and_keeps_counts_separate():
    summary = dict(
        total_norms=3,
        violated_norms=2,
        passed_norms=1,
        unverifiable_norms=0,
        unsupported_norms=0,
        partial_norms=0,
        results=[
            {
                "restriction_id": "r1",
                "compliance_status": "violated",
                "source": {"document_name": "Документ", "clause_number": "1.1"},
                "summary": {"violated_objects": 17},
            },
            {
                "restriction_id": "r2",
                "compliance_status": "violated",
                "source": {},
                "summary": {"violated_objects": 15},
            },
            {
                "restriction_id": "passed",
                "compliance_status": "passed",
                "source": {},
                "summary": {"violated_objects": 0},
            },
        ],
    )
    text = RestrictionParserService._compliance_summary_text(summary)
    assert "Документ, п. 1.1: нарушений на объектах — 17" in text
    assert "Источник не указан: нарушений на объектах — 15" in text
    assert "(passed)" not in text


@pytest.mark.parametrize(
    "provenance, expected",
    [
        (
            {"name": "СП 42.13330.2016 Градостроительство", "numbering": "7.1"},
            "СП 42.13330.2016, п. 7.1",
        ),
        ({"name": "СП 42.13330.2016"}, "СП 42.13330.2016"),
        ({"numbering": "7.1"}, "Источник не указан, п. 7.1"),
        ({"name": "  \n ", "numbering": " "}, "Источник не указан"),
    ],
)
async def test_source_metadata_reaches_layers_summary_and_checkpoint(
    provenance, expected
):
    from copy import deepcopy

    service = object.__new__(RestrictionParserService)
    service.state_store = SimpleNamespace(
        buffer_event=AsyncMock(), save_checkpoint=AsyncMock(), set_status=AsyncMock()
    )
    geometry = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Point", "coordinates": [30, 60]},
            }
        ],
    }

    async def execute(_client, plan, scenario_id):
        result = ComplianceResult(
            restriction_id=plan["source"]["restriction_id"],
            template=plan["template"],
            template_version=1,
            verification_status="complete",
            compliance_status="violated",
            coverage=dict(
                applicable_objects=1,
                checked_objects=1,
                unchecked_objects=0,
                fill_rate=1,
            ),
            summary=dict(violated_objects=1, passed_objects=0),
            source=plan["source"],
            violated_features=geometry,
        )
        return SimpleNamespace(result=result, tool_calls=[], timings_ms={})

    service.compliance_executor = SimpleNamespace(
        execute=AsyncMock(side_effect=execute)
    )
    hits = []
    for number in range(2):
        plan = _plan()
        plan["source"].update(restriction_id=f"norm-{number}", document_name=" ")
        plan["params"]["distance_m"] += number
        hits.append(
            dict(
                check_plan=plan,
                provenance=provenance,
                extraction_text="Минимальное расстояние",
            )
        )
    original = deepcopy(hits)
    events = [
        event
        async for event in service._run_executable_compliance(
            mcp_client=object(),
            request_id="sources",
            scenario_id=772,
            restrictions=hits,
            checkpoint={},
        )
    ]
    assert hits == original
    names = [
        event["content"]["name"]
        for event in events
        if event["type"] == "feature_collection"
    ]
    assert names == [
        f"Нарушение нормы — {expected}",
        f"Нарушение нормы — {expected} (2)",
    ]
    text = next(
        event["content"]["text"] for event in events if event["type"] == "chunk"
    )
    assert expected in text and "Минимальное расстояние" in text
    assert "norm-0" not in text and "norm-1" not in text
    checkpoints = service.state_store.save_checkpoint.await_args_list
    results = next(
        call.args[2]
        for call in checkpoints
        if call.args[1] == PipelineStep.TEMPLATE_EXECUTION
    )
    assert [result["restriction_id"] for result in results] == ["norm-0", "norm-1"]
    assert results[0]["source"].get("clause_number") == (
        provenance["numbering"].strip() if "numbering" in provenance else None
    )
