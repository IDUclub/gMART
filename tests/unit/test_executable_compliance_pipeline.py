import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.schema.restrictions_response import RestrictionsResponse
from src.agents.services.compilance.compliance_executor import (
    ComplianceTemplateExecutor,
)
from src.agents.services.compilance.compliance_retrieval import ScenarioComplianceSource
from src.agents.services.compilance.compliance_scope import ScopeOutcome
from src.agents.services.compilance.compliance_territory import TerritoryDocuments
from src.agents.services.pipeline_state import PipelineStep
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from tests.unit.test_compliance_phase2 import IDS, Llm, Mcp
from tests.unit.test_compliance_phase2 import plan as executable_plan
from tests.unit.test_compliance_phase2 import run as run_checks
from tests.unit.test_compliance_phase2 import service as compliance_service


def _plan():
    return executable_plan(0, radius=50)


async def test_pipeline_emits_and_checkpoints_structured_compliance_events(tmp_path):
    service, checkpoints = compliance_service(tmp_path)
    events = await run_checks(service, Mcp(), plans=[executable_plan(0)])
    for event in events:
        RestrictionsResponse.model_validate(event)
    event_types = [event["type"] for event in events]
    assert {
        "check_plan",
        "requirement_resolution",
        "compliance_result",
        "compliance_summary",
    } <= set(event_types)
    progress = [
        event["content"] for event in events if event["type"] == "compliance_progress"
    ]
    assert progress[0]["completed_norms"] == 0
    assert progress[-1]["completed_norms"] == progress[-1]["total_norms"] == 1
    for step in (
        PipelineStep.CHECK_PLAN_VALIDATION,
        PipelineStep.REQUIREMENTS_RESOLUTION,
        PipelineStep.TEMPLATE_EXECUTION,
        PipelineStep.VERDICT_AGGREGATION,
    ):
        assert step in checkpoints
    summary = next(
        event["content"] for event in events if event["type"] == "compliance_summary"
    )
    assert summary == checkpoints[PipelineStep.VERDICT_AGGREGATION]
    assert summary["total_norms"] == 1
    assert summary["results"][0]["archive_path"].startswith("violated/")
    assert not {"violated_features", "passed_features"} & summary["results"][0].keys()


async def test_old_checkpoint_missing_and_unsupported_plans_never_reach_executor(
    tmp_path, monkeypatch
):
    service, checkpoints = compliance_service(tmp_path)
    service.compliance_executor.execute = AsyncMock(
        side_effect=AssertionError("Неисполняемые планы не запускаются")
    )
    source_loading = AsyncMock(
        side_effect=AssertionError("Источники не нужны без исполнимых норм")
    )
    monkeypatch.setattr(ScenarioComplianceSource, "load_sources", source_loading)
    unsupported = executable_plan(0)
    unsupported["planner_status"] = "unsupported"
    invalid = executable_plan(1)
    invalid["template_version"] = 999
    hits = [
        {"id": "without-plan"},
        {"id": IDS[0], "check_plan": unsupported},
        {"id": IDS[1], "check_plan": invalid},
    ]
    old_checkpoint = {PipelineStep.NORMGRAPH: {"restrictions": hits}}
    mcp = Mcp()
    events = [
        event
        async for event in service._run_executable_compliance(
            mcp_client=mcp,
            request_id="old-checkpoint",
            scenario_id=772,
            restrictions=old_checkpoint[PipelineStep.NORMGRAPH]["restrictions"],
            checkpoint=old_checkpoint,
        )
    ]
    service.compliance_executor.execute.assert_not_awaited()
    source_loading.assert_not_awaited()
    assert mcp.calls == []
    assert not any(
        event["type"] == "status" and event["content"]["status"] == "source_layers"
        for event in events
    )
    summary = next(
        event["content"] for event in events if event["type"] == "compliance_summary"
    )
    assert summary["total_norms"] == 0
    assert summary["skipped_without_plan"] == 3
    assert summary["counts"]["not_checked"] == 3
    assert summary == checkpoints[PipelineStep.VERDICT_AGGREGATION]


async def test_one_norm_failure_does_not_prevent_the_next_plan(tmp_path):
    class FirstCheckFails(Mcp):
        async def execute_tool(self, name, arguments, meta=None):
            if (
                name == "CheckDistanceFromSource"
                and arguments["restriction_id"] == IDS[0]
            ):
                raise RuntimeError("Первая проверка недоступна")
            return await super().execute_tool(name, arguments, meta)

    service, checkpoints = compliance_service(tmp_path)
    events = await run_checks(service, FirstCheckFails())
    results = [
        event["content"] for event in events if event["type"] == "compliance_result"
    ]
    assert [result["restriction_id"] for result in results] == IDS
    assert results[0]["verification_status"] == "unverifiable"
    assert results[0]["status"] == "not_checked"
    assert results[1]["status"] == "violated"
    summary = next(
        event["content"] for event in events if event["type"] == "compliance_summary"
    )
    assert summary["total_norms"] == 2 and summary["counts"] == {
        "not_checked": 1,
        "violated": 1,
    }
    assert len(checkpoints[PipelineStep.TEMPLATE_EXECUTION]) == 2
    assert events[-1]["content"]["mime_type"] == "application/zip"


async def test_source_metadata_reaches_layers_summary_and_checkpoint(tmp_path):
    service, checkpoints = compliance_service(tmp_path)
    raw = executable_plan(0)
    raw["source"] = {"restriction_id": IDS[0]}
    source = {
        "document_name": "СП 42.13330.2016 Градостроительство",
        "clause_number": "9.2",
        "extraction_text": "Полный исходный текст: расстояние до школы не менее 100 м.",
    }
    hit = {
        "id": IDS[0],
        "check_plan": raw,
        "provenance": {
            "name": source["document_name"],
            "numbering": source["clause_number"],
        },
        "extraction_text": source["extraction_text"],
    }
    mcp = Mcp()
    events = [
        event
        async for event in service._run_executable_compliance(
            mcp_client=mcp,
            request_id="metadata",
            scenario_id=772,
            restrictions=[hit],
            checkpoint={},
            model="mock",
        )
    ]
    summary = next(
        event["content"] for event in events if event["type"] == "compliance_summary"
    )
    result = summary["results"][0]
    assert {key: result["source"][key] for key in source} == source
    assert summary == checkpoints[PipelineStep.VERDICT_AGGREGATION]
    assert (
        checkpoints[PipelineStep.CHECK_PLAN_VALIDATION][0]["source"]["extraction_text"]
        == source["extraction_text"]
    )
    assert checkpoints[PipelineStep.TEMPLATE_EXECUTION][0]["source"] == result["source"]
    layer = next(
        event["content"] for event in events if event["type"] == "feature_collection"
    )
    assert layer["name"] == "Норма — СП 42.13330.2016, п. 9.2"
    for feature in layer["feature_collection"]["features"]:
        properties = feature["properties"]
        assert properties["norm_id"] == IDS[0]
        assert properties["norm_text"] == source["extraction_text"]
        assert properties["document_name"] == source["document_name"]
        assert properties["clause_number"] == source["clause_number"]
    check_args = next(
        args for name, args in mcp.calls if name == "CheckDistanceFromSource"
    )
    assert {key: check_args["provenance"][key] for key in source} == source


@pytest.mark.parametrize("inventory", [False, True])
async def test_empty_plans_do_not_load_sources_or_initialize_executor(
    tmp_path, monkeypatch, inventory
):
    service, checkpoints = compliance_service(tmp_path)
    del service.compliance_executor
    del service.zone_builder
    service.llm_client.chat = AsyncMock(
        side_effect=AssertionError("Пустая проверка не требует LLM")
    )
    source_loading = AsyncMock(
        side_effect=AssertionError("Нет планов для загрузки источников")
    )
    monkeypatch.setattr(ScenarioComplianceSource, "load_sources", source_loading)
    mcp = Mcp()
    method = (
        service._run_restriction_inventory
        if inventory
        else service._run_executable_compliance
    )
    events = [
        event
        async for event in method(
            mcp_client=mcp,
            request_id="empty",
            scenario_id=772,
            restrictions=[],
            checkpoint={},
        )
    ]
    source_loading.assert_not_awaited()
    service.llm_client.chat.assert_not_awaited()
    assert not hasattr(service, "compliance_executor")
    assert not hasattr(service, "zone_builder")
    assert mcp.calls == []
    summary_type = "restriction_inventory" if inventory else "compliance_summary"
    summary = next(
        event["content"] for event in events if event["type"] == summary_type
    )
    assert summary["total_norms"] == 0
    assert checkpoints["compliance_completed"] is True


async def test_execution_dependencies_are_initialized_lazily(tmp_path):
    service, _ = compliance_service(tmp_path)
    del service.compliance_executor
    del service.zone_builder
    events = await run_checks(service, Mcp(), plans=[executable_plan(0)])
    assert isinstance(service.compliance_executor, ComplianceTemplateExecutor)
    assert service.zone_builder.executor is service.compliance_executor
    summary = next(
        event["content"] for event in events if event["type"] == "compliance_summary"
    )
    assert summary["total_norms"] == 1
    assert summary["results"][0]["status"] == "violated"


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
            checkpoint={"compliance_completed": True},
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
    for index, rid in enumerate(["r1", "r2"]):
        plan = deepcopy(_plan())
        plan["source"]["restriction_id"] = rid
        plan["params"]["distance_m"] += index
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
    service.compliance_territory = SimpleNamespace(
        resolve=AsyncMock(
            return_value=TerritoryDocuments(status="ok", allowed=("СП 42.13330.2016",))
        )
    )
    service.normgraph_retriever = NormGraphRestrictionRetriever(llm)
    service._build_plan = AsyncMock(side_effect=AssertionError("No LLM replanning"))

    def page(limit, after_id=None, **filters):
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
    service.compliance_executor = ComplianceTemplateExecutor(
        effects_client_factory=None
    )
    execute_template = service.compliance_executor.execute

    async def execute(_client, plan, scenario_id, *, user_id=None, layer_source=None):
        rid = plan["source"]["restriction_id"]
        if rid == "r2":
            assert "result:r1" in seen
        seen.append("execute:" + rid)
        return await execute_template(
            _client, plan, scenario_id, user_id=user_id, layer_source=layer_source
        )

    service.compliance_executor.execute = AsyncMock(side_effect=execute)
    service.llm_client = Llm()
    mcp = Mcp()
    events = []
    async for event in service._run_restriction_execution_pipline(
        mcp_client=mcp,
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
    assert "discard-me" not in json.dumps(service.llm_client.messages)
    summary = next(
        event["content"] for event in events if event["type"] == "compliance_summary"
    )
    assert summary["total_norms"] == 2
    assert any("Пропущено норм" in event["content"].get("text", "") for event in events)


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
