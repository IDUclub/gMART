"""Before/after comparison across services: contracts and real pipeline events."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.common.exceptions.token_exceptions import TokenExpiredError
from src.agents.services.pipeline_state import PipelineStatus, PipelineStep
from src.agents.services.provision.provision_context import ProvisionContextBuilder
from src.agents.services.provision.provision_plan_builder import ProvisionPlanBuilder
from src.agents.services.provision.provision_tool_executor import ProvisionToolExecutor
from src.agents.services.provision.provsion_service import (
    BASE_SCENARIO_EFFECTS_MESSAGE,
    EFFECTS_SUMMARY_ANALYSIS_INSTRUCTIONS,
    POPULATION_HINT,
    ProvisionService,
)
from src.agents.services.service_entities.provision_plan import ProvisionPlan


def effects(delta=0.2):
    metrics = {
        "provision_before": 0.4,
        "provision_after": 0.4 + delta,
        "provision_delta": delta,
        "deficit_before": 100,
        "deficit_after": 50,
        "deficit_delta": -50,
        "unmet_demand_after": 60,
        "total_capacity_after": 200,
        "unused_metric": "large unwanted context",
    }
    fc = {"type": "FeatureCollection", "features": []}
    return {
        "text_pivot": json.dumps(
            {"all": metrics, "project": metrics, "context": {"secret": 123}}
        ),
        "pivot": {"large": "raw pivot"},
        "before_prove_data": {"buildings": fc},
        "after_prove_data": {"services": fc},
        "effects": fc,
    }


@pytest.mark.parametrize("missing", [None, "", "   "])
async def test_effects_without_service_falls_back_to_all(missing):
    llm = SimpleNamespace(
        chat=AsyncMock(
            return_value={
                "message": {
                    "content": json.dumps(
                        {
                            "mode": "effects",
                            "service_name": missing,
                            "target_population": 123,
                        }
                    )
                }
            }
        )
    )
    plan = await ProvisionPlanBuilder(llm).build_plan("m", "до и после", ["Школы"])
    assert plan.mode == "effects_summary"
    assert plan.service_names == []
    assert plan.target_population == 123


async def test_planner_schema_prompt_and_canonical_names():
    llm = SimpleNamespace(
        chat=AsyncMock(
            return_value={
                "message": {
                    "content": json.dumps(
                        {
                            "mode": "effects_summary",
                            "service_names": [" школы ", "Школы", "unknown"],
                            "layer_service_names": ["школы"],
                        }
                    )
                }
            }
        )
    )
    plan = await ProvisionPlanBuilder(llm).build_plan("m", "сравни", ["Школы"])
    assert plan.service_names == plan.layer_service_names == ["Школы"]
    kwargs = llm.chat.call_args.kwargs
    assert "effects_summary" in kwargs["format"]["$defs"]["ProvisionPlanMode"]["enum"]
    prompt = kwargs["messages"][0]["content"]
    for example in (
        "сравни обеспеченность по всем сервисам до и после реализации проекта",
        "как проект повлияет на обеспеченность всеми сервисами?",
        "сравнение эффектов обеспеченности по всем сервисам",
        "какие сервисы проект улучшит сильнее всего?",
    ):
        assert example in prompt
    assert '"summary" и "effects_summary"' in prompt


@pytest.mark.parametrize(
    "mode,fields",
    [
        ("effects", {"service_name": "unknown"}),
        ("effects_summary", {"service_names": ["unknown"]}),
    ],
)
def test_unknown_service_requires_clarification(mode, fields):
    plan = ProvisionPlanBuilder(None)._canonicalize_plan(
        ProvisionPlan(mode=mode, **fields), ["Школы"]
    )
    assert plan.mode == "needs_clarification"


def service_for(store, plan):
    svc = object.__new__(ProvisionService)
    svc.urban_api_client = SimpleNamespace(
        get_project_base_scenario_id=AsyncMock(return_value=771)
    )
    svc.state_store = store
    svc.resolve_model = AsyncMock(return_value="m")
    svc.context_builder = ProvisionContextBuilder()
    svc.tool_executor = ProvisionToolExecutor()
    svc._resolve_service_plan = AsyncMock(
        return_value=(plan, {"Школы": 22, "Сады": 21})
    )
    svc.analysis_calls = []

    async def analysis(*args, **kwargs):
        svc.analysis_calls.append((args, kwargs))
        yield svc._chunk("Сравнение сервисов.", done=False)
        yield svc._chunk(kwargs["trailing_note"], done=True)

    svc._generate_analysis = analysis
    return svc


async def run(svc, mcp, **kwargs):
    return [
        event
        async for event in svc._run_provision_pipeline(
            idu_mcp_client=object(),
            effects_mcp_client=mcp,
            model="m",
            temperature=0,
            user_query="Сравни обеспеченность по всем сервисам до и после проекта",
            scenario_id=772,
            token_ref=["t"],
            persist_history=False,
            **kwargs,
        )
    ]


@pytest.mark.parametrize("population", [None, 25000])
async def test_calls_layers_compact_checkpoint_analysis_and_resume(
    state_store, population
):
    plan = ProvisionPlan(
        mode="effects_summary",
        layer_service_names=["Школы", "Сады"],
        target_population=population,
    )
    svc = service_for(state_store, plan)
    mcp = SimpleNamespace(calculate_object_effects=AsyncMock(return_value=effects()))
    events = await run(svc, mcp)
    request_id = events[0]["content"]["request_id"]
    assert mcp.calculate_object_effects.await_count == 2
    assert {
        call.kwargs["service_type_id"]
        for call in mcp.calculate_object_effects.await_args_list
    } == {21, 22}
    assert all(
        call.kwargs["scenario_id"] == 772
        and call.kwargs["target_population"] == population
        for call in mcp.calculate_object_effects.await_args_list
    )
    tools = [e["content"] for e in events if e["type"] == "tool_call"]
    assert len(tools) == 1
    assert tools[0]["mcp_source"] == "OBJECTS_EFFECTS_MCP_URL"
    assert [call["function"]["name"] for call in tools[0]["tool_calls"]] == [
        "CalculateObjectEffects"
    ] * 2
    layers = [e["content"]["name"] for e in events if e["type"] == "feature_collection"]
    assert len(layers) == len(set(layers)) == 6
    assert any("Школы" in name for name in layers) and any(
        "Сады" in name for name in layers
    )
    assert sum(e["type"] == "table" for e in events) == 1
    args, kwargs = svc.analysis_calls[0]
    assert kwargs["instructions"] == EFFECTS_SUMMARY_ANALYSIS_INSTRUCTIONS
    assert (
        kwargs["trailing_note"] == POPULATION_HINT
        if population is None
        else str(population) in args[2]
    )
    checkpoint = await state_store.get_checkpoint(request_id)
    assert checkpoint[PipelineStep.FINAL_RESPONSE] is True
    assert (await state_store.get_state(request_id))["status"] == PipelineStatus.DONE
    await run(svc, mcp, request_id=request_id)
    assert mcp.calculate_object_effects.await_count == 2
    assert len(svc.analysis_calls) == 1


@pytest.mark.parametrize("layers", [[], ["Школы"]])
async def test_only_requested_layers_are_saved_and_streamed(state_store, layers):
    svc = service_for(
        state_store, ProvisionPlan(mode="effects_summary", layer_service_names=layers)
    )
    events = await run(
        svc, SimpleNamespace(calculate_object_effects=AsyncMock(return_value=effects()))
    )
    checkpoint = await state_store.get_checkpoint(events[0]["content"]["request_id"])
    services = checkpoint[PipelineStep.CALCULATE_EFFECTS_SUMMARY]["services"]
    for entry in services.values():
        assert ("effects" in entry) == (entry["name"] in layers)
    layer_events = [e for e in events if e["type"] == "feature_collection"]
    assert len(layer_events) == len(layers) * 3
    assert all("Школы" in event["content"]["name"] for event in layer_events)


@pytest.mark.parametrize("all_failed", [False, True])
async def test_failures_logged_private_and_partial_results_continue(
    state_store, all_failed
):
    svc = service_for(state_store, ProvisionPlan(mode="effects_summary"))

    async def calculate(**kwargs):
        if all_failed or kwargs["service_type_id"] == 22:
            raise RuntimeError("PRIVATE_INTERNAL_ERROR service_type_id=22")
        return effects()

    logs = []
    sink = logger.add(lambda message: logs.append(str(message)))
    try:
        events = await run(svc, SimpleNamespace(calculate_object_effects=calculate))
    finally:
        logger.remove(sink)
    request_id = events[0]["content"]["request_id"]
    assert "PRIVATE_INTERNAL_ERROR" not in json.dumps(events)
    assert any(
        request_id in line
        and "scenario_id=772" in line
        and "service_type_id=22" in line
        and "PRIVATE_INTERNAL_ERROR" in line
        for line in logs
    )
    state = await state_store.get_state(request_id)
    if all_failed:
        assert state["status"] == PipelineStatus.FAILED
        assert events[-1]["type"] == "error"
        assert not svc.analysis_calls
    else:
        assert state["status"] == PipelineStatus.DONE
        table = next(e["content"] for e in events if e["type"] == "table")
        assert table["rows"][-1]["status"] == "Нет данных"
        assert svc.analysis_calls
        assert "PRIVATE_INTERNAL_ERROR" not in svc.analysis_calls[0][0][2]


async def test_token_expiration_propagates_without_checkpoint(state_store):
    svc = service_for(state_store, ProvisionPlan(mode="effects_summary"))
    state_store.new_request_id = lambda: "expired-effects"
    mcp = SimpleNamespace(
        calculate_object_effects=AsyncMock(side_effect=TokenExpiredError("expired"))
    )
    with pytest.raises(TokenExpiredError):
        await run(svc, mcp, request_id="expired-effects-run")
    checkpoint = await state_store.get_checkpoint("expired-effects-run")
    assert PipelineStep.CALCULATE_EFFECTS_SUMMARY not in checkpoint
    assert PipelineStep.FINAL_RESPONSE not in checkpoint
    checkpoint = await state_store.get_checkpoint("expired-effects")
    assert PipelineStep.CALCULATE_EFFECTS_SUMMARY not in checkpoint


async def test_resume_after_analysis_interruption_skips_calculations(state_store):
    svc = service_for(state_store, ProvisionPlan(mode="effects_summary"))
    state_store.new_request_id = lambda: "resume-effects"
    mcp = SimpleNamespace(calculate_object_effects=AsyncMock(return_value=effects()))
    original_analysis = svc._generate_analysis

    async def interrupted(*args, **kwargs):
        raise RuntimeError("analysis interrupted")
        yield  # pragma: no cover

    svc._generate_analysis = interrupted
    with pytest.raises(RuntimeError, match="analysis interrupted"):
        await run(svc, mcp)
    checkpoint = await state_store.get_checkpoint("resume-effects")
    assert PipelineStep.CALCULATE_EFFECTS_SUMMARY in checkpoint
    assert PipelineStep.FINAL_RESPONSE not in checkpoint
    svc._generate_analysis = original_analysis
    svc.urban_api_client.get_project_base_scenario_id.reset_mock()
    events = await run(svc, mcp, request_id="resume-effects")
    svc.urban_api_client.get_project_base_scenario_id.assert_not_awaited()
    assert mcp.calculate_object_effects.await_count == 2
    assert events[-1]["content"]["done"]
    assert len(svc.analysis_calls) == 1


async def test_concurrency_is_bounded_and_run_is_kept_alive(state_store):
    svc = service_for(state_store, ProvisionPlan(mode="effects_summary"))
    svc._resolve_service_plan.return_value = (
        ProvisionPlan(mode="effects_summary"),
        {f"service {i}": i for i in range(8)},
    )
    active = maximum = 0

    async def calculate(**kwargs):
        nonlocal active, maximum
        active += 1
        maximum = max(active, maximum)
        await asyncio.sleep(0.01)
        active -= 1
        return effects()

    original = state_store.keep_alive
    state_store.keep_alive = AsyncMock(wraps=original)
    await run(svc, SimpleNamespace(calculate_object_effects=calculate))
    assert maximum == 3
    assert state_store.keep_alive.await_count >= 1


async def test_explicit_service_subset(state_store):
    svc = service_for(
        state_store, ProvisionPlan(mode="effects_summary", service_names=["Сады"])
    )
    mcp = SimpleNamespace(calculate_object_effects=AsyncMock(return_value=effects()))
    await run(svc, mcp)
    mcp.calculate_object_effects.assert_awaited_once_with(
        service_type_id=21, scenario_id=772, target_population=None
    )


async def test_resume_calculation_checkpoint_still_generates_answer(state_store):
    plan = ProvisionPlan(mode="effects_summary")
    svc = service_for(state_store, plan)
    request_id = "effects-resume"
    await state_store.create(
        request_id,
        chat_id=None,
        user_query="compare",
        scenario_id=772,
        model="m",
        temperature=0,
    )
    await state_store.save_checkpoint(
        request_id,
        PipelineStep.RESOLVE_SERVICE,
        {
            "plan": plan.model_dump(mode="json"),
            "service_types": {"Школы": 22},
        },
    )
    await state_store.save_checkpoint(
        request_id,
        PipelineStep.CALCULATE_EFFECTS_SUMMARY,
        {
            "services": {"22": {"name": "Школы", "error": None, **effects()}},
        },
    )
    mcp = SimpleNamespace(calculate_object_effects=AsyncMock())
    events = await run(svc, mcp, request_id=request_id)
    mcp.calculate_object_effects.assert_not_awaited()
    svc.urban_api_client.get_project_base_scenario_id.assert_not_awaited()
    assert len(svc.analysis_calls) == 1
    assert any(event["type"] == "table" for event in events)
    assert (await state_store.get_state(request_id))["status"] == PipelineStatus.DONE


@pytest.mark.parametrize("payload", [{}, {"text_pivot": "invalid JSON"}])
async def test_missing_calculation_data_is_a_failure(state_store, payload):
    svc = service_for(state_store, ProvisionPlan(mode="effects_summary"))
    mcp = SimpleNamespace(calculate_object_effects=AsyncMock(return_value=payload))
    events = await run(svc, mcp)
    assert events[-1]["type"] == "error"
    assert not svc.analysis_calls
    request_id = events[0]["content"]["request_id"]
    assert (await state_store.get_state(request_id))["status"] == PipelineStatus.FAILED


async def test_base_scenario_skips_effects_summary(state_store):
    svc = service_for(state_store, ProvisionPlan(mode="effects_summary"))
    svc.urban_api_client.get_project_base_scenario_id.return_value = 772
    mcp = SimpleNamespace(calculate_object_effects=AsyncMock())
    events = await run(svc, mcp)
    mcp.calculate_object_effects.assert_not_awaited()
    svc.urban_api_client.get_project_base_scenario_id.assert_awaited_once_with("t", 772)
    assert [e for e in events if e["type"] == "chunk"] == [
        svc._chunk(BASE_SCENARIO_EFFECTS_MESSAGE, done=True)
    ]
    assert any(e["type"] == "status" for e in events)
    assert not any(e["type"] in {"tool_call", "table"} for e in events)
    assert not svc.analysis_calls
    request_id = events[0]["content"]["request_id"]
    assert (await state_store.get_state(request_id))["status"] == PipelineStatus.DONE


@pytest.mark.parametrize("base_id", [771, None, RuntimeError("lookup failed")])
async def test_base_lookup_fallback_runs_effects_summary(state_store, base_id):
    svc = service_for(state_store, ProvisionPlan(mode="effects_summary"))
    lookup = svc.urban_api_client.get_project_base_scenario_id
    if isinstance(base_id, Exception):
        lookup.side_effect = base_id
    else:
        lookup.return_value = base_id
    mcp = SimpleNamespace(calculate_object_effects=AsyncMock(return_value=effects()))
    logs = []
    sink = logger.add(lambda message: logs.append(message.record))
    try:
        events = await run(svc, mcp)
    finally:
        logger.remove(sink)
    assert mcp.calculate_object_effects.await_count == 2
    assert svc.analysis_calls
    assert events[-1]["content"]["done"]
    if base_id is None or isinstance(base_id, Exception):
        assert any(record["level"].name == "WARNING" for record in logs)


async def test_base_lookup_token_expiration_propagates(state_store):
    svc = service_for(state_store, ProvisionPlan(mode="effects_summary"))
    svc.urban_api_client.get_project_base_scenario_id.side_effect = TokenExpiredError(
        "expired"
    )
    mcp = SimpleNamespace(calculate_object_effects=AsyncMock())
    with pytest.raises(TokenExpiredError):
        await run(svc, mcp)
    mcp.calculate_object_effects.assert_not_awaited()


@pytest.mark.parametrize(
    "scenario,project,expected",
    [
        (
            {"project": {"project_id": 604}, "is_based": True},
            {"base_scenario": {"id": 772}},
            772,
        ),
        ({}, {}, None),
        ({"project": None}, {}, None),
        ({"project": {}}, {}, None),
        ({"project": {"project_id": None}}, {}, None),
        ({"project": {"project_id": 604}}, {}, None),
        ({"project": {"project_id": 604}}, {"base_scenario": None}, None),
        ({"project": {"project_id": 604}}, {"base_scenario": {}}, None),
        ({"project": {"project_id": 604}}, {"base_scenario": {"id": None}}, None),
    ],
)
async def test_get_project_base_scenario_id(scenario, project, expected):
    handler = SimpleNamespace(get=AsyncMock(side_effect=[scenario, project]))
    assert (
        await UrbanApiClient(handler).get_project_base_scenario_id("t", 1146)
        == expected
    )
    assert handler.get.await_args_list[0].args == ("/v1/scenarios/1146",)
    assert handler.get.await_args_list[0].kwargs == {"auth_token": "t"}
    if (scenario.get("project") or {}).get("project_id") is None:
        assert handler.get.await_count == 1
    else:
        assert handler.get.await_count == 2
        assert handler.get.await_args_list[1].args == ("/v1/projects/604",)
        assert handler.get.await_args_list[1].kwargs == {"auth_token": "t"}
