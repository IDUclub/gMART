"""Fault injection through the goal-mode orchestration loop and persistence seam."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.runtime.budget import BudgetExceeded
from src.agents.services.orchestrator.analysis_goal import GoalDecision
from tests.unit.test_analytical_orchestrator import final, table
from tests.unit.test_goal_orchestrator import action, contract, school_artifacts
from tests.unit.test_orchestrator_service_events import (
    FakePipeline,
    orchestrator,
    run_pipeline,
)


@pytest.mark.parametrize(
    "resource", ["tokens", "model_calls", "tool_calls", "steps", "context", "time"]
)
async def test_each_budget_stop_retains_goal_evidence_and_free_replay(
    orchestrator, monkeypatch, resource
):
    monkeypatch.setenv("ORCHESTRATOR_ANALYSIS_MODE", "goal")
    orchestrator.goal_manager.create = AsyncMock(return_value=contract())
    data = FakePipeline(school_artifacts())
    orchestrator.scenario_data_service = SimpleNamespace(
        run_scenario_data_pipeline=data
    )

    async def review(*args):
        if not data.calls:
            return action("schools", "scenario_data")
        raise BudgetExceeded(resource)

    orchestrator.goal_manager.review = review
    args = dict(
        user_query="Сравни школы и обеспеченность",
        urban_mcp_client=AsyncMock(),
        persist_history=False,
    )
    events = await run_pipeline(orchestrator, **args)
    result = final(events)
    assert result["status"] == "blocked"
    assert result["missing"] and result["missing"][0]["owner"] == "budget"
    assert result["continue_from"]
    assert result["goal"]["requirements"][0]["status"] == "satisfied"
    assert sum(a["confirmed"] for a in result["artifacts"]) == 3
    replay = await run_pipeline(
        orchestrator, **args, request_id=result["continue_from"]
    )
    assert replay == events
    assert len(data.calls) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "error",
        "failure",
        "pipeline_failed",
        "pipeline_suspended",
        "clarification",
        "clarification_required",
        "exception",
    ],
)
async def test_specialist_fault_keeps_independent_work_and_rejects_draft(
    orchestrator, monkeypatch, fault
):
    monkeypatch.setenv("ORCHESTRATOR_ANALYSIS_MODE", "goal")
    orchestrator.goal_manager.create = AsyncMock(return_value=contract())
    data = FakePipeline(school_artifacts())
    calls = []

    async def failing(**kwargs):
        calls.append(kwargs)
        yield table(999)
        if fault == "exception":
            raise ConnectionError("synthetic transport failure")
        yield {
            "type": fault,
            "content": {
                "message": "synthetic failure",
                "question": "Уточните норматив",
            },
        }

    orchestrator.scenario_data_service = SimpleNamespace(
        run_scenario_data_pipeline=data
    )
    orchestrator.provision_service.run_provision_pipeline = failing

    async def review(*args):
        if not calls:
            return action("provision", "provision")
        if not data.calls:
            return action("schools", "scenario_data")
        return GoalDecision(
            action="blocked",
            missing=[
                {
                    "missing": "Норматив",
                    "reason": "Нет результата",
                    "question": "Укажите норматив",
                    "owner": "user",
                }
            ],
        )

    orchestrator.goal_manager.review = review
    result = final(
        await run_pipeline(
            orchestrator,
            user_query="Сравни школы и обеспеченность",
            urban_mcp_client=AsyncMock(),
        )
    )
    assert result["status"] == "blocked" and result["missing"]
    assert [r["status"] for r in result["goal"]["requirements"]] == [
        "satisfied",
        "blocked",
    ]
    assert len(calls) == len(data.calls) == 1
    assert not any(a["confirmed"] for a in result["artifacts"] if a["step"] == 1)
    assert sum(a["confirmed"] for a in result["artifacts"]) == 3


async def test_model_control_failure_is_bounded_and_preserves_finished_work(
    orchestrator, monkeypatch
):
    monkeypatch.setenv("ORCHESTRATOR_ANALYSIS_MODE", "goal")
    orchestrator.goal_manager.create = AsyncMock(return_value=contract())
    data = FakePipeline(school_artifacts())
    orchestrator.scenario_data_service = SimpleNamespace(
        run_scenario_data_pipeline=data
    )
    calls = 0

    async def review(*args):
        nonlocal calls
        calls += 1
        if not data.calls:
            return action("schools", "scenario_data")
        return GoalDecision(action="complete", answer="Недоказанное завершение")

    orchestrator.goal_manager.review = review
    result = final(
        await run_pipeline(
            orchestrator,
            user_query="Сравни школы и обеспеченность",
            urban_mcp_client=AsyncMock(),
        )
    )
    assert result["status"] == "blocked"
    assert calls == 7 and len(data.calls) == 1
    assert any(step["agent"] == "provision" for step in result["steps"])
    assert result["goal"]["requirements"][0]["status"] == "satisfied"
    assert result["continue_from"] and result["missing"]


async def test_closing_goal_stream_preserves_unconfirmed_partial_and_releases_budget(
    orchestrator, monkeypatch
):
    import asyncio
    from contextlib import aclosing
    from unittest.mock import Mock

    from src.agents.runtime.budget import current_budget
    from src.agents.services.orchestrator.analysis_support import context_scope

    monkeypatch.setenv("ORCHESTRATOR_ANALYSIS_MODE", "goal")
    orchestrator.goal_manager.create = AsyncMock(return_value=contract())
    orchestrator.goal_manager.review = AsyncMock(
        return_value=action("provision", "provision")
    )
    closed = asyncio.Event()

    async def pipeline(**kwargs):
        try:
            yield table()
            await asyncio.sleep(10)
        finally:
            closed.set()

    orchestrator.provision_service.run_provision_pipeline = pipeline
    async with aclosing(
        orchestrator.run_orchestration_pipeline(
            idu_mcp_client=Mock(),
            effects_mcp_client=Mock(),
            dvd_mcp_client=None,
            normgraph_mcp_client=None,
            token="tok",
            model="m",
            temperature=0,
            user_query="Сравни школы и обеспеченность",
            scenario_id=772,
            persist_history=False,
        )
    ) as stream:
        async for event in stream:
            if event["type"] == "pipeline_started":
                request_id = event["content"]["request_id"]
            if (
                event["type"] == "step_event"
                and event["content"]["event"]["type"] == "table"
            ):
                break
    saved = await orchestrator.state_store.get_analysis_context(
        context_scope("tok", "run:" + request_id)
    )
    assert closed.is_set() and current_budget.get() is None
    assert saved["goal"]["contract"] == contract().model_dump(mode="json")
    assert saved["artifacts"] and not any(a["confirmed"] for a in saved["artifacts"])
