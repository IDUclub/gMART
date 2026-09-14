"""Goal completion is derived from scoped evidence, never a replaceable plan."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.services.orchestrator.analysis_context import AnalysisContext
from src.agents.services.orchestrator.analysis_goal import (
    AnalysisGoal,
    GoalDecision,
    GoalState,
)
from src.agents.services.orchestrator.analysis_support import missing_input
from src.agents.services.service_entities.orchestrator_plan import OrchestratorStep
from tests.unit.test_analytical_orchestrator import LAYER, final, table
from tests.unit.test_orchestrator_service_events import (
    FakePipeline,
    orchestrator,
    run_pipeline,
)


def contract():
    return AnalysisGoal.model_validate(
        {
            "objective": "Школы и обеспеченность",
            "requirements": [
                {
                    "id": "schools",
                    "description": "Получи школы",
                    "source_quote": "школы",
                    "agent": "scenario_data",
                    "scenario_id": 772,
                    "subject": "Школа",
                    "entity_kind": "services",
                    "required_artifacts": ["table", "feature_collection"],
                },
                {
                    "id": "provision",
                    "description": "Рассчитай обеспеченность школами",
                    "source_quote": "обеспеченность",
                    "agent": "provision",
                    "scenario_id": 772,
                    "required_artifacts": ["table"],
                },
            ],
        }
    )


def action(rid, agent, task="Получить результат"):
    return GoalDecision(
        action="continue", requirement_id=rid, agent=agent, task=task, scenario_id=772
    )


def school_artifacts():
    from copy import deepcopy

    t, layer = table(), deepcopy(LAYER)
    t["content"]["title"] = "Школа"
    t["content"]["rows"][0]["service_id"] = 101
    layer["content"]["name"] = "Школа"
    layer["content"]["feature_collection"]["features"][0]["properties"][
        "service_id"
    ] = 101
    return [t, layer]


def test_requirement_cannot_be_satisfied_by_partial_table_or_wrong_scope():
    context = AnalysisContext()
    state = GoalState(context, contract())
    partial, layer = school_artifacts()
    partial["content"]["complete"] = False
    for event in [partial, layer]:
        context.add_artifact(event, 1, "data")
    context.finish(1, "schools", 772, "completed", "", "data")
    state.record(
        OrchestratorStep(
            agent="scenario_data", task="schools", requirement_id="schools"
        ),
        "data",
        "completed",
    )
    assert state.progress()[0]["status"] == "pending"
    assert state.progress()[0]["missing_artifacts"] == ["table"]
    with pytest.raises(ValueError, match="Unfulfilled"):
        state.validate_decision(
            GoalDecision(action="complete", answer="Готово"),
            {"scenario_data", "provision"},
        )
    with pytest.raises(ValueError, match="differs"):
        state.validate_decision(
            action("schools", "provision"), {"scenario_data", "provision"}
        )
    assert GoalState(AnalysisContext(context.dump())).view() == state.view()


def test_supporting_research_cannot_satisfy_calculation_and_scope_is_preserved():
    context = AnalysisContext()
    state = GoalState(context, contract())
    decision = action("provision", "scenario_data", "Получи население")
    decision.support = True
    review = state.validate_decision(decision, {"scenario_data", "provision"})
    context.add_artifact(table(), 1, "population")
    context.finish(1, "population", 772, "completed", "", "population")
    state.record(review.steps[0], "population", "completed")
    assert state.progress()[1]["status"] == "pending"
    decision.scenario_id = 999
    with pytest.raises(ValueError, match="differs"):
        state.validate_decision(decision, {"scenario_data", "provision"})


def test_blocker_does_not_skip_independent_work_and_can_resume_again():
    context = AnalysisContext()
    state = GoalState(context, contract())
    with pytest.raises(ValueError, match="confirmed by a specialist"):
        state.validate_decision(
            GoalDecision(
                action="blocked",
                missing=[missing_input("service", "Нормативы не заданы")],
            ),
            {"scenario_data", "provision"},
        )
    step = OrchestratorStep(
        agent="provision", task="Расчёт", requirement_id="provision"
    )
    for i in range(3):
        state.record(step, str(i), "failed", missing_input("service", "Нет норматива"))
        with pytest.raises(ValueError, match="independent"):
            state.validate_decision(
                GoalDecision(action="blocked", missing=[missing_input("service")]),
                {"scenario_data", "provision"},
            )
        state.resume()
        state.validate_decision(
            action("provision", "provision"), {"scenario_data", "provision"}
        )


def test_wrong_service_and_mismatching_layer_do_not_fulfill_requirement():
    context = AnalysisContext()
    state = GoalState(context, contract())
    t, layer = school_artifacts()
    t["content"]["rows"] = []
    t["content"]["total_rows"] = 0
    for event in [t, layer]:
        context.add_artifact(event, 1, "mismatch")
    context.finish(1, "schools", 772, "completed", "", "mismatch")
    step = OrchestratorStep(
        agent="scenario_data", task="schools", requirement_id="schools"
    )
    state.record(step, "mismatch", "completed")
    assert state.progress()[0]["status"] == "pending"
    assert "matching_table_and_layer" in state.progress()[0]["missing_artifacts"]
    other = table()
    other["content"]["title"] = "Детский сад"
    context.add_artifact(other, 2, "wrong")
    context.finish(2, "schools", 772, "completed", "", "wrong")
    state.record(step, "wrong", "completed")
    assert state.attempts[-1]["evidence_ids"] == []
    malformed = {
        "type": "feature_collection",
        "content": {
            "name": "Школа",
            "feature_collection": {"type": "FeatureCollection"},
        },
    }
    context.add_artifact(malformed, 3, "malformed")
    context.finish(3, "schools", 772, "completed", "", "malformed")
    state.record(step, "malformed", "completed")
    assert state.attempts[-1]["evidence_ids"] == []


def test_missing_normative_has_domain_recovery_without_http_dump():
    blocker = missing_input(
        "service",
        "HTTPException: 400: {'detail': {'code': 'missing_service_normative', 'required_action': 'Задайте норматив для территории 58 и вида услуг 22.'}}",
    )
    assert blocker.owner == "service"
    assert "территории 58" in blocker.question
    assert "HTTPException" not in blocker.reason


async def test_sdk_repairs_old_plan_shape_without_executing_it(fake_llm):
    import json

    from src.agents.runtime.budget import RunBudget, budget_scope
    from src.agents.services.orchestrator.analysis_goal import GoalManager

    fake_llm.json_responses = [
        json.dumps(
            {
                "action": "continue",
                "steps": [{"agent": "scenario_data", "task": "школы"}],
            }
        ),
        json.dumps(
            {"action": "continue", "requirement_id": "schools", "task": "Получи школы"}
        ),
    ]
    budget = RunBudget()
    with budget_scope(budget):
        decision = await GoalManager(fake_llm).review(
            "m", "школы", [], {"goal": contract().model_dump()}, [], {}
        )
    assert decision.requirement_id == "schools"
    assert len(fake_llm.chat_calls) == 2
    assert budget.tool_calls == 0


async def test_goal_source_text_is_bound_by_application_after_sdk_reference_repair(
    fake_llm,
):
    from src.agents.services.orchestrator.analysis_goal import GoalManager

    requirement = contract().requirements[0].model_dump(exclude={"source_quote"})
    fake_llm.json_responses = [
        json.dumps(
            {
                "objective": "Школы",
                "requirements": [{**requirement, "source_ids": [99]}],
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "objective": "Школы",
                "requirements": [{**requirement, "source_ids": [1]}],
            },
            ensure_ascii=False,
        ),
    ]
    goal = await GoalManager(fake_llm).create("m", "Получи школы.", [], 772)
    assert goal.requirements[0].source_quote == "Получи школы."
    assert len(fake_llm.chat_calls) == 2


async def test_budget_stop_preserves_goal_and_completed_artifacts(
    orchestrator, monkeypatch
):
    from src.agents.runtime.budget import BudgetExceeded

    monkeypatch.setenv("ORCHESTRATOR_ANALYSIS_MODE", "goal")
    orchestrator.goal_manager.create = AsyncMock(return_value=contract())
    data = FakePipeline(school_artifacts())
    orchestrator.scenario_data_service = SimpleNamespace(
        run_scenario_data_pipeline=data
    )

    async def review(*args):
        if not data.calls:
            return action("schools", "scenario_data")
        raise BudgetExceeded("tokens")

    orchestrator.goal_manager.review = review
    result = final(
        await run_pipeline(
            orchestrator,
            user_query="Сравни школы и обеспеченность",
            urban_mcp_client=AsyncMock(),
        )
    )
    assert result["missing"][0]["owner"] == "budget"
    assert result["goal"]["requirements"][0]["status"] == "satisfied"
    assert result["goal"]["requirements"][1]["status"] == "pending"
    assert "Школа" in result["answer"]


async def test_goal_skips_plan_and_keeps_independent_artifacts_after_blocker(
    orchestrator, monkeypatch
):
    monkeypatch.setenv("ORCHESTRATOR_ANALYSIS_MODE", "goal")
    orchestrator.goal_manager.create = AsyncMock(return_value=contract())
    orchestrator.plan_builder.build_plan = AsyncMock(
        side_effect=AssertionError("Detailed plan must not run")
    )
    data = FakePipeline(school_artifacts())
    provision = FakePipeline(
        [
            {
                "type": "error",
                "content": {"message": "Нет норматива для территории 58 и типа 22"},
            }
        ]
    )
    orchestrator.scenario_data_service = SimpleNamespace(
        run_scenario_data_pipeline=data
    )
    orchestrator.provision_service.run_provision_pipeline = provision

    async def review(model, query, agents, context, remaining, budget):
        assert not remaining
        if not provision.calls:
            return action("provision", "provision")
        if not data.calls:
            return action("schools", "scenario_data")
        return GoalDecision(
            action="blocked",
            answer="Доступные результаты сохранены.",
            evidence_ids=[a["id"] for a in context["artifacts"] if a["confirmed"]],
            missing=[
                {
                    "missing": "Норматив",
                    "reason": "Расчёт недоступен",
                    "question": "Укажите применимый норматив",
                    "owner": "service",
                }
            ],
        )

    orchestrator.goal_manager.review = review
    events = await run_pipeline(
        orchestrator,
        user_query="Сравни школы и обеспеченность",
        urban_mcp_client=AsyncMock(),
    )
    result = final(events)
    assert result["status"] == "blocked"
    assert [r["status"] for r in result["goal"]["requirements"]] == [
        "satisfied",
        "blocked",
    ]
    assert "территории 58" in result["answer"]
    assert len([a for a in result["artifacts"] if a["confirmed"]]) == 3
    assert len(data.calls) == len(provision.calls) == 1
    assert "«Школа»" in data.calls[0]["user_query"]
    orchestrator.plan_builder.build_plan.assert_not_called()
    assert events == await run_pipeline(
        orchestrator, request_id=result["continue_from"]
    )

    # Continue the same goal with a now functioning calculation. Do not fetch
    # schools again or ask the model to reinterpret/erase the original goal.
    orchestrator.goal_manager.create.reset_mock()
    repaired = FakePipeline([table(90)])
    orchestrator.provision_service.run_provision_pipeline = repaired

    async def resume(model, query, agents, context, remaining, budget):
        if not repaired.calls:
            return action("provision", "provision")
        return GoalDecision(
            action="complete",
            answer="Расчёт получен",
            evidence_ids=[a["id"] for a in context["artifacts"] if a["confirmed"]],
        )

    orchestrator.goal_manager.review = resume
    resumed = final(
        await run_pipeline(
            orchestrator, user_query="Продолжи", continue_from=result["continue_from"]
        )
    )
    assert resumed["status"] == "completed"
    assert len(data.calls) == 1
    orchestrator.goal_manager.create.assert_not_called()


async def test_false_completion_and_duplicate_are_repaired_without_replaying_tools(
    orchestrator, monkeypatch
):
    monkeypatch.setenv("ORCHESTRATOR_ANALYSIS_MODE", "goal")
    goal = contract().model_copy(update={"requirements": contract().requirements[:1]})
    orchestrator.goal_manager.create = AsyncMock(return_value=goal)
    data = FakePipeline(school_artifacts())
    orchestrator.scenario_data_service = SimpleNamespace(
        run_scenario_data_pipeline=data
    )
    calls = 0

    async def review(model, query, agents, context, remaining, budget):
        nonlocal calls
        calls += 1
        if calls == 1:
            return GoalDecision(action="complete", answer="Готово")
        if calls in {2, 3}:
            return action("schools", "scenario_data")
        return GoalDecision(
            action="complete",
            answer="Подтверждено",
            evidence_ids=[a["id"] for a in context["artifacts"]],
        )

    orchestrator.goal_manager.review = review
    result = final(
        await run_pipeline(
            orchestrator, user_query="Сравни школы", urban_mcp_client=AsyncMock()
        )
    )
    assert result["status"] == "completed"
    assert calls == 4
    assert len(data.calls) == 1


async def test_unsupported_final_claim_is_repaired_without_replaying_calculation(
    orchestrator, monkeypatch
):
    monkeypatch.setenv("ORCHESTRATOR_ANALYSIS_MODE", "goal")
    goal = contract().model_copy(update={"requirements": contract().requirements[1:]})
    orchestrator.goal_manager.create = AsyncMock(return_value=goal)
    provision = FakePipeline([table(80)])
    orchestrator.provision_service.run_provision_pipeline = provision
    answers = iter(["Обеспеченность 100%.", "Обеспеченность 80%."])
    views = []

    async def review(model, query, agents, context, remaining, budget):
        if not provision.calls:
            return action("provision", "provision")
        views.append(context)
        return GoalDecision(
            action="complete",
            answer=next(answers),
            evidence_ids=[a["id"] for a in context["artifacts"] if a["confirmed"]],
        )

    orchestrator.goal_manager.review = review
    orchestrator.goal_manager.validate_answer = AsyncMock(
        side_effect=[ValueError("Обеспеченность 100%: таблица подтверждает 80%."), None]
    )
    result = final(await run_pipeline(orchestrator, user_query="Сравни обеспеченность"))
    assert result["status"] == "completed"
    assert result["answer"] == "Обеспеченность 80%."
    assert "80%" in views[-1]["review_validation_error"]
    assert len(provision.calls) == 1


async def test_grounding_failure_preserves_results_without_publishing_false_answer(
    orchestrator, monkeypatch
):
    monkeypatch.setenv("ORCHESTRATOR_ANALYSIS_MODE", "goal")
    goal = contract().model_copy(update={"requirements": contract().requirements[1:]})
    orchestrator.goal_manager.create = AsyncMock(return_value=goal)
    provision = FakePipeline([table(80)])
    orchestrator.provision_service.run_provision_pipeline = provision

    async def review(model, query, agents, context, remaining, budget):
        if not provision.calls:
            return action("provision", "provision")
        return GoalDecision(
            action="complete",
            answer="Обеспеченность 100%.",
            evidence_ids=[a["id"] for a in context["artifacts"] if a["confirmed"]],
        )

    orchestrator.goal_manager.review = review
    orchestrator.goal_manager.validate_answer = AsyncMock(
        side_effect=ValueError("Неверные 100%")
    )
    result = final(await run_pipeline(orchestrator, user_query="Сравни обеспеченность"))
    assert result["status"] == "blocked"
    assert "100%" not in result["answer"]
    assert result["continue_from"]
    assert result["missing"][0]["owner"] == "service"
    assert any(a["confirmed"] and a["kind"] == "table" for a in result["artifacts"])
    assert len(provision.calls) == 1
    assert orchestrator.goal_manager.validate_answer.await_count == 3


async def test_goal_creation_failure_keeps_prior_artifacts_and_pending_request(
    orchestrator, monkeypatch
):
    from src.agents.services.orchestrator.analysis_support import context_scope

    monkeypatch.setenv("ORCHESTRATOR_ANALYSIS_MODE", "goal")
    context = AnalysisContext()
    context.query = "Исходный расчёт для 8000 жителей"
    aid = context.add_artifact(table(80), 1, "prior-calc")
    context.finish(1, "Расчёт", 772, "completed", "Сохранено", "prior-calc")
    await orchestrator.state_store.save_analysis_context(
        context_scope("tok", "existing-chat"), context.dump()
    )
    orchestrator.goal_manager.create = AsyncMock(side_effect=ValueError("Invalid goal"))
    result = final(
        await run_pipeline(
            orchestrator,
            chat_id="existing-chat",
            user_query="Теперь сопоставь источники",
        )
    )
    assert result["status"] == "blocked"
    assert any(a["id"] == aid and a["confirmed"] for a in result["artifacts"])
    saved = await orchestrator.state_store.get_analysis_context(
        context_scope("tok", "run:" + result["continue_from"])
    )
    assert "8000" in saved["query"]
    assert "сопоставь источники" in saved["query"]
    assert saved["goal"] is None
