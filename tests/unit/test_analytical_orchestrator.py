"""Acceptance contracts for analysis, evidence delivery and bounded continuation."""

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.runtime.budget import (
    BudgetExceeded,
    BudgetLimits,
    RunBudget,
    budget_scope,
    current_budget,
)
from src.agents.schema.orchestrator_response import OrchestratorResponse
from src.agents.services.orchestrator.analysis_context import AnalysisContext
from src.agents.services.orchestrator.analysis_support import context_scope
from src.agents.services.service_entities.orchestrator_plan import (
    AnalysisReview,
    MetricComparison,
    OrchestratorPlan,
)
from tests.unit.test_orchestrator_service_events import (
    FakePipeline,
    orchestrator,
    run_pipeline,
)


def table(value=80):
    return {
        "type": "table",
        "content": {
            "name": "provision",
            "title": "Обеспеченность",
            "columns": [{"key": "percent", "label": "Обеспеченность, %"}],
            "rows": [{"percent": value}],
            "complete": True,
            "total_rows": 1,
        },
    }


LAYER = {
    "type": "feature_collection",
    "content": {
        "name": "Школы",
        "feature_collection": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [30, 60]},
                    "properties": {"capacity": 100},
                }
            ],
        },
    },
}


def plan(*steps):
    return OrchestratorPlan.model_validate(
        {
            "mode": "execute",
            "analytical": True,
            "steps": list(steps)
            or [{"agent": "provision", "task": "Проверь обеспеченность"}],
        }
    )


def final(events):
    event = next(e for e in reversed(events) if e["type"] == "orchestrator_final")
    # Exercise the actual REST serialization contract, not just service dicts.
    return OrchestratorResponse.model_validate(event).model_dump()["content"]


async def test_scenario_data_receives_task_without_control_context(orchestrator):
    task = "Получи услуги типа школа в сценарии 772 и их слой."
    first = FakePipeline([table()])
    second = FakePipeline([table(20)])
    orchestrator.provision_service.run_provision_pipeline = first
    orchestrator.scenario_data_service = SimpleNamespace(
        run_scenario_data_pipeline=second
    )
    orchestrator.plan_builder.build_plan = AsyncMock(return_value=plan())

    async def review(model, query, agents, context, remaining, budget):
        if not second.calls:
            return AnalysisReview(
                action="continue", steps=[{"agent": "scenario_data", "task": task}]
            )
        return AnalysisReview(
            action="complete",
            answer="Готово",
            evidence_ids=[a["id"] for a in context["artifacts"]],
        )

    orchestrator.plan_builder.review = review
    events = await run_pipeline(orchestrator, urban_mcp_client=AsyncMock())
    assert final(events)["status"] == "completed"
    assert second.calls[0]["user_query"] == task


async def test_native_dates_are_json_safe_in_evidence_and_terminal_replay(orchestrator):
    event = table()
    event["content"]["rows"][0]["created_at"] = datetime(
        2026, 9, 13, tzinfo=timezone.utc
    )
    orchestrator.provision_service.run_provision_pipeline = FakePipeline([event])
    orchestrator.plan_builder.build_plan = AsyncMock(return_value=plan())

    async def review(model, query, agents, context, remaining, budget):
        json.dumps(context)
        return AnalysisReview(
            action="complete",
            answer="Готово",
            evidence_ids=[a["id"] for a in context["artifacts"]],
        )

    orchestrator.plan_builder.review = review
    events = await run_pipeline(orchestrator)
    result = final(events)
    assert result["status"] == "completed"
    assert json.loads(json.dumps(events)) == await run_pipeline(
        orchestrator, request_id=result["continue_from"]
    )


async def test_scenario_clarification_is_not_confirmed_evidence(orchestrator):
    question = "Считать услуги или здания?"
    pipeline = FakePipeline(
        [{"type": "clarification_required", "content": {"text": question}}]
    )
    orchestrator.scenario_data_service = SimpleNamespace(
        run_scenario_data_pipeline=pipeline
    )
    orchestrator.plan_builder.build_plan = AsyncMock(
        return_value=plan({"agent": "scenario_data", "task": "Посчитай объекты"})
    )
    orchestrator.plan_builder.review = AsyncMock(
        return_value=AnalysisReview(action="complete", answer="Готово", evidence_ids=[])
    )
    result = final(await run_pipeline(orchestrator, urban_mcp_client=AsyncMock()))
    assert result["status"] == "blocked"
    assert result["steps"][0]["status"] == "needs_clarification"
    assert question in result["answer"]
    assert result["artifacts"] == []
    orchestrator.plan_builder.review.assert_not_called()


async def test_review_cannot_skip_calculation_or_repeat_completed_data(orchestrator):
    task = "Получи школы"
    data = FakePipeline([table()])
    calculation = FakePipeline(
        [{"type": "error", "content": {"message": "Нет норматива"}}]
    )
    orchestrator.scenario_data_service = SimpleNamespace(
        run_scenario_data_pipeline=data
    )
    orchestrator.provision_service.run_provision_pipeline = calculation
    orchestrator.plan_builder.build_plan = AsyncMock(
        return_value=plan(
            {"agent": "scenario_data", "task": task},
            {"agent": "provision", "task": "Рассчитай обеспеченность"},
        )
    )
    attempts = []

    async def review(model, query, agents, context, remaining, budget):
        attempts.append(context)
        aid = context["artifacts"][0]["id"]
        if len(attempts) == 1:
            return AnalysisReview(
                action="complete", answer="Расчёт невозможен", evidence_ids=[aid]
            )
        assert context["review_validation_error"]
        if len(attempts) == 2:
            return AnalysisReview(
                action="continue",
                steps=[{"agent": "scenario_data", "task": task, "evidence_ids": [aid]}],
            )
        return AnalysisReview(
            action="continue",
            steps=[{"agent": "provision", "task": "Рассчитай обеспеченность"}],
        )

    orchestrator.plan_builder.review = review
    result = final(await run_pipeline(orchestrator, urban_mcp_client=AsyncMock()))
    assert result["status"] == "blocked"
    assert len(data.calls) == len(calculation.calls) == 1
    assert len(attempts) == 3


@pytest.mark.parametrize("repair", [True, False])
async def test_invalid_comparison_is_repaired_without_replaying_specialist(
    orchestrator, repair
):
    calls = []

    async def pipeline(**kwargs):
        calls.append(kwargs)
        yield {
            "type": "chunk",
            "content": {"text": "Требование: 50 м [1]", "iteration": 1, "done": True},
        }

    orchestrator.provision_service.run_provision_pipeline = pipeline
    orchestrator.plan_builder.build_plan = AsyncMock(return_value=plan())
    reviews = []

    async def review(model, query, agents, context, remaining, budget):
        reviews.append(context)
        aid = context["artifacts"][0]["id"]
        result = {
            "action": "complete",
            "answer": "В обоих источниках 50 м.",
            "evidence_ids": [aid],
        }
        if len(reviews) > 1:
            assert "table" in context["review_validation_error"]
        if len(reviews) == 1 or not repair:
            ref = {"artifact_id": aid, "row": 0, "column": "value"}
            result["comparisons"] = [
                {"name": "Расстояние", "unit": "м", "before": ref, "after": ref}
            ]
        return AnalysisReview.model_validate(result)

    orchestrator.plan_builder.review = review
    events = await run_pipeline(orchestrator)
    assert final(events)["status"] == ("completed" if repair else "blocked")
    assert len(calls) == 1 and len(reviews) == (2 if repair else 3)
    assert final(events)["artifacts"][0]["confirmed"]


@pytest.mark.asyncio
async def test_compare_three_scenarios_replans_and_returns_full_artifacts(orchestrator):
    calls = []

    async def pipeline(**kwargs):
        calls.append(kwargs)
        yield table({10: 80, 20: 90, 30: 85}[kwargs["scenario_id"]])
        yield LAYER

    orchestrator.provision_service.run_provision_pipeline = pipeline
    orchestrator.plan_builder.build_plan = AsyncMock(
        return_value=plan({"agent": "provision", "task": "База", "scenario_id": 10})
    )

    async def review(model, query, agents, context, remaining, budget):
        if len(calls) < 3:
            return AnalysisReview.model_validate(
                {
                    "action": "continue",
                    "steps": [
                        {
                            "agent": "provision",
                            "task": f"Вариант {len(calls)}",
                            "scenario_id": 20 if len(calls) == 1 else 30,
                        }
                    ],
                }
            )
        ids = [a["id"] for a in context["artifacts"] if a["kind"] == "table"]
        return AnalysisReview.model_validate(
            {
                "action": "complete",
                "answer": "Варианты повышают обеспеченность; выбор зависит от остальных критериев.",
                "evidence_ids": ids,
                "comparisons": [
                    {
                        "name": "Обеспеченность",
                        "unit": "%",
                        "before": {
                            "artifact_id": ids[0],
                            "row": 0,
                            "column": "percent",
                        },
                        "after": {"artifact_id": id, "row": 0, "column": "percent"},
                    }
                    for id in ids[1:]
                ],
            }
        )

    orchestrator.plan_builder.review = review
    events = await run_pipeline(orchestrator)
    result = final(events)
    assert result["status"] == "completed"
    assert [c["scenario_id"] for c in calls] == [10, 20, 30]
    assert len(result["artifacts"]) == 7
    assert all(a["confirmed"] for a in result["artifacts"])
    comparison = [
        e["content"]["event"]["content"]
        for e in events
        if e["type"] == "step_event"
        and e["content"]["event"].get("content", {}).get("name")
        == "analysis_comparison"
    ][0]
    assert [r["delta"] for r in comparison["rows"]] == ["10", "5"]
    parts = orchestrator.add_complex_message.await_args.args[3]
    assert len([p for p in parts if p.kind == "table"]) == 4
    assert (
        len(
            [
                p
                for p in parts
                if p.kind == "data" and p.payload["event_type"] == "feature_collection"
            ]
        )
        == 3
    )


@pytest.mark.asyncio
async def test_changed_assumptions_reuse_confirmed_context_without_rerunning_base(
    orchestrator,
):
    first = FakePipeline([table(), LAYER])
    orchestrator.provision_service.run_provision_pipeline = first
    orchestrator.plan_builder.build_plan = AsyncMock(return_value=plan())

    async def done(model, query, agents, context, remaining, budget):
        return AnalysisReview(
            action="complete",
            answer="Базовый результат сохранён.",
            evidence_ids=[a["id"] for a in context["artifacts"]],
        )

    orchestrator.plan_builder.review = done
    original = final(await run_pipeline(orchestrator, persist_history=False))

    async def revise(model, query, agents, context, remaining, budget):
        assert context["completed"]
        if len(first.calls) == 1:
            return AnalysisReview.model_validate(
                {
                    "action": "continue",
                    "steps": [
                        {
                            "agent": "provision",
                            "task": "Пересчитать при населении +20%; геометрию школ использовать прежнюю",
                            "evidence_ids": original["evidence_ids"],
                        }
                    ],
                }
            )
        return await done(model, query, agents, context, remaining, budget)

    orchestrator.plan_builder.review = revise
    events = await run_pipeline(
        orchestrator,
        user_query="Теперь население +20%",
        continue_from=original["continue_from"],
        persist_history=False,
    )
    assert final(events)["status"] == "completed"
    assert len(first.calls) == 2
    assert "+20%" in first.calls[1]["user_query"]
    assert original["evidence_ids"][0] in first.calls[1]["user_query"]
    assert any(e["type"] == "step_event" and e["content"]["step"] == 0 for e in events)


@pytest.mark.asyncio
async def test_unexpected_result_separates_hypothesis_and_requests_missing_inputs(
    orchestrator,
):
    orchestrator.plan_builder.build_plan = AsyncMock(return_value=plan())
    orchestrator.provision_service.run_provision_pipeline = FakePipeline([table(60)])
    orchestrator.plan_builder.review = AsyncMock(
        return_value=AnalysisReview.model_validate(
            {
                "action": "blocked",
                "answer": "Показатель снизился, причина пока не установлена.",
                "hypotheses": ["Возможно, увеличилось расчётное население."],
                "missing": [
                    {
                        "missing": "Население и мощность школ до и после",
                        "reason": "Без них нельзя проверить причину изменения.",
                        "question": "Укажите эти значения и методику расчёта.",
                        "example": "Было N жителей и M мест; стало…",
                        "owner": "user",
                    }
                ],
            }
        )
    )
    result = final(await run_pipeline(orchestrator))
    assert result["status"] == "blocked"
    assert "Неподтверждённые гипотезы" in result["answer"]
    assert "Пример полезного уточнения" in result["answer"]
    assert result["missing"][0]["owner"] == "user"
    assert result["artifacts"][0]["confirmed"]


@pytest.mark.asyncio
async def test_time_limit_retains_finished_artifacts_and_replay_is_free(orchestrator):
    orchestrator.plan_builder.build_plan = AsyncMock(return_value=plan())
    orchestrator.provision_service.run_provision_pipeline = FakePipeline(
        [table(), LAYER]
    )

    async def slow(*args):
        await asyncio.sleep(1)

    orchestrator.plan_builder.review = slow
    events = await run_pipeline(
        orchestrator, budget_seconds=0.05, persist_history=False
    )
    result = final(events)
    assert result["status"] == "blocked" and result["missing"][0]["owner"] == "budget"
    assert len(result["artifacts"]) == 2
    assert all(a["confirmed"] for a in result["artifacts"])
    orchestrator.resolve_model = AsyncMock(
        side_effect=AssertionError("replay must not resolve model")
    )
    replay = await run_pipeline(
        orchestrator, request_id=result["continue_from"], persist_history=False
    )
    assert final(replay) == result
    assert current_budget.get() is None


@pytest.mark.asyncio
async def test_failed_draft_is_never_evidence(orchestrator):
    orchestrator.plan_builder.build_plan = AsyncMock(return_value=plan())
    orchestrator.provision_service.run_provision_pipeline = FakePipeline(
        [table(), {"type": "error", "content": {"message": "down"}}]
    )
    orchestrator.plan_builder.review = AsyncMock()
    result = final(await run_pipeline(orchestrator))
    assert result["status"] == "blocked"
    assert not result["artifacts"][0]["confirmed"]
    orchestrator.plan_builder.review.assert_not_awaited()
    assert not any(
        p.kind == "table" for p in orchestrator.add_complex_message.await_args.args[3]
    )


@pytest.mark.asyncio
async def test_empty_normative_corpus_is_not_success(orchestrator):
    orchestrator.plan_builder.build_plan = AsyncMock(
        return_value=plan({"agent": "compliance", "task": "Проверить нормативы"})
    )
    orchestrator.restriction_service.run_compliance_pipeline = FakePipeline(
        [{"type": "compliance_summary", "content": {"total_norms": 0}}]
    )
    result = final(await run_pipeline(orchestrator))
    assert result["status"] == "blocked"
    assert result["missing"][0]["missing"] == "Применимые нормативы"


@pytest.mark.asyncio
async def test_repeated_step_stops_without_duplicate_tool_work(orchestrator):
    p = plan()
    orchestrator.plan_builder.build_plan = AsyncMock(return_value=p)
    pipeline = FakePipeline([table()])
    orchestrator.provision_service.run_provision_pipeline = pipeline
    orchestrator.plan_builder.review = AsyncMock(
        return_value=AnalysisReview(action="continue", steps=p.steps)
    )
    result = final(await run_pipeline(orchestrator))
    assert result["status"] == "blocked"
    assert len(pipeline.calls) == 1
    assert "Повторение" in result["answer"]


def test_context_is_bounded_but_full_artifact_survives_and_unknown_refs_fail():
    context = AnalysisContext()
    event = table()
    event["content"]["rows"] = [
        {"percent": i, "text": "данные" * 100} for i in range(200)
    ]
    event["content"]["total_rows"] = 300
    event["content"]["complete"] = False
    aid = context.add_artifact(event, 1, "r")
    with pytest.raises(ValueError):
        context.get(aid)
    context.finish(1, "test", 1, "completed", "summary", "r")
    assert len(json.dumps(context.view(), ensure_ascii=False).encode()) <= 9000
    assert len(context.get(aid)["content"]["rows"]) == 200
    assert context.slice(aid, 150, 2)["rows"][0]["percent"] == 150
    assert context.slice(aid, 0, 2)["source_complete"] is False
    with pytest.raises(ValueError):
        context.get("invented")
    assert context_scope("user1", "chat") != context_scope("user2", "chat")


def test_budget_reserves_answer_counts_unknown_calls_and_never_increases_limit():
    budget = RunBudget(replace(BudgetLimits(), total_tokens=30000, model_calls=3))
    r = budget.reserve([{"content": "q"}], None, 1000)
    r.settle({"total_tokens": 500})
    assert budget.tokens == 500
    r.settle({"total_tokens": 1})
    assert budget.tokens == 500
    budget.reserve([], None, 1000).settle()
    assert budget.estimated_calls == 1
    with pytest.raises(BudgetExceeded):
        budget.reserve([], None)
    budget.finalizing = True
    budget.reserve([], None, 1000).settle({"total_tokens": 100})
    assert budget.model_calls == 3
    with pytest.raises(BudgetExceeded):
        budget.reserve([], None)


@pytest.mark.asyncio
async def test_continuation_cannot_read_another_subject_context(orchestrator):
    orchestrator.plan_builder.build_plan = AsyncMock(return_value=plan())
    orchestrator.provision_service.run_provision_pipeline = FakePipeline(
        [table(), {"type": "error", "content": {}}]
    )
    result = final(await run_pipeline(orchestrator, persist_history=False))
    events = await run_pipeline(
        orchestrator,
        token="different-user",
        continue_from=result["continue_from"],
        persist_history=False,
    )
    assert any(e["type"] == "clarification" for e in events)
    assert not any(e["type"] == "step_started" for e in events)


@pytest.mark.asyncio
async def test_population_change_is_computed_from_evidence_before_dispatch(
    orchestrator,
):
    from src.agents.services.service_entities.orchestrator_plan import (
        PopulationAdjustment,
    )

    ctx = AnalysisContext()
    event = table(10000)
    aid = ctx.add_artifact(event, 1, "base")
    ctx.finish(1, "Население", 772, "completed", "Население 10000", "base")
    scope = context_scope("tok", "run:prior")
    await orchestrator.state_store.save_analysis_context(scope, ctx.dump())
    adjustment = PopulationAdjustment.model_validate(
        {
            "base": {"artifact_id": aid, "row": 0, "column": "percent"},
            "multiplier": "1.2",
        }
    )
    pipeline = FakePipeline([table(70)])
    orchestrator.provision_service.run_provision_pipeline = pipeline
    orchestrator.plan_builder.review = AsyncMock(
        side_effect=[
            AnalysisReview(
                action="continue",
                steps=[
                    plan()
                    .steps[0]
                    .model_copy(update={"population_adjustment": adjustment})
                ],
            ),
            AnalysisReview(
                action="complete", answer="Результат пересчёта", evidence_ids=[aid]
            ),
        ]
    )
    result = final(
        await run_pipeline(
            orchestrator,
            continue_from="prior",
            user_query="Население +20%",
            persist_history=False,
        )
    )
    assert result["status"] == "completed"
    assert "12000 человек" in pipeline.calls[0]["user_query"]
    assert len(pipeline.calls) == 1


@pytest.mark.asyncio
async def test_concurrent_context_saves_do_not_drop_other_runs(state_store):
    first, second = AnalysisContext(), AnalysisContext()
    first.add_artifact(table(10), 1, "a")
    second.add_artifact(table(20), 1, "b")
    await asyncio.gather(
        state_store.save_analysis_context("same", first.dump()),
        state_store.save_analysis_context("same", second.dump()),
    )
    stored = await state_store.get_analysis_context("same")
    assert {a["request_id"] for a in stored["artifacts"]} == {"a", "b"}


def test_incompatible_units_cannot_be_compared():
    ctx = AnalysisContext()
    ids = []
    for n, unit in enumerate(("%", "человек")):
        event = table(10)
        event["content"]["rows"][0]["unit"] = unit
        ids.append(ctx.add_artifact(event, n, str(n)))
        ctx.finish(n, "task", 1, "completed", "result", str(n))
    spec = MetricComparison.model_validate(
        {
            "name": "metric",
            "unit": "%",
            "before": {"artifact_id": ids[0], "row": 0, "column": "percent"},
            "after": {"artifact_id": ids[1], "row": 0, "column": "percent"},
        }
    )
    with pytest.raises(ValueError, match="dimension"):
        ctx.compare([spec])


@pytest.mark.asyncio
async def test_close_mid_step_preserves_artifact_as_unconfirmed(orchestrator):
    from contextlib import aclosing
    from unittest.mock import Mock

    orchestrator.plan_builder.build_plan = AsyncMock(return_value=plan())
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
            user_query="test",
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
    assert closed.is_set()
    saved = await orchestrator.state_store.get_analysis_context(
        context_scope("tok", "run:" + request_id)
    )
    assert not saved["artifacts"][0]["confirmed"]
    assert current_budget.get() is None
