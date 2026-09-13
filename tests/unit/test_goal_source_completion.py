"""Available document evidence must not produce a fictional storage dependency."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.services.orchestrator.analysis_goal import AnalysisGoal, GoalDecision
from src.agents.services.source_evidence import source_event
from tests.unit.test_analytical_orchestrator import final
from tests.unit.test_orchestrator_service_events import (
    FakePipeline,
    orchestrator,
    run_pipeline,
)


@pytest.mark.parametrize("owner", ["service", "user"])
async def test_ready_sources_reject_invented_service_blocker_but_allow_user_criteria(
    orchestrator, monkeypatch, owner
):
    monkeypatch.setenv("ORCHESTRATOR_ANALYSIS_MODE", "goal")
    goal = AnalysisGoal.model_validate(
        {
            "objective": "Сравнить источники",
            "requirements": [
                {
                    "id": name,
                    "description": "Получить исходные записи",
                    "agent": name,
                    "required_artifacts": ["analysis_text", "source_evidence"],
                    "source_quote": "Сравнить источники",
                }
                for name in ["documents", "norms"]
            ],
        }
    )
    orchestrator.goal_manager.create = AsyncMock(return_value=goal)
    for name, attribute, method in [
        ("documents", "dvd_service", "run_document_qa_pipeline"),
        ("norms", "normgraph_service", "run_norms_qa_pipeline"),
    ]:
        pipeline = FakePipeline(
            [
                source_event(name, [{"id": name, "text": "Synthetic source: >= 50 m"}]),
                {"type": "chunk", "content": {"text": ">= 50 m", "done": True}},
            ]
        )
        setattr(orchestrator, attribute, SimpleNamespace(**{method: pipeline}))
    requests = []

    async def review(model, query, agents, context, remaining, budget):
        requests.append(context)
        for requirement in context["goal"]["requirements"]:
            if requirement["status"] == "pending":
                return GoalDecision(action="continue", requirement_id=requirement["id"])
        if len(requests) == 3:
            return GoalDecision(
                action="blocked",
                missing=[
                    {
                        "missing": (
                            "A numeric table"
                            if owner == "service"
                            else "Критерий приоритета"
                        ),
                        "reason": (
                            "Need a table for source comparison"
                            if owner == "service"
                            else "У источников разные приоритеты"
                        ),
                        "question": (
                            "Which table contains the values?"
                            if owner == "service"
                            else "Какой источник считать приоритетным?"
                        ),
                        "owner": owner,
                    }
                ],
            )
        return GoalDecision(
            action="complete",
            answer="Оба источника задают >= 50 м.",
            evidence_ids=[a["id"] for a in context["artifacts"]],
        )

    orchestrator.goal_manager.review = review
    result = final(
        await run_pipeline(
            orchestrator, user_query="Сравни два источника", scenario_id=None
        )
    )
    assert result["status"] == ("completed" if owner == "service" else "blocked")
    assert len(result["steps"]) == 2
