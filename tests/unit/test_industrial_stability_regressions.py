"""Failures reduced from the industrial dialogue traces, without live inference."""

import json

import pytest

from src.agents.services.orchestrator.analysis_context import AnalysisContext
from src.agents.services.orchestrator.analysis_goal import (
    AnalysisGoal,
    GoalDecision,
    GoalManager,
    GoalState,
)
from src.agents.services.scenario_data.scenario_data_read import scoped_tools
from tests.unit.test_scenario_data_read import make_tool


def test_functional_zone_request_keeps_geometry_source_in_catalogue():
    zones = make_tool("GetScenarioFunctionalZones", "projects")
    card = make_tool("GetProjectById", "projects")
    assert scoped_tools(
        [card, zones], "Получить слой функциональных зон проекта 91001"
    ) == [zones]


@pytest.mark.parametrize("agent", ["documents", "norms", "compliance"])
def test_delegation_preserves_document_scope_when_controller_shortens_task(agent):
    goal = AnalysisGoal.model_validate(
        {
            "objective": "Проверить только указанную норму",
            "requirements": [
                {
                    "id": "norm",
                    "agent": agent,
                    "scenario_id": 91001,
                    "description": "Получить пункт 1.1 документа LOCAL SDK TEST, версия 2026",
                    "source_quote": "Используй только LOCAL SDK TEST, версия 2026, пункт 1.1",
                    "required_artifacts": ["analysis_text"],
                }
            ],
        }
    )
    step = (
        GoalState(AnalysisContext(), goal)
        .validate_decision(
            GoalDecision(
                action="continue", requirement_id="norm", task="Проверь норму"
            ),
            {agent},
        )
        .steps[0]
    )
    assert "LOCAL SDK TEST" in step.task
    assert "2026" in step.task
    assert "1.1" in step.task


def test_multi_variant_goal_can_hold_all_atomic_results():
    goal = AnalysisGoal.model_validate(
        {
            "objective": "Три варианта с независимыми результатами",
            "requirements": [
                {
                    "id": f"r{i}",
                    "agent": "scenario_data",
                    "scenario_id": i + 1,
                    "description": "Получить результат",
                    "source_quote": "Все варианты",
                    "required_artifacts": ["table"],
                }
                for i in range(15)
            ],
        }
    )
    assert len(goal.requirements) == 15


async def test_goal_rejects_combined_type_and_repairs_to_atomic_selections(fake_llm):
    draft = {
        "objective": "Получить услуги",
        "requirements": [
            {
                "id": "services",
                "agent": "scenario_data",
                "scenario_id": 17,
                "description": "Получить школы и детские сады",
                "source_ids": [1],
                "subject": "Школа, Детский сад",
                "entity_kind": "services",
                "required_artifacts": ["table", "feature_collection"],
            }
        ],
    }
    repaired = {
        "objective": draft["objective"],
        "requirements": [
            {**draft["requirements"][0], "id": f"r{i}", "subject": subject}
            for i, subject in enumerate(["Школа", "Детский сад"])
        ],
    }
    fake_llm.json_responses = [json.dumps(draft), json.dumps(repaired)]
    goal = await GoalManager(fake_llm).create(
        "m", "Получи школы и детские сады.", [], 17
    )
    assert [r.subject for r in goal.requirements] == ["Школа", "Детский сад"]


def test_synthesis_view_includes_source_proof_without_manual_inspection():
    context = AnalysisContext()
    aid = context.add_artifact(
        {
            "type": "source_evidence",
            "content": {
                "source": "dvd",
                "document_name": "Example",
                "version": "2026",
                "text": "Minimum 50 m",
            },
        },
        1,
        "r",
    )
    context.finish(1, "Read source", None, "completed", "Read", "r")
    view = context.view()
    assert any(p.get("artifact_id") == aid for p in view["selected_evidence"])


def test_calculation_evidence_retains_scenario_and_rows_in_crowded_context():
    context = AnalysisContext()
    aid = context.add_artifact(
        {
            "type": "table",
            "content": {
                "name": "provision_summary",
                "title": "Обеспеченность",
                "columns": [{"key": "deficit", "label": "Дефицит"}],
                "rows": [{"deficit": 75}],
            },
        },
        1,
        "calculation",
    )
    context.finish(1, "Расчёт", 17, "completed", "Расчёт готов", "calculation")
    for i in range(35):
        context.add_artifact(
            {
                "type": "analysis_text",
                "content": {"text": "Ready", "title": f"Other {i}"},
            },
            2,
            "later",
        )
    context.finish(2, "Другая работа", 18, "completed", "Готово", "later")
    preview = next(
        p for p in context.view()["selected_evidence"] if p.get("artifact_id") == aid
    )
    assert preview["scenario_id"] == 17
    assert preview["rows"][0]["deficit"] == 75


async def test_followup_can_cite_prior_service_and_layer_conditions(fake_llm):
    fake_llm.json_responses = [
        json.dumps(
            {
                "objective": "Повторный расчёт",
                "requirements": [
                    {
                        "id": "p",
                        "agent": "provision",
                        "description": "Рассчитать обеспеченность школами",
                        "source_ids": [1, 2],
                        "required_artifacts": ["table"],
                    }
                ],
            }
        )
    ]
    goal = await GoalManager(fake_llm).create(
        "m",
        "Теперь сценарий 18.",
        [],
        18,
        [
            {
                "role": "user",
                "content": "Рассчитай обеспеченность школами и верни расчётные слои.",
            }
        ],
    )
    assert "школами" in goal.requirements[0].source_quote
    assert "feature_collection" in goal.requirements[0].required_artifacts
