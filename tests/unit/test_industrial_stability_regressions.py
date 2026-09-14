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


async def test_zone_layer_resolves_required_source_and_year_from_catalogue(
    monkeypatch, fake_llm, fake_urban, state_store
):
    from unittest.mock import AsyncMock

    from src.agents.services.scenario_data.scenario_data_service import (
        ScenarioDataService,
    )
    from tests.unit.test_scenario_data_read import read_plan

    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *a, **k: fake_llm,
    )
    source = make_tool(
        "GetScenarioFunctionalZoneSources",
        "projects",
        {"scenario_id": {"type": "integer"}},
    )
    zones = make_tool(
        "GetScenarioFunctionalZones",
        "projects",
        {
            "scenario_id": {"type": "integer"},
            "source": {"type": "string"},
            "year": {"type": "integer"},
        },
    )
    zones.input_schema["required"] = ["scenario_id", "source", "year"]
    fake_llm.json_responses = [
        read_plan(
            zones, {"scenario_id": 17, "source": "survey", "year": 2024}
        ).model_dump_json()
    ]
    mcp = AsyncMock()
    mcp.load_tools.return_value = [source, zones]
    layer = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [30, 60]},
                "properties": {"source": "survey", "year": 2024},
            }
        ],
    }
    mcp.execute_tool.side_effect = [[{"source": "survey", "year": 2024}], layer]
    service = ScenarioDataService("http://llm", None, fake_urban, state_store)
    events = [
        e
        async for e in service.run_scenario_data_pipeline(
            mcp,
            "t",
            "m",
            0,
            "Покажи функциональные зоны сценария 17",
            scenario_id=17,
            persist_history=False,
        )
    ]
    assert [c.args[1] for c in mcp.execute_tool.await_args_list] == [
        source.name,
        zones.name,
    ]
    assert any(
        e["type"] == "feature_collection"
        and e["content"]["feature_collection"] == layer
        for e in events
    )


async def test_goal_creation_does_not_copy_large_assistant_history(fake_llm):
    from src.agents.runtime.budget import RunBudget, budget_scope

    fake_llm.json_responses = [
        json.dumps(
            {
                "objective": "Повторить расчёт",
                "requirements": [
                    {
                        "id": "p",
                        "agent": "provision",
                        "source_ids": [1],
                        "description": "Обеспеченность школами",
                        "required_artifacts": ["table"],
                    }
                ],
            }
        )
    ]
    with budget_scope(RunBudget()):
        result = await GoalManager(fake_llm).create(
            "m",
            "Рассчитай обеспеченность школами.",
            [],
            17,
            [{"role": "assistant", "content": "large saved artifact " * 4000}],
        )
    assert result.requirements[0].agent == "provision"


async def test_model_combined_physical_selection_is_split_without_extra_read(fake_llm):
    fake_llm.json_responses = [
        json.dumps(
            {
                "objective": "Исходные объекты",
                "requirements": [
                    {
                        "id": "physical",
                        "agent": "scenario_data",
                        "source_ids": [1],
                        "description": "Получить физические объекты типов «Жилой дом» и «Парк».",
                        "required_artifacts": ["table", "feature_collection"],
                    }
                ],
            }
        )
    ]
    result = await GoalManager(fake_llm).create(
        "m", "Покажи физические объекты типов «Жилой дом» и «Парк».", [], 17
    )
    assert [(r.entity_kind, r.subject) for r in result.requirements] == [
        ("physical_objects", "Жилой дом"),
        ("physical_objects", "Парк"),
    ]


def test_provision_comparison_uses_real_scoped_cells_and_excludes_other_scenarios():
    context = AnalysisContext()
    for sid, deficit in [(17, 700), (18, 0), (19, 999)]:
        context.add_artifact(
            {
                "type": "table",
                "content": {
                    "name": "provision_summary",
                    "title": "Обеспеченность",
                    "columns": [
                        {"key": "service", "label": "Услуга"},
                        {"key": "deficit", "label": "Дефицит"},
                    ],
                    "rows": [{"service": "Школа", "deficit": deficit}],
                },
            },
            1,
            str(sid),
        )
        context.finish(1, "Расчёт", sid, "completed", "Расчёт", str(sid))
    specs = context.provision_comparisons("Сравни дефициты сценариев 17 и 18")
    result = context.compare(specs)
    assert len(result["content"]["rows"]) == 1
    row = result["content"]["rows"][0]
    assert (row["before"], row["after"], row["delta"]) == ("700", "0", "-700")
    assert context.provision_comparisons("Рассчитай сценарий 18") == []


def test_draft_calculation_domain_cannot_be_confused_with_retrieval_kind():
    from src.agents.services.orchestrator.analysis_goal import GoalDraftRequirement

    r = GoalDraftRequirement.model_validate(
        {
            "id": "p",
            "agent": "provision",
            "subject": "Школа",
            "entity_kind": "services",
            "description": "Рассчитать школы",
            "source_ids": [1],
            "required_artifacts": ["table"],
        }
    )
    assert r.agent == "provision" and r.entity_kind == "other"
