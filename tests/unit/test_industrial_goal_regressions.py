"""Defects observed by real industrial dialogues, reduced to public service seams."""

from src.agents.services.orchestrator.analysis_context import AnalysisContext
from src.agents.services.orchestrator.analysis_goal import (
    AnalysisGoal,
    GoalDecision,
    GoalState,
)


def test_spatial_distance_survives_controller_rewording():
    goal = AnalysisGoal.model_validate(
        {
            "objective": "Проверить проект",
            "requirements": [
                {
                    "id": "zone",
                    "agent": "restriction",
                    "scenario_id": 91001,
                    "description": "Создать 50-метровую зону вокруг открытой автомобильной стоянки",
                    "source_quote": "Верни 50-метровую зону вокруг стоянки",
                    "required_artifacts": ["feature_collection"],
                }
            ],
        }
    )
    state = GoalState(AnalysisContext(), goal)
    decision = GoalDecision(
        action="continue",
        requirement_id="zone",
        task="Получить feature_collection для стоянки",
    )
    assert "50" in state.validate_decision(decision, {"restriction"}).steps[0].task
    support = decision.model_copy(update={"support": True, "agent": "scenario_data"})
    assert (
        state.validate_decision(support, {"restriction", "scenario_data"}).steps[0].task
        == decision.task
    )


def test_controller_recovery_finishes_unattempted_work_before_repeating():
    goal = AnalysisGoal.model_validate(
        {
            "objective": "Два независимых расчёта",
            "requirements": [
                {
                    "id": name,
                    "agent": "provision",
                    "scenario_id": sid,
                    "description": "Рассчитать обеспеченность школами",
                    "source_quote": "Нужны оба сценария",
                    "required_artifacts": ["table"],
                }
                for name, sid in (("before", 91003), ("after", 91004))
            ],
        }
    )
    state = GoalState(AnalysisContext(), goal)
    step = state.validate_decision(
        GoalDecision(action="continue", requirement_id="before"), {"provision"}
    ).steps[0]
    state.record(step, "r1", "completed")  # The specialist emitted no required table.
    assert state.recovery_decision({"provision"}).requirement_id == "after"
    step = state.validate_decision(
        GoalDecision(action="continue", requirement_id="after"), {"provision"}
    ).steps[0]
    state.record(step, "r2", "completed")
    assert state.recovery_decision({"provision"}) is None


def test_controller_stagnation_does_not_invent_missing_user_inputs():
    from src.agents.services.orchestrator.analysis_support import missing_input

    blocker = missing_input("stalled")
    assert blocker.owner == "service"


def test_required_calculation_layers_survive_controller_task_rewording():
    goal = AnalysisGoal.model_validate(
        {
            "objective": "Обеспеченность и слои",
            "requirements": [
                {
                    "id": "provision",
                    "agent": "provision",
                    "scenario_id": 91001,
                    "description": "Рассчитай обеспеченность школами и детскими садами",
                    "source_quote": "Верни расчётные слои",
                    "required_artifacts": ["table", "feature_collection"],
                }
            ],
        }
    )
    review = GoalState(AnalysisContext(), goal).validate_decision(
        GoalDecision(
            action="continue", requirement_id="provision", task="Дай краткий ответ"
        ),
        {"provision"},
    )
    assert "слои" in review.steps[0].task.casefold()


async def test_structured_entity_selection_cannot_be_rerouted_by_context_wording(
    monkeypatch, fake_llm, fake_urban, state_store
):
    from src.agents.services.scenario_data.scenario_data_service import (
        ScenarioDataService,
    )
    from src.agents.services.service_entities.orchestrator_plan import EntitySelection
    from tests.unit.test_scenario_data_selection import EntityMcp

    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *a, **kw: fake_llm,
    )
    service = ScenarioDataService("http://llm", None, fake_urban, state_store)
    mcp = EntityMcp()
    events = [
        e
        async for e in service.run_scenario_data_pipeline(
            urban_mcp_client=mcp,
            token="t",
            model="m",
            temperature=0,
            user_query="Получи библиотеки сценария 17. В описании указан статус проект/контекст.",
            scenario_id=17,
            entity_selection=EntitySelection(subject="Библиотека", kind="services"),
            persist_history=False,
        )
    ]
    layers = [e["content"] for e in events if e["type"] == "feature_collection"]
    assert len(layers) == 1
    assert len(layers[0]["feature_collection"]["features"]) == 2


async def test_explicit_calculation_layers_are_part_of_goal_even_if_model_omits_them(
    fake_llm,
):
    import json

    from src.agents.services.orchestrator.analysis_goal import GoalManager

    fake_llm.json_responses = [
        json.dumps(
            {
                "objective": "Рассчитать обеспеченность",
                "requirements": [
                    {
                        "id": "provision",
                        "agent": "provision",
                        "scenario_id": 91001,
                        "description": "Рассчитать обеспеченность школами",
                        "source_ids": [1],
                        "required_artifacts": ["table"],
                    }
                ],
            }
        )
    ]
    goal = await GoalManager(fake_llm).create(
        "m", "Рассчитай обеспеченность школами и верни расчётные слои.", [], 91001
    )
    assert "feature_collection" in goal.requirements[0].required_artifacts
