"""Provider-output regressions captured by the immutable September live series."""

import json

import pytest

from src.agents.services.orchestrator.analysis_goal import GoalManager


@pytest.mark.parametrize(
    "status,kind,expected",
    [
        ("pending", "other", "medium"),
        ("blocked", "other", "medium"),
        ("satisfied", "services", "medium"),
        ("satisfied", "other", "high"),
    ],
)
async def test_high_reasoning_is_reserved_for_ready_analytical_synthesis(
    fake_llm, status, kind, expected
):
    complete = fake_llm.chat
    calls = []

    async def record(*args, **kwargs):
        calls.append(kwargs)
        return await complete(*args, **kwargs)

    fake_llm.chat = record
    fake_llm.json_responses = [
        json.dumps(
            {"action": "complete", "answer": "Синтез", "evidence_ids": ["evidence"]}
        )
    ]
    await GoalManager(fake_llm).review(
        "m",
        "Сравнить результаты",
        [],
        {
            "goal": {
                "objective": "Сравнение",
                "requirements": [
                    {
                        "id": "r",
                        "agent": "documents",
                        "status": status,
                        "entity_kind": kind,
                    }
                ],
            }
        },
        [],
        {},
    )
    assert calls[0]["reasoning_effort"] == expected


async def test_school_provision_cannot_expand_to_all_services(fake_llm):
    from src.agents.services.provision.provision_plan_builder import (
        ProvisionPlanBuilder,
    )

    fake_llm.json_responses = [
        json.dumps(
            {"mode": "summary", "service_names": [], "layer_service_names": ["Школа"]}
        ),
        json.dumps({"mode": "provision", "service_name": "Школа"}),
    ]
    plan = await ProvisionPlanBuilder(fake_llm).build_plan(
        "m",
        "Рассчитать обеспеченность школами, вернуть таблицу, слой и аналитический разбор.",
        ["Школа", "Детский сад", "Парк"],
    )
    assert plan.mode.value == "provision"
    assert plan.service_name == "Школа"


async def test_explicit_all_service_summary_remains_supported(fake_llm):
    from src.agents.services.provision.provision_plan_builder import (
        ProvisionPlanBuilder,
    )

    fake_llm.json_responses = [
        json.dumps(
            {"mode": "summary", "service_names": [], "layer_service_names": ["Школа"]}
        )
    ]
    plan = await ProvisionPlanBuilder(fake_llm).build_plan(
        "m",
        "Сводка по всем услугам, а слой только для школы.",
        ["Школа", "Парк"],
    )
    assert plan.mode.value == "summary" and plan.service_names == []


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, None])
async def test_goal_creation_recovers_one_transient_model_failure(fake_llm, status):
    from src.agents.model_clients.llm_base import LlmResponseError

    complete = fake_llm.chat
    calls = []

    async def flaky(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise LlmResponseError("synthetic transient failure", status)
        return await complete(*args, **kwargs)

    fake_llm.chat = flaky
    fake_llm.json_responses = [
        json.dumps(
            {
                "objective": "Получить школы",
                "requirements": [
                    {
                        "id": "schools",
                        "description": "Получить школы",
                        "source_ids": [1],
                        "agent": "scenario_data",
                        "scenario_id": 772,
                        "subject": "Школа",
                        "entity_kind": "services",
                        "required_artifacts": ["table"],
                    }
                ],
            }
        )
    ]
    goal = await GoalManager(fake_llm).create("m", "Получить школы.", [], 772)
    assert len(calls) == 2
    assert goal.requirements[0].subject == "Школа"


@pytest.mark.parametrize(
    "status,expected_calls", [(401, 1), (400, 1), (404, 1), (503, 2)]
)
async def test_goal_model_failures_are_bounded_and_permanent_errors_not_retried(
    fake_llm, status, expected_calls
):
    from src.agents.model_clients.llm_base import LlmResponseError

    calls = []

    async def unavailable(*args, **kwargs):
        calls.append(kwargs)
        raise LlmResponseError("synthetic failure", status)

    fake_llm.chat = unavailable
    with pytest.raises(LlmResponseError):
        await GoalManager(fake_llm).create("m", "Получить школы.", [], 772)
    assert len(calls) == expected_calls


async def test_reasoning_fallback_remains_active_for_later_controller_actions(fake_llm):
    from src.agents.runtime.budget import RunBudget, budget_scope

    efforts = []

    async def respond(*args, **kwargs):
        efforts.append(kwargs["reasoning_effort"])
        return {
            "message": {
                "content": (
                    ""
                    if len(efforts) == 1
                    else json.dumps(
                        {
                            "action": "complete",
                            "answer": "Подтверждено",
                            "evidence_ids": ["a1"],
                        }
                    )
                )
            },
            "done_reason": "length" if len(efforts) == 1 else "stop",
        }

    fake_llm.chat = respond
    context = {
        "goal": {
            "objective": "Сравнить источники",
            "requirements": [
                {"id": "source", "status": "satisfied", "entity_kind": "other"}
            ],
        }
    }
    budget = RunBudget()
    with budget_scope(budget):
        manager = GoalManager(fake_llm)
        await manager.review("m", "Сравнить источники", [], context, [], {})
        await manager.review("m", "Сравнить источники", [], context, [], {})
    assert efforts == ["high", "medium", "medium"]
    assert budget.reasoning_fallbacks == 1


@pytest.mark.parametrize(
    "description,artifacts",
    [
        ("Сопоставь количества школ и детских садов.", ["analysis_text"]),
        (
            "Получить таблицу с количеством школ и детских садов и рассчитать разницу количества.",
            ["table"],
        ),
        ("Проверить наличие данных о школах и детских садах.", ["analysis_text"]),
    ],
)
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("comparison_kind", ["other", "services"])
async def test_comparing_retrieved_counts_is_not_a_new_data_fetch(
    fake_llm, description, artifacts, filtered, comparison_kind
):
    independent = (
        comparison_kind == "services"
        and description.startswith("Проверить наличие")
        and not filtered
    )
    comparison_subject = "Поликлиника" if independent else "сопоставление количеств"
    if independent:
        description = "Проверить наличие данных о поликлиниках."
    if filtered:
        description += " Только в радиусе 500 метров от заданного адреса."
    requirements = [
        {
            "id": f"type-{index}",
            "description": f"Получи услуги типа {subject}: таблицу и слой.",
            "source_ids": [1],
            "agent": "scenario_data",
            "scenario_id": 772,
            "subject": subject,
            "entity_kind": "services",
            "required_artifacts": ["table", "feature_collection"],
        }
        for index, subject in enumerate(["Школа", "Детский сад"])
    ]
    valid = {
        "objective": "Сопоставить количества школ и детских садов",
        "requirements": requirements,
    }
    invalid = {
        **valid,
        "requirements": [
            *requirements,
            {
                "id": "comparison",
                "description": description,
                "source_ids": [1],
                "agent": "scenario_data",
                "scenario_id": 772,
                "subject": comparison_subject,
                "entity_kind": "other" if filtered else comparison_kind,
                "required_artifacts": artifacts,
            },
        ],
    }
    fake_llm.json_responses = [json.dumps(invalid), json.dumps(valid)]
    goal = await GoalManager(fake_llm).create(
        "m", "Получи школы и детские сады и сопоставь их количества.", [], 772
    )
    expected = ["Школа", "Детский сад"]
    if filtered or independent:
        expected.append(comparison_subject)
    if filtered:
        assert "радиусе 500 метров" in goal.requirements[-1].description
    assert [r.subject for r in goal.requirements] == expected
    assert "Сопоставить" in goal.objective


@pytest.mark.parametrize("agent", ["scenario_data", "restriction", "compliance"])
@pytest.mark.parametrize("independent_spatial_check", [False, True])
async def test_provision_failure_diagnosis_is_not_an_independent_specialist_goal(
    fake_llm, agent, independent_spatial_check
):
    calculation = {
        "id": "provision",
        "description": "Вычислить обеспеченность школами в сценарии 772 и проверить причину недоступности расчёта.",
        "agent": "provision",
        "scenario_id": 772,
        "subject": "Школа",
        "entity_kind": "other",
        "required_artifacts": ["table"],
        "source_ids": [1],
    }
    valid = {
        "objective": "Рассчитать обеспеченность или сообщить причину невозможности расчёта",
        "requirements": [calculation],
    }
    invalid = {
        **valid,
        "requirements": [
            calculation,
            {
                "id": "diagnosis",
                "description": "Проверить причину недоступности расчёта обеспеченности школами.",
                "agent": agent,
                "scenario_id": 772,
                "entity_kind": "other",
                "required_artifacts": ["analysis_text"],
                "source_ids": [1],
            },
        ],
    }
    if independent_spatial_check:
        invalid["requirements"][-1][
            "description"
        ] = "Проверить соблюдение минимального расстояния от школы до стоянки."
    fake_llm.json_responses = [json.dumps(invalid), json.dumps(valid)]
    goal = await GoalManager(fake_llm).create(
        "m",
        "Рассчитай обеспеченность школами; если расчёт недоступен, проверь причину через сервис.",
        [],
        772,
    )
    assert [r.agent for r in goal.requirements] == (
        ["provision", agent] if independent_spatial_check else ["provision"]
    )
    assert "причину" in goal.requirements[0].description


@pytest.mark.parametrize(
    "transport,constrained", [("chat", False), ("responses_function", True)]
)
async def test_goal_controller_uses_opted_in_structured_transport(
    fake_llm, transport, constrained
):
    from pydantic import BaseModel

    class Decision(BaseModel):
        answer: str

    fake_llm.structured_transport = transport
    fake_llm.json_responses = ['{"answer":"ok"}']
    complete = fake_llm.chat
    calls = []

    async def record(*args, **kwargs):
        calls.append(kwargs)
        return await complete(*args, **kwargs)

    fake_llm.chat = record
    result = await GoalManager(fake_llm)._call(
        "m", "test", "Decide", {}, Decision, reasoning_effort="low"
    )
    assert result.answer == "ok"
    assert bool(calls[0].get("format")) is constrained
