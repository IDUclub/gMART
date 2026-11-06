"""Unit tests for ``OrchestratorPlanBuilder`` (LLM routing planner)."""

from __future__ import annotations

import json

import pytest

from src.agents.services.orchestrator.orchestrator_catalog import AGENT_CATALOG
from src.agents.services.orchestrator.orchestrator_plan_builder import (
    URBAN_SCENARIO_QUESTION,
    OrchestratorPlanBuilder,
)
from src.agents.services.service_entities.orchestrator_plan import (
    MAX_PLAN_STEPS,
    OrchestratorAgent,
    OrchestratorPlanMode,
)
from tests.helpers import FakeLlmClient


def orchestration_plan_json(
    steps: list[dict] | None = None,
    mode: str = "execute",
    clarification_question: str | None = None,
) -> str:
    return json.dumps(
        {
            "mode": mode,
            "steps": steps or [],
            "clarification_question": clarification_question,
        },
        ensure_ascii=False,
    )


ALL_AGENTS = list(AGENT_CATALOG.values())


@pytest.fixture
def fake_llm() -> FakeLlmClient:
    return FakeLlmClient()


@pytest.fixture
def builder(fake_llm) -> OrchestratorPlanBuilder:
    return OrchestratorPlanBuilder(fake_llm)


@pytest.mark.asyncio
async def test_valid_single_step_plan(builder, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json(
            [{"agent": "provision", "task": "Рассчитай обеспеченность школами"}]
        )
    ]
    plan = await builder.build_plan("m", "обеспеченность школами", ALL_AGENTS)
    assert plan.mode == OrchestratorPlanMode.EXECUTE
    assert len(plan.steps) == 1
    assert plan.steps[0].agent == OrchestratorAgent.PROVISION
    assert plan.steps[0].task == "Рассчитай обеспеченность школами"


@pytest.mark.asyncio
async def test_unavailable_agent_downgrades_to_clarification(builder, fake_llm):
    """A plan referencing an agent excluded from the catalogue → clarification."""
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": "restriction", "task": "Ограничения"}])
    ]
    documents_only = [AGENT_CATALOG[OrchestratorAgent.DOCUMENTS]]
    plan = await builder.build_plan("m", "построй ограничения", documents_only)
    assert plan.mode == OrchestratorPlanMode.NEEDS_CLARIFICATION
    assert not plan.steps
    assert plan.clarification_question


@pytest.mark.asyncio
async def test_excess_steps_require_clarification_without_dropping_work(
    builder, fake_llm
):
    steps = [
        {"agent": "provision", "task": f"задача {i}"} for i in range(MAX_PLAN_STEPS + 2)
    ]
    fake_llm.json_responses = [orchestration_plan_json(steps)]
    plan = await builder.build_plan("m", "много задач", ALL_AGENTS)
    assert plan.mode == OrchestratorPlanMode.NEEDS_CLARIFICATION
    assert not plan.steps
    assert "разделите" in plan.clarification_question


@pytest.mark.asyncio
async def test_invalid_json_retries_then_succeeds(builder, fake_llm):
    fake_llm.json_responses = [
        "это не json",
        orchestration_plan_json([{"agent": "documents", "task": "вопрос"}]),
    ]
    plan = await builder.build_plan("m", "вопрос по нормам", ALL_AGENTS)
    assert plan.mode == OrchestratorPlanMode.EXECUTE
    # first call + one repair round-trip
    assert len(fake_llm.chat_calls) == 2
    # the repair message asks for valid JSON
    assert "невалидный JSON" in fake_llm.chat_calls[1].messages[-1]["content"]


@pytest.mark.asyncio
async def test_invalid_json_exhausts_retries(builder, fake_llm):
    fake_llm.json_responses = ["не json", "опять не json", "и снова"]
    with pytest.raises(ValueError, match="invalid orchestration plan"):
        await builder.build_plan("m", "вопрос", ALL_AGENTS)


@pytest.mark.asyncio
async def test_clarification_without_question_gets_default(builder, fake_llm):
    fake_llm.json_responses = [orchestration_plan_json(mode="needs_clarification")]
    plan = await builder.build_plan("m", "приготовь борщ", ALL_AGENTS)
    assert plan.mode == OrchestratorPlanMode.NEEDS_CLARIFICATION
    assert plan.clarification_question


@pytest.mark.asyncio
async def test_history_is_passed_to_the_llm(builder, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": "documents", "task": "вопрос"}])
    ]
    history = [
        {"role": "user", "content": "прошлый вопрос"},
        {"role": "assistant", "content": "прошлый ответ"},
    ]
    await builder.build_plan("m", "а теперь уточни", ALL_AGENTS, history=history)
    messages = fake_llm.chat_calls[0].messages
    assert messages[0]["role"] == "system"
    assert len(messages) == 2
    request = json.loads(messages[-1]["content"])
    assert request["completed_dialogue_context"] == history
    assert request["current_request"] == "а теперь уточни"


@pytest.mark.asyncio
async def test_planner_runs_deterministically(builder, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": "norms", "task": "вопрос"}])
    ]
    await builder.build_plan("m", "ограничения на школы", ALL_AGENTS)
    assert fake_llm.chat_calls[0].options["temperature"] == 0


@pytest.mark.asyncio
async def test_selected_scenario_reaches_planner_context(builder, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json(
            [{"agent": "provision", "task": "Обеспеченность школами"}]
        )
    ]
    await builder.build_plan("m", "Обеспеченность школами", ALL_AGENTS, scenario_id=848)
    assert "Выбранный scenario_id: 848" in fake_llm.chat_calls[0].messages[0]["content"]


def test_a_base_comparison_goes_to_scenario_data_without_asking_for_an_id():
    prompt = OrchestratorPlanBuilder._build_prompt(ALL_AGENTS, scenario_id=848)

    for rule in (
        "Сравнение выбранного сценария с базовым сценарием проекта — scenario_data",
        "не спрашивай и не требуй его ID или название",
        "task слова «с базовым сценарием»",
        "«Сравни все показатели выбранного сценария с базовым сценарием проекта»",
        "clarification_question никогда не просит ID",
    ):
        assert rule in prompt


@pytest.mark.asyncio
async def test_blank_task_is_repaired_before_dispatch(builder, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": "documents", "task": "   "}]),
        orchestration_plan_json([{"agent": "documents", "task": "Найди текст нормы"}]),
    ]
    plan = await builder.build_plan("m", "Найди текст нормы", ALL_AGENTS)
    assert len(fake_llm.chat_calls) == 2
    assert plan.steps[0].task == "Найди текст нормы"


@pytest.mark.asyncio
async def test_pzz_attachment_manifest_reaches_planner_without_file_contents(
    builder, fake_llm
):
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": "pzz", "task": "Проверь ПЗЗ"}])
    ]
    await builder.build_plan(
        "m",
        "Проверь ПЗЗ",
        ALL_AGENTS,
        scenario_id=843,
        pzz_inputs={
            "mode": "pzz_check",
            "cadastral_upload_id": "private-upload-id",
            "pzz_zones_geojson": {
                "type": "FeatureCollection",
                "features": ["private-geometry"],
            },
            "labels_upload_id": "private-labels-id",
        },
    )
    prompt = fake_llm.chat_calls[0].messages[0]["content"]
    assert '"mode": "pzz_check"' in prompt
    assert '"cadastral_layer": true' in prompt
    assert '"zones_layer": true' in prompt
    assert '"zone_descriptions": true' in prompt
    assert "год и источник зон не нужны" in prompt
    assert "private-" not in prompt


def test_empty_pzz_inputs_still_require_one_pzz_step():
    prompt = OrchestratorPlanBuilder._build_prompt(ALL_AGENTS, pzz_inputs={})

    for layer in (
        "cadastral_layer",
        "zones_layer",
        "buildings_layer",
        "zone_descriptions",
        "classifier",
    ):
        assert f'"{layer}": false' in prompt
    for rule in (
        "Отсутствие входов (все false) — не повод для needs_clarification",
        "всё равно составь\nодин шаг pzz",
        "о недостающих файлах спросит сам агент pzz",
        "не утверждай в task, что файлы переданы",
        "передай его агенту pzz, сохранив файловый режим\nиз запроса",
    ):
        assert rule in prompt


def test_compliance_uploads_rule_remains_in_prompt():
    prompt = OrchestratorPlanBuilder._build_prompt(ALL_AGENTS)

    assert (
        "Без загрузок упоминание файла в тексте не означает,\nчто он передан." in prompt
    )


@pytest.mark.parametrize("routed", ["norms", "documents"])
@pytest.mark.parametrize(
    "query",
    [
        "Какие есть регламенты застройки школ?",
        "Приведи перечень градостроительных ограничений на размещение школы.",
        "В каких документах требования к постройке школ?",
    ],
)
async def test_regulation_overview_runs_norms_then_documents(
    builder, fake_llm, routed, query
):
    fake_llm.json_responses = [
        orchestration_plan_json(
            [{"agent": routed, "task": "Найти регламенты застройки школ"}]
        )
    ]
    plan = await builder.build_plan("m", query, ALL_AGENTS)
    assert [step.agent for step in plan.steps] == [
        OrchestratorAgent.NORMS,
        OrchestratorAgent.DOCUMENTS,
    ]
    assert {step.task for step in plan.steps} == {"Найти регламенты застройки школ"}


@pytest.mark.parametrize(
    "query, agents",
    [
        ("Что в пункте 3.3 СП 42 о школах?", ALL_AGENTS),
        ("Сколько школ на проекте?", ALL_AGENTS),
        ("Какие требования к инсоляции жилых помещений?", ALL_AGENTS),
        (
            "Какие есть регламенты застройки школ?",
            [e for e in ALL_AGENTS if e.key != OrchestratorAgent.NORMS],
        ),
    ],
)
async def test_regulation_completion_leaves_other_plans_alone(
    builder, fake_llm, query, agents
):
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": "documents", "task": query}])
    ]
    plan = await builder.build_plan("m", query, agents)
    assert [step.agent for step in plan.steps] == [OrchestratorAgent.DOCUMENTS]


@pytest.mark.asyncio
async def test_urban_mode_clears_model_steps(builder, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json(
            [{"agent": "compliance", "task": "Проверка"}], mode="urban_analysis"
        )
    ]
    plan = await builder.build_plan(
        "m", "Анализ проекта", ALL_AGENTS, scenario_id=17, urban_analysis_available=True
    )
    assert plan.mode == OrchestratorPlanMode.URBAN_ANALYSIS
    assert plan.steps == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "Проведи градостроительный анализ проекта",
        "сделай градоанализ сценария",
        "нужен град анализ",
        "Хочу провести градостроительный анализ",
        "Проведите градостроительный анализ проекта",
        "Градостроительный анализ проекта нужен сейчас",
    ],
)
@pytest.mark.parametrize("mode", ["execute", "needs_clarification"])
async def test_urban_fallback_promotes_inconsistent_routing(
    builder, fake_llm, query, mode
):
    steps = [{"agent": "norms", "task": "Нормы"}] if mode == "execute" else []
    fake_llm.json_responses = [orchestration_plan_json(steps, mode=mode)]
    plan = await builder.build_plan(
        "m", query, ALL_AGENTS, scenario_id=17, urban_analysis_available=True
    )
    assert plan.mode == OrchestratorPlanMode.URBAN_ANALYSIS
    assert not plan.steps


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,agent,promoted",
    [
        ("Какие нормы нужны для градостроительного анализа?", "norms", False),
        ("Как провести градостроительный анализ?", None, False),
        ("Что нужно для градостроительного анализа?", "norms", False),
        ("Не нужно делать градоанализ, проверь только нормы", "compliance", False),
        ("Сделай граданализ проекта", "compliance", True),
        ("Нужно провести град. анализ сценария", None, True),
        ("Не надо делать градоанализ", "compliance", False),
        ("Не делай градоанализ", "compliance", False),
        ("Не проводи градостроительный анализ", "compliance", False),
        ("Не запускай градостроительный анализ", "compliance", False),
        ("Для проекта не нужно делать градоанализ", "compliance", False),
        ("Сделай град.анализ проекта", "compliance", True),
    ],
)
async def test_urban_fallback_questions_negations_and_abbreviations(
    builder, fake_llm, query, agent, promoted
):
    mode = "execute" if agent else "needs_clarification"
    steps = [{"agent": agent, "task": query}] if agent else []
    fake_llm.json_responses = [orchestration_plan_json(steps, mode=mode)]
    plan = await builder.build_plan(
        "m", query, ALL_AGENTS, scenario_id=17, urban_analysis_available=True
    )
    assert plan.mode == (OrchestratorPlanMode.URBAN_ANALYSIS if promoted else mode)
    if promoted:
        assert not plan.steps
    elif agent:
        assert plan.steps[0].agent == agent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario,available,question",
    [
        (
            None,
            True,
            "Градостроительный анализ выполняется только для выбранного сценария. Выберите сценарий и повторите запрос.",
        ),
        (
            None,
            False,
            "Градостроительный анализ выполняется только для выбранного сценария. Выберите сценарий и повторите запрос.",
        ),
        (17, False, "Сервис градостроительного анализа не настроен."),
    ],
)
async def test_urban_prerequisites(builder, fake_llm, scenario, available, question):
    fake_llm.json_responses = [orchestration_plan_json(mode="urban_analysis")]
    plan = await builder.build_plan(
        "m",
        "Проведи градоанализ",
        ALL_AGENTS,
        scenario_id=scenario,
        urban_analysis_available=available,
    )
    assert plan.mode == OrchestratorPlanMode.NEEDS_CLARIFICATION
    assert plan.clarification_question == question


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,agent",
    [
        ("Что такое градостроительный анализ?", "documents"),
        ("Хочу узнать, что такое градостроительный анализ", "norms"),
        ("Построй буфер 100 метров от рек", "restriction"),
    ],
)
async def test_urban_fallback_preserves_other_work(builder, fake_llm, query, agent):
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": agent, "task": query}])
    ]
    plan = await builder.build_plan(
        "m", query, ALL_AGENTS, scenario_id=17, urban_analysis_available=True
    )
    assert plan.mode == OrchestratorPlanMode.EXECUTE
    assert plan.steps[0].agent == agent


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_mode", ["urban_analysis", "execute"])
@pytest.mark.parametrize(
    "scenario_id,question",
    [
        (
            17,
            "Градостроительный анализ запускается отдельным запросом. "
            "Отправьте анализ и остальные задачи отдельными сообщениями.",
        ),
        (None, URBAN_SCENARIO_QUESTION),
    ],
)
async def test_urban_fallback_clarifies_mixed_request(
    builder, fake_llm, scenario_id, question, llm_mode
):
    query = "Проведи градостроительный анализ и построй буфер 100 метров от рек"
    steps = [{"agent": "restriction", "task": query}] if llm_mode == "execute" else []
    fake_llm.json_responses = [orchestration_plan_json(steps, mode=llm_mode)]
    plan = await builder.build_plan(
        "m",
        query,
        ALL_AGENTS,
        scenario_id=scenario_id,
        urban_analysis_available=True,
    )
    assert plan.mode == OrchestratorPlanMode.NEEDS_CLARIFICATION
    assert plan.clarification_question == question
    assert not plan.steps


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,llm_agent,expected_mode",
    [
        (
            "Проведи градостроительный анализ и построй буфер 100 метров от рек",
            None,
            "needs_clarification",
        ),
        (
            "Проведи градостроительный анализ и построй буфер 100 метров от рек",
            "restriction",
            "needs_clarification",
        ),
        (
            "Сделай градоанализ. Затем посчитай школы",
            None,
            "needs_clarification",
        ),
        ("Сделай градоанализ и покажи результат", None, "urban_analysis"),
        (
            "Проведи градостроительный анализ и проверь нормы",
            None,
            "urban_analysis",
        ),
        ("Проведи градостроительный анализ проекта", None, "urban_analysis"),
    ],
)
async def test_urban_extra_task_independent_of_llm_routing(
    builder, fake_llm, query, llm_agent, expected_mode
):
    steps = [{"agent": llm_agent, "task": query}] if llm_agent else []
    fake_llm.json_responses = [
        orchestration_plan_json(
            steps, mode="execute" if llm_agent else "urban_analysis"
        )
    ]
    plan = await builder.build_plan(
        "m", query, ALL_AGENTS, scenario_id=17, urban_analysis_available=True
    )
    assert plan.mode == expected_mode
    assert not plan.steps
    if expected_mode == "needs_clarification":
        assert plan.clarification_question == (
            "Градостроительный анализ запускается отдельным запросом. "
            "Отправьте анализ и остальные задачи отдельными сообщениями."
        )


@pytest.mark.asyncio
async def test_report_agent_is_not_available_to_planner(builder, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": "urban_analysis_report", "task": "Отчёт"}])
    ]
    plan = await builder.build_plan(
        "m",
        "Проведи градостроительный анализ",
        ALL_AGENTS,
        scenario_id=17,
        urban_analysis_available=True,
    )
    assert plan.mode == OrchestratorPlanMode.NEEDS_CLARIFICATION
    assert OrchestratorAgent.URBAN_ANALYSIS_REPORT not in AGENT_CATALOG
