"""Unit tests for ``OrchestratorPlanBuilder`` (LLM routing planner)."""

from __future__ import annotations

import json

import pytest

from src.agents.services.orchestrator_catalog import AGENT_CATALOG
from src.agents.services.orchestrator_plan_builder import OrchestratorPlanBuilder
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
async def test_steps_truncated_to_max(builder, fake_llm):
    steps = [
        {"agent": "provision", "task": f"задача {i}"} for i in range(MAX_PLAN_STEPS + 2)
    ]
    fake_llm.json_responses = [orchestration_plan_json(steps)]
    plan = await builder.build_plan("m", "много задач", ALL_AGENTS)
    assert len(plan.steps) == MAX_PLAN_STEPS


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
    assert messages[1:3] == history
    assert messages[-1] == {"role": "user", "content": "а теперь уточни"}


@pytest.mark.asyncio
async def test_planner_runs_deterministically(builder, fake_llm):
    fake_llm.json_responses = [
        orchestration_plan_json([{"agent": "norms", "task": "вопрос"}])
    ]
    await builder.build_plan("m", "ограничения на школы", ALL_AGENTS)
    assert fake_llm.chat_calls[0].options["temperature"] == 0
