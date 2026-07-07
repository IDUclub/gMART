"""Unit tests for the provision intent plan: modes, legacy mapping, canonicalization."""

from __future__ import annotations

import json

import pytest

from src.agents.services.provision_plan_builder import ProvisionPlanBuilder
from src.agents.services.service_entities.provision_plan import (
    ProvisionPlan,
    ProvisionPlanMode,
)

CATALOG = ["Школы", "Детские сады", "Поликлиники"]


class FakeLlmClient:
    """Returns a canned JSON plan and records the messages it was called with."""

    def __init__(self, plan: dict) -> None:
        self.plan = plan
        self.calls: list[dict] = []

    async def chat(self, model, messages=None, options=None, **kwargs):
        self.calls.append({"model": model, "messages": messages or kwargs})
        return {"message": {"content": json.dumps(self.plan, ensure_ascii=False)}}


def build_builder(plan: dict) -> ProvisionPlanBuilder:
    return ProvisionPlanBuilder(FakeLlmClient(plan))


# ---------------------------------------------------------------------------
# ProvisionPlan model
# ---------------------------------------------------------------------------


def test_legacy_found_mode_maps_to_effects():
    plan = ProvisionPlan.model_validate({"mode": "found", "service_name": "Школы"})
    assert plan.mode == ProvisionPlanMode.EFFECTS
    assert plan.service_name == "Школы"


def test_all_modes_are_accepted():
    for mode in ("effects", "provision", "summary", "list_services"):
        assert ProvisionPlan(mode=mode).mode == ProvisionPlanMode(mode)


# ---------------------------------------------------------------------------
# Plan builder canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["effects", "provision"])
async def test_single_service_name_is_canonicalized(mode):
    builder = build_builder({"mode": mode, "service_name": "школы"})
    plan = await builder.build_plan("model", "запрос", CATALOG)
    assert plan.mode == ProvisionPlanMode(mode)
    assert plan.service_name == "Школы"


@pytest.mark.asyncio
async def test_unknown_service_name_needs_clarification():
    builder = build_builder({"mode": "effects", "service_name": "Аптеки"})
    plan = await builder.build_plan("model", "запрос", CATALOG)
    assert plan.mode == ProvisionPlanMode.NEEDS_CLARIFICATION
    assert "Аптеки" in plan.clarification_question
    assert "Школы" in plan.clarification_question


@pytest.mark.asyncio
async def test_summary_names_are_canonicalized_and_unknown_dropped():
    builder = build_builder(
        {
            "mode": "summary",
            "service_names": ["школы", "Аптеки", "детские сады"],
            "layer_service_names": ["школы"],
        }
    )
    plan = await builder.build_plan("model", "запрос", CATALOG)
    assert plan.mode == ProvisionPlanMode.SUMMARY
    assert plan.service_names == ["Школы", "Детские сады"]
    assert plan.layer_service_names == ["Школы"]


@pytest.mark.asyncio
async def test_summary_with_only_unknown_names_needs_clarification():
    builder = build_builder({"mode": "summary", "service_names": ["Аптеки"]})
    plan = await builder.build_plan("model", "запрос", CATALOG)
    assert plan.mode == ProvisionPlanMode.NEEDS_CLARIFICATION


@pytest.mark.asyncio
async def test_summary_without_names_means_full_catalog():
    builder = build_builder({"mode": "summary"})
    plan = await builder.build_plan("model", "запрос", CATALOG)
    assert plan.mode == ProvisionPlanMode.SUMMARY
    assert plan.service_names == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["summary", "provision"])
async def test_target_population_is_preserved(mode):
    plan_payload = {"mode": mode, "target_population": 25000}
    if mode == "provision":
        plan_payload["service_name"] = "Школы"
    builder = build_builder(plan_payload)
    plan = await builder.build_plan("model", "запрос", CATALOG)
    assert plan.target_population == 25000


@pytest.mark.asyncio
async def test_history_is_passed_to_llm_before_current_query():
    builder = build_builder({"mode": "list_services"})
    history = [
        {"role": "user", "content": "старый вопрос"},
        {"role": "assistant", "content": "старый ответ"},
    ]
    await builder.build_plan("model", "текущий запрос", CATALOG, history=history)
    messages = builder.llm_client.calls[0]["messages"]
    roles_and_content = [(m["role"], m["content"]) for m in messages]
    assert roles_and_content[0][0] == "system"
    assert roles_and_content[1:] == [
        ("user", "старый вопрос"),
        ("assistant", "старый ответ"),
        ("user", "текущий запрос"),
    ]


@pytest.mark.asyncio
async def test_list_services_passes_through():
    builder = build_builder({"mode": "list_services"})
    plan = await builder.build_plan("model", "какие сервисы есть?", CATALOG)
    assert plan.mode == ProvisionPlanMode.LIST_SERVICES


@pytest.mark.asyncio
async def test_clarification_without_question_gets_default_text():
    builder = build_builder({"mode": "needs_clarification"})
    plan = await builder.build_plan("model", "запрос", CATALOG)
    assert plan.mode == ProvisionPlanMode.NEEDS_CLARIFICATION
    assert plan.clarification_question
    assert "Школы" in plan.clarification_question
