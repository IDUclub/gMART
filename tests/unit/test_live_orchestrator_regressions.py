"""Regressions observed with the live orchestrator, 2026-09-10."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.services.dvd.dvd_reasoning import RetrievalPlanner
from src.agents.services.provision.provision_plan_builder import ProvisionPlanBuilder
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.service_entities.dvd_plan import validate_retrieval_plan
from src.agents.services.service_entities.restriction_plan import RestrictionPlan


async def test_dvd_planner_constrains_non_nullable_search_modes():
    client = AsyncMock()
    client.chat.return_value = {
        "message": {
            "content": json.dumps(
                {"retrieval_mode": "semantic", "search_query": "инсоляция"}
            )
        }
    }
    await RetrievalPlanner(client).build_plan("m", "Найди требования к инсоляции")
    args = client.chat.await_args.kwargs
    assert args["think"] is False
    schema = args["format"]
    assert schema["discriminator"]["propertyName"] == "retrieval_mode"
    assert len(schema["oneOf"]) == 3
    assert "name_query" in schema["$defs"]["NameRetrievalPlan"]["required"]
    assert "pattern" in schema["$defs"]["StructureRetrievalPlan"]["required"]
    assert args["options"]["num_predict"] >= 1024


def test_literal_null_is_not_a_fragment_address():
    plan = validate_retrieval_plan(
        {
            "retrieval_mode": "semantic",
            "pattern": "null",
            "name_query": "null",
            "doc_id": "None",
            "version": "",
        }
    )
    plan = RetrievalPlanner._clamp(plan, "Найди требования к инсоляции")
    assert plan.retrieval_mode == "semantic"
    assert plan.pattern is None and plan.doc_id is None and plan.name_query is None


async def test_provision_schema_uses_actual_catalog_not_pluralized_example():
    client = AsyncMock()
    client.chat.return_value = {
        "message": {
            "content": json.dumps({"mode": "provision", "service_name": "Школа"})
        }
    }
    await ProvisionPlanBuilder(client).build_plan(
        "m", "Обеспеченность школами", ["Школа", "Детский сад"]
    )
    args = client.chat.await_args.kwargs
    assert args["think"] is False
    assert args["format"]["properties"]["service_name"]["enum"] == [
        "Школа",
        "Детский сад",
        None,
    ]
    assert args["format"]["properties"]["service_names"]["items"]["enum"] == [
        "Школа",
        "Детский сад",
    ]


async def test_empty_provision_catalog_needs_no_invalid_empty_enum_request():
    client = AsyncMock()
    plan = await ProvisionPlanBuilder(client).build_plan(
        "m", "Обеспеченность школами", []
    )
    assert plan.mode == "needs_clarification"
    client.chat.assert_not_awaited()


@pytest.mark.parametrize("distance", ["50 м", "50 метров", "50-метровой", "0,1 км"])
async def test_compliance_does_not_drop_user_distance_when_graph_is_empty(distance):
    service = object.__new__(RestrictionParserService)
    service.state_store = SimpleNamespace(
        new_request_id=lambda: "test",
        create=AsyncMock(),
        get_checkpoint=AsyncMock(return_value={}),
        save_checkpoint=AsyncMock(),
        set_status=AsyncMock(),
    )
    service._buf = AsyncMock(side_effect=lambda _id, event: event)
    service.compliance_result_harness = SimpleNamespace(
        prepare_follow_up=lambda *args: None
    )
    service.normgraph_retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=SimpleNamespace(
                restrictions=[],
                unsupported_count=0,
                tool_call={},
            )
        )
    )
    service._build_plan = AsyncMock(
        return_value=RestrictionPlan(
            mode="needs_clarification",
            original="query",
            clarification_question="Какие дома проверить?",
        )
    )
    events = [
        event
        async for event in service._run_restriction_execution_pipline(
            mcp_client=object(),
            temperature=0,
            model="m",
            user_query=f"Проверь дома по условию {distance} от дороги",
            scenario_id=772,
            token_ref=["test"],
            persist_history=False,
            normgraph_mcp_client=object(),
            history_agent="compliance",
        )
    ]
    service._build_plan.assert_awaited_once()
    assert distance in service._build_plan.await_args.args[2]
    assert any(e["type"] == "clarification" for e in events)
    assert not any(e["type"] == "compliance_summary" for e in events)
