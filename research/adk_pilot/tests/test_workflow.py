"""What the ADK version adds over the current pipeline: a checked graph,
retry policies on MCP steps, a harness that sees every step, and isolation of
concurrent requests sharing one workflow."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from google.adk.workflow import RetryConfig

from research.adk_pilot.gmart_adk.harness import AuditPlugin
from research.adk_pilot.gmart_adk.restriction.workflow import current_mcp_client
from research.adk_pilot.tests.fakes import (
    MODEL,
    Scenario,
    adk_service,
    plan_json,
    run_adk,
    run_legacy,
    tool_executor,
)

OBJECTS_QUERY = "Какие жилые дома попадают в зону 100 м от школ? Выведи объекты."


def test_graph_declares_every_branch():
    service = adk_service(Scenario("q", []).llm())
    graph = service.workflow.graph
    routes = {
        (edge.from_node.name, edge.route, edge.to_node.name)
        for edge in graph.edges
        if edge.route is not None
    }
    assert routes == {
        ("check_plan", "semantic_repair", "planner_semantic_repair"),
        ("check_plan", "names_repair", "planner_names_repair"),
        ("check_plan", "clarify", "clarification"),
        ("check_plan", "explain", "announce_explanation"),
        ("build_buffers", "buffers_only", "report_buffers"),
        ("build_buffers", "restrictions", "announce_restrictions"),
    }
    loops = {(e.from_node.name, e.to_node.name) for e in graph.edges if e.route is None}
    assert ("planner_semantic_repair", "check_plan") in loops
    assert ("planner_names_repair", "check_plan") in loops


async def test_mcp_step_is_retried_without_repeating_its_events():
    scenario = Scenario(OBJECTS_QUERY, [plan_json("restrictions")])
    flaky = tool_executor()
    layers = flaky.retrieve_layers_for_plan.return_value
    flaky.retrieve_layers_for_plan = AsyncMock(
        side_effect=[ConnectionError("MCP dropped the call"), layers]
    )

    events, _ = await run_adk(
        scenario,
        tool_executor=flaky,
        tool_retry=RetryConfig(max_attempts=2, initial_delay=0),
    )
    legacy_events, _ = await run_legacy(scenario)

    assert flaky.retrieve_layers_for_plan.await_count == 2
    assert events == legacy_events


async def test_audit_plugin_sees_every_model_call_and_node():
    audit = AuditPlugin()
    scenario = Scenario(
        OBJECTS_QUERY, [plan_json("buffers_only"), plan_json("restrictions")]
    )
    await run_adk(scenario, plugins=[audit])

    (run,) = audit.runs.values()
    assert [call.agent for call in run.model_calls] == [
        "planner",
        "planner_semantic_repair",
        "explainer",
        "responder",
    ]
    assert {call.model for call in run.model_calls} == {MODEL}
    assert run.nodes[:5] == [
        "load_catalogs",
        "planner",
        "check_plan",
        "planner_semantic_repair",
        "check_plan",
    ]
    assert run.nodes[-1] == "finish_response"


async def test_concurrent_requests_keep_their_own_mcp_client():
    executor = tool_executor()
    seen: list[object] = []
    original = executor.retrieve_layers_for_plan

    async def record(mcp_client, plan, scenario_id):
        seen.append(mcp_client)
        await asyncio.sleep(0.01)  # let the other request interleave
        assert current_mcp_client.get() is mcp_client
        return await original(mcp_client, plan, scenario_id)

    executor.retrieve_layers_for_plan = record
    llm = Scenario(OBJECTS_QUERY, []).llm()
    llm.json_responses = [plan_json("restrictions")] * 2
    llm.answer_texts = ["a", "b", "c", "d"]
    service = adk_service(llm, tool_executor=executor)
    clients = [object(), object()]

    async def run(i: int):
        return [
            event
            async for event in service.run_restriction_execution_pipline(
                mcp_client=clients[i],
                temperature=0.5,
                model=MODEL,
                user_query=OBJECTS_QUERY,
                scenario_id=1,
                request_id=f"req-{i}",
            )
        ]

    results = await asyncio.gather(run(0), run(1))
    assert sorted(map(id, seen)) == sorted(map(id, clients))
    assert all(result[-1]["content"]["done"] for result in results)
