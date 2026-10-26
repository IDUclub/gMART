"""The ADK workflow behaves exactly like RestrictionParserService.

Both engines get the same scripted LLM and the same MCP results; each case
asserts that they emit the same pipeline items and send the model the same
requests (messages, options, schema, think, stream), and checks the case
actually took the branch it is named after.
"""

from __future__ import annotations

import pytest

from research.adk_pilot.tests.fakes import Scenario, plan_json, run_adk, run_legacy

OBJECTS_QUERY = "Какие жилые дома попадают в зону 100 м от школ? Выведи объекты."
BUFFER_QUERY = "Построй зону 100 м вокруг школ"


def _types(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


SCENARIOS = {
    "restrictions": Scenario(OBJECTS_QUERY, [plan_json("restrictions")]),
    "buffers_only": Scenario(
        BUFFER_QUERY, [plan_json("buffers_only")], texts=["Объяснение."]
    ),
    "clarification": Scenario("Покажи зону", [plan_json("needs_clarification")]),
    "semantic_repair": Scenario(
        OBJECTS_QUERY, [plan_json("buffers_only"), plan_json("restrictions")]
    ),
    "semantic_repair_fails": Scenario(
        OBJECTS_QUERY, [plan_json("buffers_only"), plan_json("buffers_only")]
    ),
    "names_repair": Scenario(
        BUFFER_QUERY,
        [
            plan_json(
                "buffers_only",
                source_entities=[{"name": "Больница", "entity_type": "service"}],
                buffer_rules=[
                    {"source_name": "Больница", "buffer_size": 100, "title": "Зона"}
                ],
            ),
            plan_json("buffers_only"),
        ],
        texts=["Объяснение."],
    ),
    "invalid_json": Scenario(OBJECTS_QUERY, ["{ not json", plan_json("restrictions")]),
    "history": Scenario(
        OBJECTS_QUERY,
        [plan_json("restrictions")],
        history=[
            {"role": "user", "content": "Что рядом со школами?"},
            {"role": "assistant", "content": "Жилые дома и парки."},
        ],
    ),
    "empty_answer_fallback": Scenario(
        OBJECTS_QUERY, [plan_json("restrictions")], texts=["Объяснение.", ""]
    ),
}


@pytest.mark.parametrize("name", SCENARIOS)
async def test_same_events_and_model_requests(name):
    scenario = SCENARIOS[name]
    legacy_events, legacy_llm = await run_legacy(scenario)
    adk_events, adk_llm = await run_adk(scenario)

    assert adk_events == legacy_events
    assert adk_llm.requests() == legacy_llm.requests()


async def test_scenarios_take_their_branch():
    async def run(name):
        events, llm = await run_adk(SCENARIOS[name])
        return _types(events), llm.requests()

    types_, requests = await run("restrictions")
    assert types_.count("tool_call") == 3 and len(requests) == 3

    types_, requests = await run("buffers_only")
    assert types_.count("tool_call") == 2 and len(requests) == 2

    types_, requests = await run("clarification")
    assert types_[-1] == "clarification" and len(requests) == 1

    # One plan request + one repair request, repair sees only its system prompt.
    for name in ("semantic_repair", "names_repair"):
        types_, requests = await run(name)
        assert "clarification" not in types_
        assert [m["role"] for m in requests[1]["messages"]] == ["system"]
    assert "Больница" in requests[1]["messages"][0]["content"]

    types_, _ = await run("semantic_repair_fails")
    assert types_[-1] == "clarification"

    _, requests = await run("invalid_json")
    assert [m["role"] for m in requests[1]["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]

    _, requests = await run("history")
    assert [m["role"] for m in requests[0]["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
