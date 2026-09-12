"""Exercise the real OpenAI client with a hermetic HTTP boundary, including streams."""

import json
from dataclasses import replace

import httpx
import pytest
from openai import AsyncOpenAI

from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter
from src.agents.runtime.budget import (
    BudgetExceeded,
    BudgetLimits,
    RunBudget,
    budget_scope,
)
from src.agents.runtime.tools import execute_planned


def response(reason="stop", tokens=42):
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-oss-20b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"value": 42}'},
                "finish_reason": reason,
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": tokens - 12,
            "total_tokens": tokens,
        },
    }


@pytest.mark.asyncio
async def test_high_effort_and_each_structured_retry_are_charged():
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200, json=response("length" if len(calls) == 1 else "stop")
        )

    adapter = OpenAiCompatAdapter("http://test/v1")
    adapter.client = AsyncOpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    budget = RunBudget()
    try:
        with budget_scope(budget):
            result = await adapter.chat(
                "gpt-oss-20b",
                [{"role": "user", "content": "42"}],
                think=False,
                reasoning_effort="high",
                format="json",
                options={"num_predict": 2048},
            )
        assert result.message.content
        assert budget.model_calls == 2 and budget.tokens == 84
        assert all(call["reasoning_effort"] == "high" for call in calls)
    finally:
        await adapter.client.close()


@pytest.mark.asyncio
async def test_usage_trailer_is_counted_before_terminal_chunk():
    def handle(request):
        assert json.loads(request.content)["stream_options"]["include_usage"]
        chunks = [
            {
                "choices": [
                    {"index": 0, "delta": {"content": "answer"}, "finish_reason": None}
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                    "total_tokens": 12,
                },
            },
        ]
        data = (
            "".join(
                "data: "
                + json.dumps(
                    {
                        "id": "c",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "m",
                        **chunk,
                    }
                )
                + "\n\n"
                for chunk in chunks
            )
            + "data: [DONE]\n\n"
        )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=data
        )

    adapter = OpenAiCompatAdapter("http://test/v1")
    adapter.client = AsyncOpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    budget = RunBudget()
    try:
        with budget_scope(budget):
            stream = await adapter.chat(
                "m", [{"role": "user", "content": "q"}], stream=True
            )
            async for part in stream:
                if part.done:
                    assert budget.tokens == 12
                    assert part.usage.total_tokens == 12
                    break
            await stream.aclose()
        assert budget.model_calls == 1 and budget.estimated_calls == 0
    finally:
        await adapter.client.close()


@pytest.mark.asyncio
async def test_nested_tools_share_cap_and_do_not_execute_after_limit():
    calls = []

    async def operation():
        calls.append(1)
        return 42

    budget = RunBudget(replace(BudgetLimits(), tool_calls=1))
    with budget_scope(budget):
        assert await execute_planned("mcp.first", operation) == 42
        with pytest.raises(BudgetExceeded):
            await execute_planned("urban.second", operation)
    assert len(calls) == 1 and budget.tool_calls == 1


@pytest.mark.asyncio
async def test_network_failure_does_not_trigger_hidden_client_retries():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(500, json={"error": {"message": "unavailable"}})

    adapter = OpenAiCompatAdapter("http://test/v1")
    adapter.client = AsyncOpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    budget = RunBudget()
    try:
        with budget_scope(budget), pytest.raises(Exception):
            await adapter.chat("m", [{"role": "user", "content": "q"}])
        assert len(calls) == budget.model_calls == budget.estimated_calls == 1
    finally:
        await adapter.client.close()


@pytest.mark.asyncio
async def test_truncated_high_plan_is_repaired_with_bounded_medium_attempt():
    from src.agents.services.orchestrator.orchestrator_catalog import AGENT_CATALOG
    from src.agents.services.orchestrator.orchestrator_plan_builder import (
        OrchestratorPlanBuilder,
    )

    calls = []

    def handle(request):
        call = json.loads(request.content)
        calls.append(call)
        payload = response("length" if len(calls) == 1 else "stop")
        payload["choices"][0]["message"]["content"] = (
            ""
            if len(calls) == 1
            else json.dumps(
                {
                    "mode": "execute",
                    "analytical": True,
                    "steps": [{"agent": "scenario_data", "task": "Получи таблицу"}],
                }
            )
        )
        return httpx.Response(200, json=payload)

    adapter = OpenAiCompatAdapter("http://test/v1")
    adapter.client = AsyncOpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    budget = RunBudget()
    try:
        with budget_scope(budget):
            result = await OrchestratorPlanBuilder(adapter).build_plan(
                "gpt-oss-20b",
                "Сравни сценарии",
                list(AGENT_CATALOG.values()),
                scenario_id=1,
            )
        assert result.analytical
        assert [c["reasoning_effort"] for c in calls] == ["high", "medium"]
        assert all("response_format" not in c for c in calls)
        assert budget.reasoning_fallbacks == 1 and budget.model_calls == 2
    finally:
        await adapter.client.close()


@pytest.mark.parametrize(
    "message,recover,expected_calls",
    [
        (
            'unexpected tokens remaining in message header: Some("<|constrain|>analysis")',
            True,
            2,
        ),
        (
            'unexpected tokens remaining in message header: Some("<|constrain|>analysis")',
            False,
            2,
        ),
        ("unrelated server failure", False, 1),
    ],
)
async def test_harmony_header_failure_has_one_charged_medium_fallback(
    message, recover, expected_calls
):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1 or not recover:
            return httpx.Response(500, json={"error": {"message": message}})
        return httpx.Response(200, json=response("stop"))

    adapter = OpenAiCompatAdapter("http://test/v1")
    adapter.client = AsyncOpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    budget = RunBudget()
    try:
        with budget_scope(budget):
            if recover:
                await adapter.chat(
                    "gpt-oss-20b",
                    [{"role": "user", "content": "synthetic"}],
                    reasoning_effort="high",
                )
            else:
                with pytest.raises(Exception):
                    await adapter.chat(
                        "gpt-oss-20b",
                        [{"role": "user", "content": "synthetic"}],
                        reasoning_effort="high",
                    )
        assert len(calls) == budget.model_calls == expected_calls
        assert budget.reasoning_fallbacks == expected_calls - 1
        assert [c["reasoning_effort"] for c in calls] == ["high", "medium"][
            :expected_calls
        ]
    finally:
        await adapter.client.close()
