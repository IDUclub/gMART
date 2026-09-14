"""Verify the optional Responses envelope with the real SDK and HTTP boundary."""

import json

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.agents.model_clients.llm_base import LlmResponseError
from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter
from src.agents.runtime.budget import RunBudget, budget_scope
from src.agents.runtime.runner import run_structured


class Answer(BaseModel):
    value: int


def response(outputs, status="completed"):
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "model": "gpt-oss-20b",
        "status": status,
        "output": outputs,
        "usage": {"input_tokens": 12, "output_tokens": 30, "total_tokens": 42},
    }


def envelope(arguments='{"value":42}'):
    return {
        "type": "function_call",
        "name": "emit_structured_response",
        "arguments": arguments,
        "call_id": "call_test",
        "id": "fc_test",
    }


@pytest.mark.parametrize(
    "first", [None, "incomplete", "invalid_schema", "unexpected_action"]
)
async def test_responses_envelope_preserves_sdk_validation_and_budget(
    monkeypatch, first
):
    monkeypatch.setenv("OPENAI_STRUCTURED_TRANSPORT", "responses_function")
    calls = []

    def handle(request):
        assert request.url.path == "/v1/responses"
        payload = json.loads(request.content)
        calls.append(payload)
        assert payload["store"] is False
        assert [tool["type"] for tool in payload["tools"]] == ["function"]
        assert (
            payload["tools"][0]["parameters"]["properties"]["value"]["type"]
            == "integer"
        )
        assert payload["reasoning"]["effort"] == "low"
        if len(calls) == 1 and first:
            if first == "unexpected_action":
                return httpx.Response(
                    200,
                    json=response([dict(envelope(), name="unregistered_domain_tool")]),
                )
            return httpx.Response(
                200,
                json=response(
                    [envelope('{"value":"bad"}')],
                    "incomplete" if first == "incomplete" else "completed",
                ),
            )
        return httpx.Response(200, json=response([envelope()]))

    adapter = OpenAiCompatAdapter("http://test/v1")
    adapter.client = AsyncOpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    budget = RunBudget()
    try:
        with budget_scope(budget):
            answer = await run_structured(
                adapter,
                "gpt-oss-20b",
                [{"role": "user", "content": "test"}],
                Answer,
                agent_name="test.structured",
                reasoning_effort="low",
                options={"num_predict": 2048},
            )
        assert answer.value == 42
        assert budget.model_calls == len(calls) == (2 if first else 1)
        if first != "unexpected_action":
            assert budget.tokens == 42 * len(calls)
        else:
            assert budget.tokens >= 42 * len(calls)
        assert budget.tool_calls == 0
    finally:
        await adapter.client.close()


@pytest.mark.parametrize(
    "outputs",
    [
        [dict(envelope(), name="unregistered_domain_tool")],
        [envelope(), envelope()],
        [
            {
                "type": "mcp_call",
                "id": "mcp_test",
                "name": "run_func_generation",
                "arguments": "{}",
                "server_label": "external",
            }
        ],
    ],
)
async def test_unexpected_calls_are_rejected_without_execution(monkeypatch, outputs):
    monkeypatch.setenv("OPENAI_STRUCTURED_TRANSPORT", "responses_function")
    adapter = OpenAiCompatAdapter("http://test/v1")
    adapter.client = AsyncOpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=response(outputs))
            )
        ),
    )
    budget = RunBudget()
    try:
        with (
            budget_scope(budget),
            pytest.raises(LlmResponseError, match="unexpected action"),
        ):
            await adapter.chat(
                "gpt-oss-20b",
                [{"role": "user", "content": "test"}],
                format=Answer.model_json_schema(),
            )
        assert budget.model_calls == 2
        assert budget.tool_calls == 0
    finally:
        await adapter.client.close()
