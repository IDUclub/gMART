"""Exercise the real Harmony parser and SDK at the HTTP boundary."""

import json

import httpx
import pytest
from openai import AsyncOpenAI
from openai_harmony import Message, Role
from pydantic import BaseModel

from src.agents.model_clients.harmony_completion import encoding
from src.agents.model_clients.llm_base import LlmResponseError
from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter
from src.agents.runtime.budget import RunBudget, budget_scope
from src.agents.runtime.runner import run_structured


class Answer(BaseModel):
    value: int


@pytest.mark.parametrize(
    "mode", ["valid", "incomplete", "invalid_schema", "unregistered", "missing_tokens"]
)
async def test_harmony_transport_validation_and_accounting(monkeypatch, mode):
    monkeypatch.setenv("OPENAI_STRUCTURED_TRANSPORT", "harmony_completion")
    enc = encoding()
    calls = []

    def handle(request):
        assert request.url.path == "/v1/completions"
        payload = json.loads(request.content)
        calls.append(payload)
        assert all(isinstance(t, int) for t in payload["prompt"])
        assert payload["return_token_ids"] is True
        text = (
            '{"value":"bad"}'
            if mode == "invalid_schema" and len(calls) == 1
            else '{"value":42}'
        )
        recipient = (
            "functions.unregistered"
            if mode == "unregistered"
            else "functions.emit_structured_response"
        )
        message = (
            Message.from_role_and_content(Role.ASSISTANT, text)
            .with_channel("commentary")
            .with_recipient(recipient)
        )
        tokens = enc.render(message)[2:-1] + [200012]
        choice = {
            "index": 0,
            "text": "",
            "logprobs": None,
            "finish_reason": (
                "length" if mode == "incomplete" and len(calls) == 1 else "stop"
            ),
        }
        if mode != "missing_tokens":
            choice["token_ids"] = tokens
        return httpx.Response(
            200,
            json={
                "id": "cmpl_test",
                "object": "text_completion",
                "created": 0,
                "model": "gpt-oss-20b",
                "choices": [choice],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 30,
                    "total_tokens": 42,
                },
            },
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
            if mode in {"unregistered", "missing_tokens"}:
                with pytest.raises(LlmResponseError):
                    await adapter.chat(
                        "gpt-oss-20b",
                        [{"role": "user", "content": "test"}],
                        format=Answer.model_json_schema(),
                    )
            else:
                result = await run_structured(
                    adapter,
                    "gpt-oss-20b",
                    [{"role": "user", "content": "test"}],
                    Answer,
                    agent_name="test",
                    reasoning_effort="low",
                    options={"num_predict": 2048},
                )
                assert result.value == 42
                assert budget.tokens == 42 * len(calls)
        assert budget.model_calls == len(calls)
    finally:
        await adapter.client.close()
