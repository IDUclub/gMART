"""Run the real SDK against fake inference/transports, including failure lifecycles."""

import asyncio
import json
from contextlib import aclosing
from unittest.mock import AsyncMock

import httpx
import pytest
from ollama import ResponseError
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.agents.common.exceptions.token_exceptions import TokenExpiredError
from src.agents.model_clients.llm_base import (
    LlmChatResponse,
    LlmMessage,
    LlmResponseError,
)
from src.agents.model_clients.ollama_adapter import OllamaAdapter
from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter
from src.agents.runtime.runner import (
    IncompleteStructuredOutput,
    run_completion,
    run_structured,
)
from src.agents.runtime.tools import execute_planned, stream_planned


class Plan(BaseModel):
    distance: int = Field(ge=1)


def response(text, reason="stop"):
    return LlmChatResponse(message=LlmMessage(content=text), done_reason=reason)


@pytest.mark.asyncio
async def test_sdk_uses_configured_endpoint_model_and_provider_options():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "local-completion",
                "created": 1,
                "model": "gpt-oss-local",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"distance":50}'},
                    }
                ],
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    backend = OpenAiCompatAdapter("http://local-inference/v1", api_key="local-key")
    await backend.client.close()
    backend.client = AsyncOpenAI(
        base_url="http://local-inference/v1",
        api_key="local-key",
        http_client=http,
    )
    try:
        result = await run_structured(
            backend,
            "gpt-oss-local",
            [{"role": "user", "content": "50 метров"}],
            Plan,
            agent_name="restriction.plan",
            think=False,
            options={"temperature": 0, "num_predict": 2048},
        )
    finally:
        await backend.client.close()
    assert result.distance == 50
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "http://local-inference/v1/chat/completions"
    body = json.loads(request.content)
    assert body["model"] == "gpt-oss-local"
    assert body["reasoning_effort"] == "low"
    assert body["max_tokens"] == 2048
    assert body["response_format"]["json_schema"]["schema"]["title"] == "Plan"


@pytest.mark.asyncio
async def test_sdk_keeps_native_ollama_schema_context_and_model():
    backend = OllamaAdapter("http://localhost:11434")
    backend.client.chat = AsyncMock(return_value=response('{"distance":12}'))
    result = await run_structured(
        backend,
        "native-model",
        [],
        Plan,
        agent_name="restriction.plan",
        think=False,
        options={"num_ctx": 16384, "num_predict": 4096},
    )
    assert result.distance == 12
    call = backend.client.chat.call_args.kwargs
    assert call["model"] == "native-model"
    assert call["think"] is False
    assert call["options"] == {"num_ctx": 16384, "num_predict": 4096}
    assert call["format"]["title"] == "Plan"


@pytest.mark.asyncio
async def test_native_ollama_failure_after_first_chunk_keeps_transport_error_type():
    closed = asyncio.Event()

    async def broken_stream():
        try:
            yield LlmChatResponse(message=LlmMessage(content="draft"), done=False)
            raise ResponseError("unavailable", 503)
        finally:
            closed.set()

    backend = OllamaAdapter("http://localhost:11434")
    backend.client.chat = AsyncMock(return_value=broken_stream())
    with pytest.raises(LlmResponseError) as caught:
        async for _ in await run_completion(
            backend, "native", [], agent_name="answer", stream=True
        ):
            pass
    assert caught.value.status_code == 503
    assert closed.is_set()


@pytest.mark.asyncio
async def test_openai_stream_failure_closes_transport_and_keeps_error_type():
    closed = asyncio.Event()

    class BrokenStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            from openai import APIConnectionError

            raise APIConnectionError(
                request=httpx.Request("POST", "http://local/v1/chat/completions")
            )

        async def close(self):
            closed.set()

    backend = OpenAiCompatAdapter("http://local/v1")
    backend.client.chat.completions.create = AsyncMock(return_value=BrokenStream())
    try:
        with pytest.raises(LlmResponseError):
            async for _ in await run_completion(
                backend, "local", [], agent_name="answer", stream=True
            ):
                pass
    finally:
        await backend.client.close()
    assert closed.is_set()


@pytest.mark.asyncio
async def test_structured_repair_keeps_original_context_and_validator_feedback():
    backend = AsyncMock()
    backend.chat.side_effect = [
        response('{"distance":0}'),
        response('```json\n{"distance":50}\n```'),
    ]
    messages = [
        {"role": "system", "content": "Правила"},
        {"role": "user", "content": "Запрос"},
    ]
    result = await run_structured(backend, "local", messages, Plan, agent_name="plan")
    assert result.distance == 50
    repair = backend.chat.call_args.kwargs["messages"]
    assert repair[:2] == messages
    assert repair[2] == {"role": "assistant", "content": '{"distance":0}'}
    assert "distance" in repair[-1]["content"]
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_empty_structured_output_is_bounded_and_can_drop_decoder_constraint():
    backend = AsyncMock()
    backend.chat.side_effect = [response(""), response('{"distance":12}')]
    result = await run_structured(
        backend,
        "local",
        [],
        Plan,
        agent_name="plan",
        retries=1,
        attempt_settings=lambda attempt, _: {"unconstrained": bool(attempt)},
    )
    assert result.distance == 12
    assert "format" in backend.chat.call_args_list[0].kwargs
    assert "format" not in backend.chat.call_args_list[1].kwargs
    assert "unconstrained" not in backend.chat.call_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["length", "max_tokens", "incomplete", "content_filter"]
)
async def test_valid_json_prefix_of_unfinished_output_is_never_accepted(reason):
    backend = AsyncMock()
    backend.chat.return_value = response('{"distance":50}', reason)
    with pytest.raises(IncompleteStructuredOutput):
        await run_structured(backend, "local", [], Plan, agent_name="plan", retries=0)
    assert backend.chat.await_count == 1


@pytest.mark.asyncio
async def test_sdk_preserves_model_not_found_without_repair_requests():
    error = LlmResponseError("model unavailable", 404)
    backend = AsyncMock()
    backend.chat.side_effect = error
    with pytest.raises(LlmResponseError) as caught:
        await run_structured(backend, "missing", [], Plan, agent_name="plan")
    assert caught.value is error
    assert backend.chat.await_count == 1


@pytest.mark.asyncio
async def test_parallel_runs_do_not_share_model_settings_or_answers():
    class Backend:
        async def chat(self, **call):
            await asyncio.sleep(0)
            return response(call["model"] + str(call["options"]["temperature"]))

    backend = Backend()
    results = await asyncio.gather(
        *[
            run_completion(
                backend,
                model,
                [],
                agent_name="answer",
                options={"temperature": temperature},
            )
            for model, temperature in [("first", 0), ("second", 1)]
        ]
    )
    assert [item.message.content for item in results] == ["first0", "second1"]


@pytest.mark.asyncio
async def test_sdk_stream_keeps_reasoning_and_marks_missing_terminal_as_incomplete():
    class Backend:
        async def chat(self, **call):
            async def chunks():
                yield LlmChatResponse(
                    message=LlmMessage(content="draft", thinking="trace"), done=False
                )

            return chunks()

    chunks = [
        item
        async for item in await run_completion(
            Backend(), "local", [], agent_name="answer", stream=True
        )
    ]
    assert chunks[0].message.content == "draft"
    assert chunks[0].message.thinking == "trace"
    assert chunks[-1].done_reason == "incomplete"


@pytest.mark.asyncio
async def test_closing_sdk_stream_closes_inference_generator():
    closed = asyncio.Event()

    class Backend:
        async def chat(self, **call):
            async def chunks():
                try:
                    yield LlmChatResponse(
                        message=LlmMessage(content="first"), done=False
                    )
                    await asyncio.Event().wait()
                finally:
                    closed.set()

            return chunks()

    async with aclosing(
        await run_completion(Backend(), "local", [], agent_name="answer", stream=True)
    ) as stream:
        assert (await anext(stream)).message.content == "first"
    await asyncio.wait_for(closed.wait(), timeout=2)


@pytest.mark.asyncio
async def test_planned_tool_returns_native_result_and_preserves_exception_identity():
    result = {"type": "FeatureCollection", "features": []}
    operation = AsyncMock(return_value=result)
    assert await execute_planned("CreateBuffers", operation) is result
    assert operation.await_count == 1
    expired = TokenExpiredError("expired")
    operation = AsyncMock(side_effect=expired)
    with pytest.raises(TokenExpiredError) as caught:
        await execute_planned("CreateBuffers", operation)
    assert caught.value is expired
    assert operation.await_count == 1


@pytest.mark.asyncio
async def test_stopping_delegation_cannot_start_work_after_clarification():
    calls = []
    closed = asyncio.Event()

    async def specialist():
        try:
            yield {"type": "clarification"}
            calls.append("must not execute")
        finally:
            closed.set()

    async with aclosing(
        stream_planned("orchestrator.documents", specialist())
    ) as events:
        assert await anext(events) == {"type": "clarification"}
        await asyncio.sleep(0)
    assert closed.is_set()
    assert calls == []


@pytest.mark.asyncio
async def test_delegated_failure_is_not_converted_to_a_successful_answer():
    error = LlmResponseError("inference failed", 502)

    async def specialist():
        yield {"type": "status"}
        raise error

    async with aclosing(
        stream_planned("orchestrator.documents", specialist())
    ) as events:
        assert await anext(events) == {"type": "status"}
        with pytest.raises(LlmResponseError) as caught:
            await anext(events)
    assert caught.value is error
