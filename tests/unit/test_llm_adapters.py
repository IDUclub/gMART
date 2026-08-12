"""Tests for the backend-neutral LLM adapters.

The agents read responses both as attributes and as mappings and rely on
Ollama-specific arguments (``format`` for a JSON schema, ``options`` for
sampling), so the OpenAI-compatible adapter is only a drop-in if it translates
those and returns the same shape. That is what these tests pin down.
"""

from __future__ import annotations

import pytest

from src.agents.model_clients.factory import build_llm_adapter, resolve_backend
from src.agents.model_clients.llm_base import (
    LlmChatResponse,
    LlmMessage,
    LlmResponseError,
)
from src.agents.model_clients.ollama_adapter import OllamaAdapter
from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class _Delta:
    def __init__(self, content=None, role="assistant"):
        self.content = content
        self.role = role


class _Choice:
    def __init__(self, message=None, delta=None, finish_reason=None):
        self.message = message
        self.delta = delta
        self.finish_reason = finish_reason


class _Completion:
    def __init__(self, choices, model="m"):
        self.choices = choices
        self.model = model


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c

        return gen()


class FakeCompletions:
    """Records the request and replays a canned answer."""

    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _adapter_with(result) -> tuple[OpenAiCompatAdapter, FakeCompletions]:
    adapter = OpenAiCompatAdapter(base_url="http://vllm:8000/v1", api_key="k")
    completions = FakeCompletions(result)
    adapter.client.chat.completions = completions  # type: ignore[assignment]
    return adapter, completions


# --------------------------------------------------------------------------- #
# response shape
# --------------------------------------------------------------------------- #
def test_response_supports_attribute_and_key_access():
    response = LlmChatResponse(message=LlmMessage(content="привет"), done=True)
    assert response.message.content == "привет"
    assert response["message"]["content"] == "привет"
    assert response.get("done") is True


@pytest.mark.asyncio
async def test_openai_chat_returns_ollama_shaped_response():
    completion = _Completion([_Choice(message=_Delta("готово"), finish_reason="stop")])
    adapter, _ = _adapter_with(completion)

    response = await adapter.chat("m", [{"role": "user", "content": "?"}])

    assert response.message.content == "готово"
    assert response["message"]["content"] == "готово"
    assert response.done is True


@pytest.mark.asyncio
async def test_openai_streaming_yields_chunks_and_marks_the_last_one():
    chunks = [
        _Completion([_Choice(delta=_Delta("часть 1"))]),
        _Completion([_Choice(delta=_Delta("часть 2"), finish_reason="stop")]),
    ]
    adapter, _ = _adapter_with(_FakeStream(chunks))

    parts = [p async for p in await adapter.chat("m", [], stream=True)]

    assert [p.message.content for p in parts] == ["часть 1", "часть 2"]
    assert [p.done for p in parts] == [False, True]


# --------------------------------------------------------------------------- #
# argument translation
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_json_schema_becomes_response_format():
    schema = {"title": "RestrictionPlan", "type": "object", "properties": {}}
    adapter, completions = _adapter_with(
        _Completion([_Choice(message=_Delta("{}"), finish_reason="stop")])
    )

    await adapter.chat("m", [], format=schema)

    sent = completions.calls[0]["response_format"]
    assert sent["type"] == "json_schema"
    assert sent["json_schema"]["name"] == "RestrictionPlan"
    assert sent["json_schema"]["schema"] is schema


@pytest.mark.asyncio
async def test_sampling_options_are_translated_and_num_ctx_dropped():
    adapter, completions = _adapter_with(
        _Completion([_Choice(message=_Delta("ok"), finish_reason="stop")])
    )

    await adapter.chat(
        "m",
        [],
        think=False,
        options={"temperature": 0, "num_predict": 4096, "num_ctx": 16384},
    )

    sent = completions.calls[0]
    assert sent["temperature"] == 0
    assert sent["max_tokens"] == 4096
    assert "num_ctx" not in sent and "think" not in sent


@pytest.mark.asyncio
async def test_generate_maps_to_a_single_user_message():
    adapter, completions = _adapter_with(
        _Completion([_Choice(message=_Delta("Название"), finish_reason="stop")])
    )

    result = await adapter.generate("m", "придумай название")

    assert result.response == "Название"
    assert completions.calls[0]["messages"] == [
        {"role": "user", "content": "придумай название"}
    ]


@pytest.mark.asyncio
async def test_model_listing_has_the_shape_get_models_expects():
    class _Models:
        async def list(self):
            class R:
                data = [type("M", (), {"id": "gpt-oss:20b"})()]

            return R()

    adapter, _ = _adapter_with(None)
    adapter.client.models = _Models()  # type: ignore[assignment]

    listed = await adapter.list()
    running = await adapter.ps()

    assert [m["model"] for m in listed["models"]] == ["gpt-oss:20b"]
    assert running == listed


@pytest.mark.asyncio
async def test_status_errors_become_llm_response_error():
    from openai import APIStatusError

    import httpx

    response = httpx.Response(
        404, request=httpx.Request("POST", "http://vllm:8000/v1/chat/completions")
    )

    class Failing:
        async def create(self, **kwargs):
            raise APIStatusError("no such model", response=response, body=None)

    adapter = OpenAiCompatAdapter(base_url="http://vllm:8000/v1")
    adapter.client.chat.completions = Failing()  # type: ignore[assignment]

    with pytest.raises(LlmResponseError) as exc:
        await adapter.chat("missing", [])
    assert exc.value.status_code == 404


# --------------------------------------------------------------------------- #
# backend selection
# --------------------------------------------------------------------------- #
def test_backend_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    assert resolve_backend() == "ollama"
    assert isinstance(build_llm_adapter("http://a.dgx:11434"), OllamaAdapter)


def test_openai_backend_is_selected_by_env(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://vllm:8000/v1")
    adapter = build_llm_adapter("http://a.dgx:11434")
    assert isinstance(adapter, OpenAiCompatAdapter)
    assert adapter.base_url == "http://vllm:8000/v1"


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "llamacpp")
    with pytest.raises(ValueError):
        build_llm_adapter("http://a.dgx:11434")


@pytest.mark.asyncio
async def test_ollama_adapter_translates_its_error_type():
    """Production path for the 404 -> ModelNotFound mapping: the service catches
    LlmResponseError, so the Ollama backend must stop leaking ollama.ResponseError."""

    from ollama import ResponseError

    class Failing:
        async def chat(self, **kwargs):
            raise ResponseError("model not found", 404)

    adapter = OllamaAdapter(host="http://a.dgx:11434")
    adapter.client = Failing()  # type: ignore[assignment]

    with pytest.raises(LlmResponseError) as exc:
        await adapter.chat("missing", [])
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_ollama_adapter_forwards_the_ollama_only_arguments():
    """think/format/options must reach Ollama untouched — that backend supports them."""

    class Recording:
        def __init__(self):
            self.call = None

        async def chat(self, **kwargs):
            self.call = kwargs
            return LlmChatResponse()

    adapter = OllamaAdapter(host="http://a.dgx:11434")
    recording = Recording()
    adapter.client = recording  # type: ignore[assignment]

    schema = {"title": "Plan", "type": "object"}
    await adapter.chat("m", [], think=False, format=schema, options={"num_ctx": 16384})

    assert recording.call["think"] is False
    assert recording.call["format"] is schema
    assert recording.call["options"] == {"num_ctx": 16384}
