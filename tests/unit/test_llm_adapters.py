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
    def __init__(self, content=None, role="assistant", **reasoning):
        self.content = content
        self.role = role
        # Servers disagree on the name: reasoning_content (vLLM) vs reasoning (Ollama).
        for name, value in reasoning.items():
            setattr(self, name, value)


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
    assert sent["reasoning_effort"] == "none"  # think=False, translated


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
    import httpx
    from openai import APIStatusError

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
def test_backend_defaults_to_openai(monkeypatch):
    """Unset means the OpenAI protocol, against OLLAMA_API_URL's /v1 by default —
    so a deployment that configures nothing still reaches its existing Ollama."""

    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert resolve_backend() == "openai"
    adapter = build_llm_adapter("http://a.dgx:11434")
    assert isinstance(adapter, OpenAiCompatAdapter)
    assert adapter.base_url == "http://a.dgx:11434/v1"


def test_ollama_backend_is_selected_by_env(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
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


# --------------------------------------------------------------------------- #
# vLLM conformance: the protocol details servers disagree on
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
@pytest.mark.asyncio
async def test_reasoning_trace_is_read_under_either_name(field):
    """vLLM names it reasoning_content, Ollama's /v1 names it reasoning."""

    message = _Delta("ответ", **{field: "рассуждение"})
    adapter, _ = _adapter_with(
        _Completion([_Choice(message=message, finish_reason="stop")])
    )

    response = await adapter.chat("m", [])

    assert response.message.content == "ответ"
    assert response.message.thinking == "рассуждение"


@pytest.mark.asyncio
async def test_streaming_skips_the_usage_only_trailer():
    """stream_options.include_usage adds a final chunk with no choices at all."""

    chunks = [
        _Completion([_Choice(delta=_Delta("текст"), finish_reason="stop")]),
        _Completion([]),  # usage-only trailer
    ]
    adapter, _ = _adapter_with(_FakeStream(chunks))

    parts = [p async for p in await adapter.chat("m", [], stream=True)]

    assert [p.message.content for p in parts] == ["текст"]
    assert [p.done for p in parts] == [True]


@pytest.mark.asyncio
async def test_streaming_always_ends_on_a_done_chunk():
    """The call sites loop until part.done, so a stream that never sends
    finish_reason must still terminate them — as Ollama's always does."""

    chunks = [_Completion([_Choice(delta=_Delta("часть"))])]
    adapter, _ = _adapter_with(_FakeStream(chunks))

    parts = [p async for p in await adapter.chat("m", [], stream=True)]

    assert [p.message.content for p in parts] == ["часть", ""]
    assert parts[-1].done is True


@pytest.mark.asyncio
async def test_think_false_becomes_reasoning_effort_by_default():
    """ "none" stays the default value, so existing deployments are unaffected."""

    adapter, completions = _adapter_with(
        _Completion([_Choice(message=_Delta("ok"), finish_reason="stop")])
    )

    await adapter.chat("m", [], think=False)

    assert completions.calls[0]["reasoning_effort"] == "none"
    assert "extra_body" not in completions.calls[0]


@pytest.mark.asyncio
async def test_think_effort_value_is_configurable():
    """The value must not be hardcoded: servers disagree on which ones they accept.

    vLLM's Harmony path (how gpt-oss is served) 400s on both "none" and "minimal" and takes
    only high/medium/low, so a contour has to be able to ask for "low".
    """

    adapter = OpenAiCompatAdapter(
        base_url="http://vllm:8000/v1", api_key="k", think_effort="low"
    )
    completions = FakeCompletions(
        _Completion([_Choice(message=_Delta("ok"), finish_reason="stop")])
    )
    adapter.client.chat.completions = completions  # type: ignore[assignment]

    await adapter.chat("m", [], think=False)

    assert completions.calls[0]["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_think_effort_reads_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_THINK_EFFORT", "medium")

    adapter, completions = _adapter_with(
        _Completion([_Choice(message=_Delta("ok"), finish_reason="stop")])
    )

    await adapter.chat("m", [], think=False)

    assert completions.calls[0]["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_an_explicit_caller_value_still_wins():
    """setdefault, not assignment — a call site may pin its own effort."""

    adapter = OpenAiCompatAdapter(
        base_url="http://vllm:8000/v1", api_key="k", think_effort="low"
    )
    completions = FakeCompletions(
        _Completion([_Choice(message=_Delta("ok"), finish_reason="stop")])
    )
    adapter.client.chat.completions = completions  # type: ignore[assignment]

    await adapter.chat("m", [], think=False, reasoning_effort="high")

    assert completions.calls[0]["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_think_true_sends_nothing_since_reasoning_is_the_default():
    adapter, completions = _adapter_with(
        _Completion([_Choice(message=_Delta("ok"), finish_reason="stop")])
    )

    await adapter.chat("m", [], think=True)

    assert "reasoning_effort" not in completions.calls[0]


@pytest.mark.asyncio
async def test_chat_template_mode_travels_as_a_chat_template_kwarg():
    adapter, completions = _adapter_with(
        _Completion([_Choice(message=_Delta("{}"), finish_reason="stop")])
    )
    adapter.think_mode = "chat_template"

    await adapter.chat("m", [], think=False)

    assert completions.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert "reasoning_effort" not in completions.calls[0]


@pytest.mark.asyncio
async def test_think_kwarg_does_not_clobber_a_caller_extra_body():
    adapter, completions = _adapter_with(
        _Completion([_Choice(message=_Delta("{}"), finish_reason="stop")])
    )
    adapter.think_mode = "chat_template"

    await adapter.chat("m", [], think=False, extra_body={"guided_regex": "[0-9]+"})

    sent = completions.calls[0]["extra_body"]
    assert sent["guided_regex"] == "[0-9]+"
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_think_mode_off_sends_nothing():
    """For a strict endpoint that rejects both spellings."""

    adapter, completions = _adapter_with(
        _Completion([_Choice(message=_Delta("ok"), finish_reason="stop")])
    )
    adapter.think_mode = "off"

    await adapter.chat("m", [], think=False)

    sent = completions.calls[0]
    assert "reasoning_effort" not in sent
    assert "extra_body" not in sent and "think" not in sent


def test_think_mode_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_THINK_MODE", "chat_template")
    monkeypatch.setenv("OPENAI_THINK_CHAT_TEMPLATE_KWARG", "thinking")

    adapter = OpenAiCompatAdapter(base_url="http://vllm:8000/v1")

    assert adapter.think_mode == "chat_template"
    assert adapter.think_kwarg == "thinking"


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("http://vllm:8000", "http://vllm:8000/v1"),
        ("http://vllm:8000/", "http://vllm:8000/v1"),
        ("http://a.dgx:11434", "http://a.dgx:11434/v1"),
        ("http://vllm:8000/v1", "http://vllm:8000/v1"),
        ("https://gw.example.ru/llm/v1", "https://gw.example.ru/llm/v1"),
    ],
)
def test_base_url_gets_the_api_path_when_it_is_a_bare_origin(
    monkeypatch, configured, expected
):
    """A bare origin would 404 every request; a mounted path must be preserved."""

    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", configured)

    assert build_llm_adapter("http://a.dgx:11434").base_url == expected


def test_openai_backend_falls_back_to_the_ollama_url_with_the_api_path(monkeypatch):
    """OPENAI_BASE_URL unset: the fallback is OLLAMA_API_URL, which is an origin."""

    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert build_llm_adapter("http://a.dgx:11434").base_url == "http://a.dgx:11434/v1"


@pytest.mark.asyncio
async def test_budget_exhausted_by_reasoning_is_logged():
    """An empty answer must not be silent: the cause is the reasoning budget,
    and the cure is more max_tokens rather than the caller's JSON retry."""

    from loguru import logger

    message = _Delta("", reasoning_content="очень длинное рассуждение")
    adapter, _ = _adapter_with(
        _Completion([_Choice(message=message, finish_reason="length")])
    )

    warnings: list[str] = []
    sink = logger.add(lambda record: warnings.append(str(record)), level="WARNING")
    try:
        response = await adapter.chat("m", [])
    finally:
        logger.remove(sink)

    assert response.message.content == ""
    assert response.done_reason == "length"
    assert any("finish_reason=length" in w for w in warnings), warnings


@pytest.mark.asyncio
async def test_budget_exhausted_is_logged_even_without_a_trace():
    """vLLM's reasoning parser only fills the field once it sees the closing tag,
    so a completion truncated mid-thought carries neither content nor trace."""

    from loguru import logger

    adapter, _ = _adapter_with(
        _Completion([_Choice(message=_Delta(None), finish_reason="length")])
    )

    warnings: list[str] = []
    sink = logger.add(lambda record: warnings.append(str(record)), level="WARNING")
    try:
        response = await adapter.chat("m", [])
    finally:
        logger.remove(sink)

    assert response.message.content == ""
    assert any("finish_reason=length" in w for w in warnings), warnings
