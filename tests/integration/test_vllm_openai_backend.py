"""Integration: the OpenAI backend against a live vLLM (or any compatible server).

Unit tests pin the translation against fakes; these pin the parts only a real server
decides — what its responses actually look like and which requests it accepts. Each
one covers a difference that was found the hard way against vLLM 0.27:

* the reasoning trace arrives in ``reasoning``, not the ``reasoning_content`` older
  versions used, and is empty when the completion is cut off mid-thought;
* ``content`` is ``null`` rather than ``""`` on such a truncated completion;
* ``think=False`` has to actually reach the server, or the trace is generated
  anyway and silently costs the whole token budget on a reasoning model;
* the planner schemas have to survive vLLM's grammar compiler, not just Pydantic.

Skips automatically unless ``VLLM_BASE_URL`` points at a live server. Budgets are
small so the suite stays usable against a CPU deployment.

    VLLM_BASE_URL=http://localhost:8800/v1 uv run pytest tests/integration/test_vllm_openai_backend.py
"""

from __future__ import annotations

import json

import pytest

from src.agents.model_clients.factory import build_llm_adapter
from src.agents.model_clients.llm_base import LlmResponseError
from src.agents.services.service_entities.orchestrator_plan import OrchestratorPlan
from src.agents.services.service_entities.restriction_plan import RestrictionPlan

pytestmark = pytest.mark.integration


@pytest.fixture
def adapter(require_openai_backend):
    """The adapter as a deployment gets it — no think-mode configuration."""

    url, model = require_openai_backend
    return build_llm_adapter(host="", backend="openai", base_url=url), model


async def test_bare_origin_is_normalised_to_the_api_path(require_openai_backend):
    """A URL without /v1 would 404 every request; OPENAI_BASE_URL falls back to
    OLLAMA_API_URL, which is always a bare origin."""

    url, model = require_openai_backend
    origin = url[: -len("/v1")] if url.endswith("/v1") else url

    client = build_llm_adapter(host="", backend="openai", base_url=origin)

    assert client.base_url.endswith("/v1")
    assert model in [m["model"] for m in (await client.list())["models"]]


async def test_chat_returns_an_ollama_shaped_response(adapter):
    client, model = adapter

    response = await client.chat(
        model,
        [{"role": "user", "content": "Столица Франции? Одним словом."}],
        think=False,
        options={"temperature": 0, "num_predict": 64},
    )

    assert response.message.content.strip()
    assert response["message"]["content"] == response.message.content
    assert response.done is True


async def test_streaming_ends_on_a_done_chunk(adapter):
    """The call sites loop until part.done — a stream that never marks the end
    would hang them."""

    client, model = adapter

    parts = [
        part
        async for part in await client.chat(
            model,
            [{"role": "user", "content": "Назови два города России."}],
            think=False,
            stream=True,
            options={"temperature": 0, "num_predict": 64},
        )
    ]

    assert len(parts) > 1
    assert parts[-1].done is True
    assert "".join(p.message.content for p in parts).strip()


async def test_reasoning_trace_is_surfaced_as_thinking(adapter):
    """vLLM 0.27 renamed reasoning_content to reasoning; the adapter reads both."""

    client, model = adapter

    response = await client.chat(
        model,
        [{"role": "user", "content": "Сколько будет 17*3? Кратко."}],
        options={"temperature": 0, "num_predict": 700},
    )

    if response.done_reason == "length":
        pytest.skip("the budget ran out before the trace closed")
    assert (response.message.thinking or "").strip()


async def test_think_false_suppresses_the_trace(adapter):
    """Out of the box, on whichever server is configured: reasoning_effort="none"
    is honoured by both vLLM and Ollama's /v1. Without it the trace is generated
    anyway and eats num_predict, which is how a planner ends up with empty content
    and finish_reason=length."""

    client, model = adapter
    messages = [{"role": "user", "content": "Сколько будет 17*3? Кратко."}]

    response = await client.chat(
        model, messages, think=False, options={"temperature": 0, "num_predict": 700}
    )

    assert response.message.content.strip()
    assert not (response.message.thinking or "").strip()


async def test_truncated_completion_has_a_string_content(adapter):
    """vLLM answers with content=null here; the call sites do .strip() on it."""

    client, model = adapter

    response = await client.chat(
        model,
        [{"role": "user", "content": "Расскажи подробно про Санкт-Петербург."}],
        options={"temperature": 0, "num_predict": 16},
    )

    assert response.done_reason == "length"
    assert isinstance(response.message.content, str)


@pytest.mark.parametrize("plan_cls", [RestrictionPlan, OrchestratorPlan])
async def test_planner_schemas_survive_guided_decoding(adapter, plan_cls):
    """The grammar compiler must accept the schema and the output must parse.

    Only JSON validity is asserted: cross-field rules (a needs_clarification plan
    carrying steps, say) are Pydantic validators that no JSON Schema can express,
    and the planners retry on those.
    """

    client, model = adapter

    response = await client.chat(
        model=model,
        think=False,
        format=plan_cls.model_json_schema(),
        options={"temperature": 0, "num_predict": 512, "num_ctx": 16384},
        messages=[
            {"role": "system", "content": "Верни план в JSON строго по схеме."},
            {
                "role": "user",
                "content": "Найди ограничения застройки рядом со школами.",
            },
        ],
    )

    assert isinstance(json.loads(response.message.content), dict)


async def test_unknown_model_is_a_404(adapter):
    """generate_chat_title maps this onto ModelNotFound."""

    client, _ = adapter

    with pytest.raises(LlmResponseError) as exc:
        await client.chat("definitely-not-served", [{"role": "user", "content": "hi"}])

    assert exc.value.status_code == 404


async def test_context_overflow_never_leaks_a_raw_sdk_error(adapter):
    """num_ctx is dropped, so the server's own limit governs — and the two engines
    disagree about it: vLLM rejects the request (400 against --max-model-len) while
    Ollama truncates and answers. Either is acceptable; what must not happen is an
    openai SDK exception reaching the services, which catch LlmResponseError only.
    """

    client, model = adapter

    try:
        response = await client.chat(
            model,
            [{"role": "user", "content": "слово " * 20000}],
            think=False,
            options={"temperature": 0, "num_predict": 16},
        )
    except LlmResponseError as exc:
        assert exc.status_code == 400
    else:
        assert isinstance(response.message.content, str)
