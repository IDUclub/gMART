"""Unit tests for the model-not-found mapping in ``generate_chat_title``.

When the model server answers 404 ("model '<name>' not found") the adapter's
``LlmResponseError`` must be mapped to the REST-facing ``ModelNotFound`` (404 +
available models) instead of escaping and crashing the pipeline. Other backend
errors must propagate unchanged. The mapping is backend-independent: both the
Ollama and the OpenAI-compatible adapter raise ``LlmResponseError``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.agents.common.exceptions.ollama_exceptions import ModelNotFound
from src.agents.model_clients.llm_base import LlmResponseError
from src.agents.services.base_llm_service import BaseLlmService


def _service() -> BaseLlmService:
    # chat_storage / urban_api clients are unused by generate_chat_title.
    return BaseLlmService("http://ollama", Mock(), Mock())


class _GenRaising:
    """Fake adapter whose ``generate`` raises a programmed backend error."""

    def __init__(self, status_code: int) -> None:
        self._status_code = status_code

    async def generate(self, model=None, prompt=None, stream=False, **kwargs):
        raise LlmResponseError(f"model '{model}' not found", self._status_code)


class _GenOk:
    def __init__(self, response: str) -> None:
        self._response = response

    async def generate(self, model=None, prompt=None, stream=False, **kwargs):
        return SimpleNamespace(response=self._response)


async def test_generate_chat_title_maps_404_to_model_not_found():
    svc = _service()
    svc.llm_client = _GenRaising(404)
    svc.get_models = AsyncMock(return_value=["llama3:8b", "qwen2:7b"])

    with pytest.raises(ModelNotFound) as ei:
        await svc.generate_chat_title("gpt-oss:20b", "запрос", "", [])

    assert ei.value.model == "gpt-oss:20b"
    assert ei.value.available_models == ["llama3:8b", "qwen2:7b"]
    assert ei.value.status_code == 404
    svc.get_models.assert_awaited_once()


async def test_generate_chat_title_reraises_non_404_backend_error():
    svc = _service()
    svc.llm_client = _GenRaising(500)
    svc.get_models = AsyncMock()

    with pytest.raises(LlmResponseError):
        await svc.generate_chat_title("some-model", "запрос", "", [])

    # The available-models lookup is only done on the 404 mapping path.
    svc.get_models.assert_not_awaited()


async def test_generate_chat_title_returns_unique_title_on_success():
    svc = _service()
    svc.llm_client = _GenOk("Анализ озеленения")

    title = await svc.generate_chat_title("m", "запрос", "", ["Другой чат"])

    assert title == "Анализ озеленения"


async def test_empty_title_falls_back_to_the_query():
    """A reasoning model can burn the whole budget on its trace and answer
    nothing. An empty title is dropped from the ChatStorage payload and comes
    back as None, which then fails the chat_created SSE event and kills the
    stream — so the title must never be empty."""

    svc = _service()
    svc.llm_client = _GenOk("")

    title = await svc.generate_chat_title(
        "m", "  Сколько   домов в зоне ограничений?  ", "", []
    )

    assert title == "Сколько домов в зоне ограничений?"


async def test_fallback_title_avoids_collisions():
    svc = _service()
    svc.llm_client = _GenOk("   ")

    title = await svc.generate_chat_title("m", "Запрос", "", ["Запрос", "Запрос (2)"])

    assert title == "Запрос (3)"


async def test_chat_storage_items_payload_is_understood():
    """``/chats/titles`` answers ``{"items": [...]}``; treating that dict as the
    list of names compared titles against its keys and dumped every stored title
    into the prompt."""

    svc = _service()
    captured = {}

    class _Capture:
        async def generate(self, model=None, prompt=None, stream=False, **kwargs):
            captured["prompt"] = prompt
            return SimpleNamespace(response="Занятое имя")

    svc.llm_client = _Capture()

    title = await svc.generate_chat_title(
        "m", "Свободный запрос", "", {"items": ["Занятое имя"]}, max_retries=0
    )

    assert "Занятое имя" in captured["prompt"]
    assert title == "Свободный запрос"      # the collision was actually detected


async def test_prompt_carries_only_a_bounded_sample_of_existing_titles():
    """A benchmark creates a chat per query, so the title list grows without
    bound and the whole of it does not fit the model's context window."""

    svc = _service()
    captured = {}

    class _Capture:
        async def generate(self, model=None, prompt=None, stream=False, **kwargs):
            captured["prompt"] = prompt
            return SimpleNamespace(response="Название")

    svc.llm_client = _Capture()
    names = [f"Чат {i}" for i in range(5000)]

    await svc.generate_chat_title("m", "запрос", "", names)

    assert "Чат 4999" in captured["prompt"]
    assert "Чат 0\"" not in captured["prompt"]
    assert len(captured["prompt"]) < 10_000


async def test_title_generation_does_not_recurse_forever_on_duplicates():
    svc = _service()
    svc.llm_client = _GenOk("Занятое имя")

    title = await svc.generate_chat_title(
        "m", "Свободный запрос", "", ["Занятое имя"], max_retries=2
    )

    assert title == "Свободный запрос"
