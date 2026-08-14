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
