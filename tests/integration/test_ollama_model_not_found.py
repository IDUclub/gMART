"""Integration: a missing Ollama model surfaces as ``ModelNotFound`` against a live Ollama.

Pins the real behaviour behind the production incident — requesting a model that is not
pulled returns HTTP 404 from Ollama, which the service maps to ``ModelNotFound`` (carrying
the list of models that *are* available). Skips automatically when Ollama is unreachable.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.agents.common.exceptions.ollama_exceptions import ModelNotFound
from src.agents.services.base_llm_service import BaseLlmService

pytestmark = pytest.mark.integration

_MISSING_MODEL = "definitely-not-a-real-model:0b"


async def test_generate_chat_title_missing_model_maps_to_model_not_found(
    require_ollama,
):
    svc = BaseLlmService(require_ollama, Mock(), Mock())

    with pytest.raises(ModelNotFound) as ei:
        await svc.generate_chat_title(_MISSING_MODEL, "тестовый запрос", "", [])

    assert ei.value.model == _MISSING_MODEL
    assert isinstance(ei.value.available_models, list)
