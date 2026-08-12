"""Chooses the LLM backend for the whole service.

``LLM_BACKEND`` selects it and defaults to ``ollama``, so an existing deployment
that sets nothing keeps the previous behaviour exactly. ``openai`` points the
agents at any OpenAI-compatible server (vLLM and friends) through
``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import os

from loguru import logger

from src.agents.model_clients.llm_base import BaseLlmAdapter
from src.agents.model_clients.ollama_adapter import OllamaAdapter
from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter

OLLAMA = "ollama"
OPENAI = "openai"


def resolve_backend(backend: str | None = None) -> str:
    value = (backend or os.getenv("LLM_BACKEND") or OLLAMA).strip().lower()
    if value not in (OLLAMA, OPENAI):
        raise ValueError(
            f"unknown LLM_BACKEND={value!r}; expected {OLLAMA!r} or {OPENAI!r}"
        )
    return value


def build_llm_adapter(
    host: str,
    backend: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> BaseLlmAdapter:
    """Build the adapter for the configured backend.

    Args:
        host: the Ollama URL the services are constructed with; also the
            fallback base URL for the OpenAI backend when OPENAI_BASE_URL is unset.
        backend: overrides LLM_BACKEND (used by tests).
        base_url / api_key / timeout: override the OpenAI-backend environment.
    """

    chosen = resolve_backend(backend)
    if chosen == OLLAMA:
        return OllamaAdapter(host=host)

    url = base_url or os.getenv("OPENAI_BASE_URL") or host
    if not url:
        raise ValueError("LLM_BACKEND=openai requires OPENAI_BASE_URL")
    key = api_key or os.getenv("OPENAI_API_KEY")
    logger.info(f"LLM backend: OpenAI-compatible at {url}")
    return OpenAiCompatAdapter(base_url=url, api_key=key, timeout=timeout)
