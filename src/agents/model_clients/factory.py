"""Chooses the LLM backend for the whole service.

``LLM_BACKEND`` selects it and defaults to ``openai``, which points the agents at
any OpenAI-compatible server (vLLM, Ollama's own ``/v1``, and friends) through
``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``. ``ollama`` keeps the native client.

A deployment that sets neither variable still starts: the base URL falls back to
``OLLAMA_API_URL`` with ``/v1`` appended, so it talks to the same Ollama as before
over the OpenAI protocol. One thing does not survive that switch — ``num_ctx`` is
not an OpenAI parameter, so the context length becomes the server's own default
(4096 on Ollama) and must be set there instead, via ``OLLAMA_CONTEXT_LENGTH`` or
vLLM's ``--max-model-len``.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from loguru import logger

from src.agents.model_clients.llm_base import BaseLlmAdapter
from src.agents.model_clients.ollama_adapter import OllamaAdapter
from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter

OLLAMA = "ollama"
OPENAI = "openai"


def resolve_backend(backend: str | None = None) -> str:
    value = (backend or os.getenv("LLM_BACKEND") or OPENAI).strip().lower()
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
    url = _with_api_path(url)
    key = api_key or os.getenv("OPENAI_API_KEY")
    logger.info(f"LLM backend: OpenAI-compatible at {url}")
    return OpenAiCompatAdapter(base_url=url, api_key=key, timeout=timeout)


def _with_api_path(url: str) -> str:
    """Append ``/v1`` to a bare origin.

    The fallback base URL is OLLAMA_API_URL (``http://a.dgx:11434``), and a vLLM
    address is just as easily written without the suffix; either would make every
    request 404 on a path that does not exist. A URL that already carries a path
    is left alone, since deployments behind a proxy mount the API elsewhere.
    """

    trimmed = url.rstrip("/")
    if urlparse(trimmed).path:
        return trimmed
    return f"{trimmed}/v1"
