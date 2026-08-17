"""The default model, taken from whatever the connected provider actually serves.

A hardcoded default is wrong the moment the backend changes: an Ollama-style id like
``gpt-oss:20b`` does not exist on a vLLM serving ``gpt-oss-20b``, and every agent then fails
with a 404 that names a model nobody configured. So the default is *resolved* instead, from the
provider's own model list (``/v1/models`` on the OpenAI backend, ``/api/tags`` on Ollama — both
already behind :meth:`BaseLlmAdapter.list`).

Selection order:

1. ``DEFAULT_LLM_MODEL``, when set — an explicit operator choice always wins. It is used even if
   the provider does not list it, so a server with a hidden or aliased model id still works; a
   wrong value surfaces as the provider's own 404 rather than being silently replaced.
2. the first served id containing ``DEFAULT_LLM_MODEL_HINT`` (default ``gpt-oss``), which matches
   ``gpt-oss-20b``, ``gpt-oss:20b`` and ``openai/gpt-oss-20b`` alike;
3. the first served id, so a provider offering something else entirely still works.

Resolution is lazy and cached process-wide: the first request that omits a model pays one list
call, later ones pay nothing. Lazy rather than at start-up on purpose — the agents must come up
even when the inference server is still loading, which is the normal case under compose where
start order is not guaranteed. The cache expires so a provider that swaps models is picked up
without a restart, and a failed lookup is never cached.
"""

from __future__ import annotations

import asyncio
import os
import time

from loguru import logger

from src.agents.model_clients.llm_base import BaseLlmAdapter

DEFAULT_MODEL_ENV = "DEFAULT_LLM_MODEL"
DEFAULT_MODEL_HINT_ENV = "DEFAULT_LLM_MODEL_HINT"
FALLBACK_HINT = "gpt-oss"
CACHE_TTL_SECONDS = 300.0

_lock = asyncio.Lock()
_cached: str | None = None
_cached_at: float = 0.0


class NoModelsAvailable(RuntimeError):
    """The provider answered, but served no models to choose a default from."""


def invalidate_default_model() -> None:
    """Drop the cached default. Called by tests and after a model-not-found error."""

    global _cached, _cached_at
    _cached = None
    _cached_at = 0.0


def _pick(models: list[str], hint: str) -> str:
    lowered = hint.strip().lower()
    if lowered:
        for name in models:
            if lowered in name.lower():
                return name
    return models[0]


async def resolve_default_model(adapter: BaseLlmAdapter) -> str:
    """Return the model to use when the caller did not name one.

    Raises:
        NoModelsAvailable: the provider serves nothing, so no default exists. Raised rather
            than falling back to a guess — a guessed id would fail later with a confusing 404
            instead of pointing at the real problem.
    """

    global _cached, _cached_at

    explicit = (os.getenv(DEFAULT_MODEL_ENV) or "").strip()
    if explicit:
        return explicit

    now = time.monotonic()
    if _cached is not None and now - _cached_at < CACHE_TTL_SECONDS:
        return _cached

    async with _lock:
        # Another coroutine may have filled the cache while this one waited for the lock.
        now = time.monotonic()
        if _cached is not None and now - _cached_at < CACHE_TTL_SECONDS:
            return _cached

        listed = await adapter.list()
        models = [
            str(entry["model"])
            for entry in (listed or {}).get("models", [])
            if entry.get("model")
        ]
        if not models:
            raise NoModelsAvailable(
                "the LLM provider serves no models, so there is no default to fall back on; "
                f"name one per request or set {DEFAULT_MODEL_ENV}"
            )

        hint = os.getenv(DEFAULT_MODEL_HINT_ENV, FALLBACK_HINT)
        chosen = _pick(models, hint)
        _cached, _cached_at = chosen, time.monotonic()
        logger.info(f"default LLM model resolved from the provider: {chosen}")
        return chosen
