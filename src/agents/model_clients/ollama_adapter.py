"""Ollama backend for :class:`BaseLlmAdapter` — the historical default.

Ollama's own response objects already satisfy the contract (attribute plus key
access, ``message.content`` / ``done`` / ``response``), so they are passed
through untouched: the default deployment keeps behaving exactly as before this
adapter layer existed. Only errors are translated, so callers can catch one
exception type regardless of backend.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from ollama import AsyncClient as AsyncOllamaClient
from ollama import ResponseError

from src.agents.model_clients.llm_base import (
    BaseLlmAdapter,
    LlmChatResponse,
    LlmGenerateResponse,
    LlmResponseError,
)


class OllamaAdapter(BaseLlmAdapter):
    """Thin pass-through to ``ollama.AsyncClient``."""

    def __init__(self, host: str):
        self.host = host
        self.client = AsyncOllamaClient(host=host)

    @staticmethod
    def _request_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
        """Apply an experiment-wide request context without changing production.

        Ollama can reallocate a loaded model when a later request omits num_ctx.
        The restriction pipeline makes several LLM calls after planning, so
        setting it on the planner alone does not hold the runtime context fixed.
        """

        configured = os.getenv("OLLAMA_REQUEST_NUM_CTX", "").strip()
        if not configured:
            return options
        merged = dict(options or {})
        merged["num_ctx"] = int(configured)
        return merged

    @staticmethod
    def _request_think(model: str, think: bool | None) -> bool | str | None:
        """Apply model-specific native reasoning levels in benchmark processes."""

        for item in os.getenv("OLLAMA_THINK_LEVELS", "").split(","):
            item = item.strip()
            if not item or "=" not in item:
                continue
            name, level = item.rsplit("=", 1)
            if name.strip() == model:
                return level.strip()
        return think

    async def chat(
        self,
        model: str,
        messages: list[dict] | None = None,
        *,
        stream: bool = False,
        think: bool | None = None,
        format: Any = None,  # noqa: A002
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LlmChatResponse | AsyncIterator[LlmChatResponse]:
        call: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        request_think = self._request_think(model, think)
        if request_think is not None:
            call["think"] = request_think
        if format is not None:
            call["format"] = format
        request_options = self._request_options(options)
        if request_options is not None:
            call["options"] = request_options
        call.update(kwargs)
        try:
            return await self.client.chat(**call)
        except ResponseError as exc:
            raise LlmResponseError(str(exc), getattr(exc, "status_code", None)) from exc

    async def generate(
        self, model: str, prompt: str, *, stream: bool = False, **kwargs: Any
    ) -> LlmGenerateResponse:
        request_options = self._request_options(kwargs.get("options"))
        if request_options is not None:
            kwargs["options"] = request_options
        try:
            return await self.client.generate(
                model=model, prompt=prompt, stream=stream, **kwargs
            )
        except ResponseError as exc:
            raise LlmResponseError(str(exc), getattr(exc, "status_code", None)) from exc

    async def list(self) -> dict[str, list[dict[str, Any]]]:
        try:
            return await self.client.list()
        except ResponseError as exc:
            raise LlmResponseError(str(exc), getattr(exc, "status_code", None)) from exc

    async def ps(self) -> dict[str, list[dict[str, Any]]]:
        try:
            return await self.client.ps()
        except ResponseError as exc:
            raise LlmResponseError(str(exc), getattr(exc, "status_code", None)) from exc
