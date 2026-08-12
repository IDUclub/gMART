"""Ollama backend for :class:`BaseLlmAdapter` — the historical default.

Ollama's own response objects already satisfy the contract (attribute plus key
access, ``message.content`` / ``done`` / ``response``), so they are passed
through untouched: the default deployment keeps behaving exactly as before this
adapter layer existed. Only errors are translated, so callers can catch one
exception type regardless of backend.
"""

from __future__ import annotations

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
        if think is not None:
            call["think"] = think
        if format is not None:
            call["format"] = format
        if options is not None:
            call["options"] = options
        call.update(kwargs)
        try:
            return await self.client.chat(**call)
        except ResponseError as exc:
            raise LlmResponseError(str(exc), getattr(exc, "status_code", None)) from exc

    async def generate(
        self, model: str, prompt: str, *, stream: bool = False, **kwargs: Any
    ) -> LlmGenerateResponse:
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
