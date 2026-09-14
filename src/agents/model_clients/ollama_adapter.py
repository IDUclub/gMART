"""Ollama backend for :class:`BaseLlmAdapter` — the historical default.

Ollama's own response objects already satisfy the contract (attribute plus key
access, ``message.content`` / ``done`` / ``response``), so they are passed
through untouched: the default deployment keeps behaving exactly as before this
adapter layer existed. Only errors are translated, so callers can catch one
exception type regardless of backend.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from ollama import AsyncClient as AsyncOllamaClient
from ollama import ResponseError

from src.agents.model_clients.llm_base import (
    BaseLlmAdapter,
    LlmChatResponse,
    LlmGenerateResponse,
    LlmResponseError,
    closing_stream,
)
from src.agents.runtime.budget import current_budget


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
        effort = call.pop("reasoning_effort", None)
        if effort is not None:
            call["think"] = effort if "gpt-oss" in model.lower() else True
        budget = current_budget.get()
        reservation = None
        if budget is not None:
            reservation = budget.reserve(
                messages, format, (options or {}).get("num_predict")
            )
            call["options"] = {**(options or {}), "num_predict": reservation.output}
        try:
            async with asyncio.timeout(budget.remaining_seconds if budget else None):
                result = await self.client.chat(**call)
            if reservation is not None:
                if stream:
                    return self._budget_stream(result, reservation)
                self._settle(reservation, result)
            return self._as_stream(result) if stream else result
        except ResponseError as exc:
            if reservation is not None:
                reservation.settle()
            raise LlmResponseError(str(exc), getattr(exc, "status_code", None)) from exc
        except BaseException:
            if reservation is not None:
                reservation.settle()
            raise

    @staticmethod
    def _settle(reservation, response):
        prompt = getattr(response, "prompt_eval_count", None)
        output = getattr(response, "eval_count", None)
        reservation.settle(
            {"total_tokens": prompt + output}
            if isinstance(prompt, int) and isinstance(output, int)
            else None
        )

    async def _budget_stream(self, stream, reservation):
        try:
            async with asyncio.timeout(reservation.budget.remaining_seconds):
                async with closing_stream(self._as_stream(stream)) as parts:
                    async for part in parts:
                        if getattr(part, "done", False):
                            self._settle(reservation, part)
                        yield part
        finally:
            reservation.settle()

    async def _as_stream(self, stream):
        try:
            async with closing_stream(stream):
                async for part in stream:
                    yield part
        except ResponseError as exc:
            raise LlmResponseError(str(exc), getattr(exc, "status_code", None)) from exc

    async def generate(
        self, model: str, prompt: str, *, stream: bool = False, **kwargs: Any
    ) -> LlmGenerateResponse:
        if current_budget.get() is not None:
            response = await self.chat(
                model, [{"role": "user", "content": prompt}], stream=False, **kwargs
            )
            return LlmGenerateResponse(
                model=model,
                response=response.message.content,
                usage={
                    "total_tokens": (getattr(response, "prompt_eval_count", 0) or 0)
                    + (getattr(response, "eval_count", 0) or 0)
                },
            )
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
