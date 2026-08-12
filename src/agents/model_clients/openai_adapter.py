"""OpenAI-compatible backend for :class:`BaseLlmAdapter` (vLLM, TGI, llama.cpp…).

Translates the Ollama-shaped calls the agents make into ``/v1/chat/completions``:

  ``format=<JSON schema>``          -> ``response_format={"type": "json_schema"...}``
  ``options={"temperature": ...}``  -> ``temperature``
  ``options={"num_predict": ...}``  -> ``max_tokens``
  ``options={"num_ctx": ...}``      -> dropped; the context length is a server-side
                                       setting there (vLLM's ``--max-model-len``)
  ``think=False``                   -> dropped; reasoning models on an OpenAI API
                                       return the trace in ``reasoning_content``
                                       and the answer in ``content`` anyway

Both dropped parameters are logged once per process so a silent behaviour change
cannot hide: structured output and the reasoning toggle are what keep the
restriction planner's JSON valid.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from loguru import logger
from openai import APIStatusError, AsyncOpenAI, OpenAIError

from src.agents.model_clients.llm_base import (
    BaseLlmAdapter,
    LlmChatResponse,
    LlmGenerateResponse,
    LlmMessage,
    LlmResponseError,
)

_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


class OpenAiCompatAdapter(BaseLlmAdapter):
    """Talks to any server exposing the OpenAI chat-completions API."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float | None = None,
    ):
        self.base_url = base_url
        # Local servers usually ignore the key but the SDK requires one.
        self.client = AsyncOpenAI(
            base_url=base_url, api_key=api_key or "not-needed", timeout=timeout
        )

    # ------------------------------------------------------------------ #
    # translation helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _response_format(format: Any) -> dict[str, Any] | None:  # noqa: A002
        if format is None:
            return None
        if format == "json":
            return {"type": "json_object"}
        if isinstance(format, dict):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": format.get("title", "response"),
                    "schema": format,
                    "strict": False,
                },
            }
        _warn_once("format", f"unsupported format={format!r}, ignored")
        return None

    @staticmethod
    def _sampling(options: dict[str, Any] | None) -> dict[str, Any]:
        if not options:
            return {}
        out: dict[str, Any] = {}
        if "temperature" in options:
            out["temperature"] = options["temperature"]
        if "num_predict" in options:
            out["max_tokens"] = options["num_predict"]
        if "top_p" in options:
            out["top_p"] = options["top_p"]
        if "num_ctx" in options:
            _warn_once(
                "num_ctx",
                "num_ctx is not an OpenAI-API parameter and was dropped; set the "
                "context length on the server (vLLM: --max-model-len)",
            )
        return out

    def _build(
        self,
        model: str,
        messages: list[dict] | None,
        stream: bool,
        think: bool | None,
        format: Any,  # noqa: A002
        options: dict[str, Any] | None,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        if think is not None:
            _warn_once(
                "think",
                "think= is Ollama-specific and was dropped; on an OpenAI API the "
                "reasoning trace arrives in reasoning_content, not in content",
            )
        call: dict[str, Any] = {
            "model": model,
            "messages": messages or [],
            "stream": stream,
            **self._sampling(options),
            **extra,
        }
        response_format = self._response_format(format)
        if response_format:
            call["response_format"] = response_format
        return call

    @staticmethod
    def _as_response(completion: Any) -> LlmChatResponse:
        choice = completion.choices[0] if completion.choices else None
        message = getattr(choice, "message", None)
        return LlmChatResponse(
            model=getattr(completion, "model", "") or "",
            message=LlmMessage(
                role=getattr(message, "role", "assistant") or "assistant",
                content=getattr(message, "content", "") or "",
                thinking=getattr(message, "reasoning_content", None),
            ),
            done=True,
            done_reason=getattr(choice, "finish_reason", None),
        )

    @staticmethod
    async def _as_stream(stream: Any) -> AsyncIterator[LlmChatResponse]:
        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            delta = getattr(choice, "delta", None)
            yield LlmChatResponse(
                model=getattr(chunk, "model", "") or "",
                message=LlmMessage(
                    role=getattr(delta, "role", "assistant") or "assistant",
                    content=getattr(delta, "content", None) or "",
                    thinking=getattr(delta, "reasoning_content", None),
                ),
                done=getattr(choice, "finish_reason", None) is not None,
                done_reason=getattr(choice, "finish_reason", None),
            )

    # ------------------------------------------------------------------ #
    # BaseLlmAdapter
    # ------------------------------------------------------------------ #
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
        call = self._build(model, messages, stream, think, format, options, kwargs)
        try:
            result = await self.client.chat.completions.create(**call)
        except APIStatusError as exc:
            raise LlmResponseError(str(exc), exc.status_code) from exc
        except OpenAIError as exc:
            raise LlmResponseError(str(exc)) from exc
        return self._as_stream(result) if stream else self._as_response(result)

    async def generate(
        self, model: str, prompt: str, *, stream: bool = False, **kwargs: Any
    ) -> LlmGenerateResponse:
        response = await self.chat(
            model, [{"role": "user", "content": prompt}], stream=False, **kwargs
        )
        return LlmGenerateResponse(
            model=response.model, response=response.message.content, done=True
        )

    async def list(self) -> dict[str, list[dict[str, Any]]]:
        try:
            models = await self.client.models.list()
        except APIStatusError as exc:
            raise LlmResponseError(str(exc), exc.status_code) from exc
        except OpenAIError as exc:
            raise LlmResponseError(str(exc)) from exc
        return {"models": [{"model": m.id, "name": m.id} for m in models.data]}

    async def ps(self) -> dict[str, list[dict[str, Any]]]:
        # An OpenAI-compatible server loads its models at start-up and has no
        # residency concept, so "running" and "available" are the same set.
        return await self.list()
