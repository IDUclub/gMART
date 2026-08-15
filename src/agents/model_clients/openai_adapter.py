"""OpenAI-compatible backend for :class:`BaseLlmAdapter` (vLLM, TGI, llama.cpp…).

Translates the Ollama-shaped calls the agents make into ``/v1/chat/completions``:

  ``format=<JSON schema>``          -> ``response_format={"type": "json_schema"...}``
  ``options={"temperature": ...}``  -> ``temperature``
  ``options={"num_predict": ...}``  -> ``max_tokens``
  ``options={"num_ctx": ...}``      -> dropped; the context length is a server-side
                                       setting there (vLLM's ``--max-model-len``)
  ``think=False``                   -> ``reasoning_effort=<OPENAI_THINK_OFF_EFFORT>``

Dropping ``think=False`` would not be cosmetic on a reasoning model: the trace is
then generated as part of the completion and eats the ``num_predict`` budget, which
can return an empty ``content`` with ``finish_reason="length"``. ``reasoning_effort``
is the one spelling both engines in use honour — verified against vLLM 0.27 and
Ollama's own ``/v1`` — so it is the default and needs no configuration.

``OPENAI_THINK_MODE`` selects another spelling for servers that need one:

  ``reasoning_effort`` (default) — ``reasoning_effort=OPENAI_THINK_OFF_EFFORT``,
                                  "none" by default; gpt-oss served through vLLM's
                                  Harmony format rejects "none" with a 400 and needs
                                  "low", which every server tested also accepts
  ``chat_template``             — ``chat_template_kwargs={"enable_thinking": False}``,
                                  which vLLM honours but Ollama ignores; the variable
                                  name comes from OPENAI_THINK_CHAT_TEMPLATE_KWARG
  ``off``                       — send nothing, and warn once

Whatever is dropped is logged once per process so a silent behaviour change cannot
hide: structured output and the reasoning toggle are what keep the restriction
planner's JSON valid.
"""

from __future__ import annotations

import os
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

THINK_REASONING_EFFORT = "reasoning_effort"
THINK_CHAT_TEMPLATE = "chat_template"
THINK_OFF = "off"

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
        think_mode: str | None = None,
        think_chat_template_kwarg: str | None = None,
        routes: dict[str, str] | None = None,
        think_off_effort: str | None = None,
    ):
        self.base_url = base_url
        # vLLM serves one model per process, so a deployment driving several models
        # needs a server per model; routes maps model name -> base URL and lets one
        # agents instance reach all of them without being restarted in between.
        self.routes = dict(routes or {})
        self._api_key = api_key or "not-needed"
        self._timeout = timeout
        self._routed: dict[str, AsyncOpenAI] = {}
        # How think= is spelled for this server; see the module docstring.
        self.think_mode = (
            think_mode
            if think_mode is not None
            else os.getenv("OPENAI_THINK_MODE", THINK_REASONING_EFFORT)
        ).strip().lower() or THINK_OFF
        # Which reasoning_effort means "as little as possible". "none" is the
        # OpenAI spelling, but gpt-oss served through vLLM's Harmony format rejects
        # it with 400 ("Harmony does not support reasoning_effort='none'") and needs
        # "low"; both servers in a mixed deployment must therefore be able to differ.
        self.think_off_effort = (
            think_off_effort
            if think_off_effort is not None
            else os.getenv("OPENAI_THINK_OFF_EFFORT", "none")
        ).strip() or "none"
        # Chat-template variable carrying think= when think_mode is chat_template.
        self.think_kwarg = (
            think_chat_template_kwarg
            if think_chat_template_kwarg is not None
            else os.getenv("OPENAI_THINK_CHAT_TEMPLATE_KWARG", "enable_thinking")
        ).strip()
        # Local servers usually ignore the key but the SDK requires one.
        self.client = AsyncOpenAI(
            base_url=base_url, api_key=self._api_key, timeout=timeout
        )

    def client_for(self, model: str) -> AsyncOpenAI:
        """The server serving ``model``; the default one when the model is unrouted."""

        url = self.routes.get(model)
        if not url:
            return self.client
        if url not in self._routed:
            self._routed[url] = AsyncOpenAI(
                base_url=url, api_key=self._api_key, timeout=self._timeout
            )
        return self._routed[url]

    # ------------------------------------------------------------------ #
    # translation helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _reasoning(part: Any) -> str | None:
        """The reasoning trace, whichever name the server gives it.

        vLLM (with ``--reasoning-parser``) calls it ``reasoning_content``; Ollama's
        own ``/v1`` endpoint and several proxies call it ``reasoning``.
        """

        return getattr(part, "reasoning_content", None) or getattr(
            part, "reasoning", None
        )

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

    def _apply_think(self, call: dict[str, Any], think: bool) -> None:
        """Spell out ``think=`` the way this server understands it."""

        if self.think_mode == THINK_REASONING_EFFORT:
            # Reasoning is on by default everywhere, so only switching it off says
            # anything; setdefault keeps an explicit caller value.
            if think is False:
                call.setdefault("reasoning_effort", self.think_off_effort)
        elif self.think_mode == THINK_CHAT_TEMPLATE:
            # Merge rather than overwrite: a caller may pass its own extra_body.
            extra_body = dict(call.get("extra_body") or {})
            template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            template_kwargs.setdefault(self.think_kwarg, think)
            extra_body["chat_template_kwargs"] = template_kwargs
            call["extra_body"] = extra_body
        else:
            _warn_once(
                "think",
                "think= was dropped (OPENAI_THINK_MODE=off); on a reasoning model the "
                "trace is generated anyway and consumes the num_predict budget",
            )

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
        call: dict[str, Any] = {
            "model": model,
            "messages": messages or [],
            "stream": stream,
            **self._sampling(options),
            **extra,
        }
        if think is not None:
            self._apply_think(call, think)
        response_format = self._response_format(format)
        if response_format:
            call["response_format"] = response_format
        return call

    @classmethod
    def _as_response(cls, completion: Any) -> LlmChatResponse:
        choice = completion.choices[0] if completion.choices else None
        message = getattr(choice, "message", None)
        content = getattr(message, "content", "") or ""
        finish_reason = getattr(choice, "finish_reason", None)
        reasoning = cls._reasoning(message)
        if not content and finish_reason == "length":
            # The budget ran out before any answer was produced — on a reasoning
            # model, inside the trace. vLLM's reasoning parser only fills the field
            # once it sees the closing tag, so truncation mid-thought leaves both
            # content and the trace empty; the warning must not depend on the trace.
            # Callers only see an empty string, so say it out loud: the fix is more
            # max_tokens or disabling thinking, not a retry.
            logger.warning(
                "empty content with finish_reason=length"
                + (
                    f" after a {len(reasoning)}-char reasoning trace"
                    if reasoning
                    else ""
                )
                + ": the whole num_predict budget was spent before an answer. Raise "
                "it, or disable thinking via OPENAI_THINK_CHAT_TEMPLATE_KWARG"
            )
        return LlmChatResponse(
            model=getattr(completion, "model", "") or "",
            message=LlmMessage(
                role=getattr(message, "role", "assistant") or "assistant",
                content=content,
                thinking=reasoning,
            ),
            done=True,
            done_reason=finish_reason,
        )

    @classmethod
    async def _as_stream(cls, stream: Any) -> AsyncIterator[LlmChatResponse]:
        """Yield chunks, always ending on one with ``done=True``.

        Two shapes have to be absorbed for the call sites — which loop until
        ``part.done`` — to terminate against any server: a trailing chunk carrying
        only ``usage`` and no choices (what ``stream_options.include_usage`` adds),
        and a stream that simply stops without ever sending ``finish_reason``.
        Ollama always closes with ``done=True``, so this keeps the two backends
        interchangeable.
        """

        model = ""
        finished = False
        async for chunk in stream:
            model = getattr(chunk, "model", "") or model
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            finish_reason = getattr(choice, "finish_reason", None)
            finished = finished or finish_reason is not None
            yield LlmChatResponse(
                model=model,
                message=LlmMessage(
                    role=getattr(delta, "role", "assistant") or "assistant",
                    content=getattr(delta, "content", None) or "",
                    thinking=cls._reasoning(delta),
                ),
                done=finish_reason is not None,
                done_reason=finish_reason,
            )
        if not finished:
            yield LlmChatResponse(model=model, done=True, done_reason="stop")

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
            result = await self.client_for(model).chat.completions.create(**call)
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
        """Union over every routed server, so validate_model() accepts a model
        that lives on a routed server rather than on the default one."""

        clients = [self.client] + [self.client_for(m) for m in self.routes]
        seen: dict[str, dict[str, Any]] = {}
        for client in clients:
            try:
                models = await client.models.list()
            except APIStatusError as exc:
                raise LlmResponseError(str(exc), exc.status_code) from exc
            except OpenAIError as exc:
                raise LlmResponseError(str(exc)) from exc
            for m in models.data:
                seen.setdefault(m.id, {"model": m.id, "name": m.id})
        return {"models": list(seen.values())}

    async def ps(self) -> dict[str, list[dict[str, Any]]]:
        # An OpenAI-compatible server loads its models at start-up and has no
        # residency concept, so "running" and "available" are the same set.
        return await self.list()
