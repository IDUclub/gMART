"""LLM adapters for the benchmark: the real gMART adapters behind a recording proxy.

Stages call gMART's own prompt builders and parsers, which talk to an
``llm_client``. :class:`RecordingLlm` is that client: it forwards every call to the
adapter of the model the profile assigned to the stage, applies the model's option
overrides, and records what was sent and received (content, reasoning, latency,
finish reason) so the report can show the raw output next to the score.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from benchmarks.harness.config import EndpointConfig, ModelConfig
from src.agents.model_clients.endpoint_policy import (
    reject_forbidden_llm_host,
    require_local_ollama_url,
)
from src.agents.model_clients.factory import _with_api_path
from src.agents.model_clients.llm_base import BaseLlmAdapter
from src.agents.model_clients.ollama_adapter import OllamaAdapter
from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter


def build_adapter(endpoint: EndpointConfig) -> BaseLlmAdapter:
    """Build a gMART adapter for one endpoint, honouring the service's host policy."""
    if endpoint.backend == "ollama":
        host = endpoint.host or endpoint.base_url
        if not host:
            raise ValueError(f"endpoint {endpoint.name!r}: ollama needs host")
        require_local_ollama_url(host, f"endpoints.{endpoint.name}.host")
        return OllamaAdapter(host=host)
    if endpoint.backend != "openai":
        raise ValueError(f"endpoint {endpoint.name!r}: unknown backend")
    url = endpoint.base_url or endpoint.host
    if not url:
        raise ValueError(f"endpoint {endpoint.name!r}: openai needs base_url")
    reject_forbidden_llm_host(url, f"endpoints.{endpoint.name}.base_url")
    return OpenAiCompatAdapter(
        base_url=_with_api_path(url),
        api_key=endpoint.api_key or None,
        timeout=endpoint.timeout,
        think_mode=endpoint.think_mode,
        think_chat_template_kwarg=endpoint.think_chat_template_kwarg,
        think_effort=endpoint.think_effort,
    )


@dataclass
class CallRecord:
    request_kind: str | None
    messages: list[dict] | None
    format: Any
    options: dict[str, Any]
    content: str = ""
    thinking: str = ""
    done_reason: str | None = None
    latency_s: float = 0.0
    first_token_s: float | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, with_prompt: bool) -> dict[str, Any]:
        data = {
            "request_kind": self.request_kind,
            "options": self.options,
            "content": self.content,
            "thinking": self.thinking,
            "done_reason": self.done_reason,
            "latency_s": round(self.latency_s, 3),
            "first_token_s": (
                round(self.first_token_s, 3) if self.first_token_s is not None else None
            ),
            "error": self.error,
            **self.extra,
        }
        if with_prompt:
            data["messages"] = self.messages
            data["format"] = self.format
        return data


def _text(part: Any, key: str) -> str:
    message = part.get("message") if hasattr(part, "get") else None
    if message is None:
        return ""
    value = message.get(key) if hasattr(message, "get") else None
    return value or ""


class RecordingLlm:
    """``llm_client`` proxy: routes to one model and records every completion.

    ``__class__`` reports the wrapped adapter's class, so gMART code that checks
    ``isinstance(llm_client, OpenAiCompatAdapter)`` (reasoning-effort choices for
    gpt-oss) behaves exactly as in production.
    """

    def __init__(self, adapter: BaseLlmAdapter, model: ModelConfig) -> None:
        self._adapter = adapter
        self._model = model
        self.calls: list[CallRecord] = []

    @property
    def __class__(self):  # noqa: D105 - isinstance transparency, see class docstring
        return type(self._adapter)

    @property
    def model_id(self) -> str:
        return self._model.model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)

    def _options(self, options: dict[str, Any] | None) -> dict[str, Any]:
        return {**(options or {}), **self._model.options}

    async def chat(
        self,
        model: str | None = None,
        messages: list[dict] | None = None,
        *,
        stream: bool = False,
        request_kind: str | None = None,
        options: dict[str, Any] | None = None,
        format: Any = None,  # noqa: A002 - the keyword gMART call sites pass
        **kwargs: Any,
    ):
        options = self._options(options)
        record = CallRecord(
            request_kind=request_kind,
            messages=messages,
            format=format,
            options=options,
            extra={
                key: kwargs[key]
                for key in ("think", "reasoning_effort")
                if key in kwargs
            },
        )
        self.calls.append(record)
        started = time.perf_counter()
        try:
            response = await self._adapter.chat(
                self._model.model,
                messages,
                stream=stream,
                request_kind=request_kind,
                options=options,
                format=format,
                **kwargs,
            )
        except Exception as exc:
            record.latency_s = time.perf_counter() - started
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        if stream:
            return self._recorded_stream(response, record, started)
        record.latency_s = time.perf_counter() - started
        record.content = _text(response, "content")
        record.thinking = _text(response, "thinking")
        record.done_reason = response.get("done_reason")
        return response

    async def _recorded_stream(self, stream, record: CallRecord, started: float):
        content: list[str] = []
        thinking: list[str] = []
        try:
            async for part in stream:
                text = _text(part, "content")
                if text and record.first_token_s is None:
                    record.first_token_s = time.perf_counter() - started
                content.append(text)
                thinking.append(_text(part, "thinking"))
                if part.get("done_reason"):
                    record.done_reason = part.get("done_reason")
                yield part
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.latency_s = time.perf_counter() - started
            record.content = "".join(content)
            record.thinking = "".join(thinking)

    async def generate(
        self,
        model: str | None = None,
        prompt: str = "",
        *,
        stream: bool = False,
        request_kind: str | None = None,
        **kwargs: Any,
    ):
        record = CallRecord(
            request_kind=request_kind,
            messages=[{"role": "user", "content": prompt}],
            format=None,
            options=self._options(kwargs.pop("options", None)),
        )
        self.calls.append(record)
        started = time.perf_counter()
        try:
            response = await self._adapter.generate(
                self._model.model,
                prompt,
                stream=stream,
                request_kind=request_kind,
                **({"options": record.options} if record.options else {}),
                **kwargs,
            )
        except Exception as exc:
            record.latency_s = time.perf_counter() - started
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        record.latency_s = time.perf_counter() - started
        record.content = response.get("response") or ""
        record.done_reason = response.get("done_reason")
        return response
