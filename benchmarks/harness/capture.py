"""A stand-in model that records the request a stage sends and stops the stage there.

Every benchmark case is one model call, so the request gMART builds for it can be
frozen: :class:`CaptureLlm` takes the place of the adapter, records the call exactly
as the agent code makes it (messages, schema, options, ``think``…) and raises
:class:`Captured` instead of answering. ``Captured`` derives from ``BaseException``
so the ``except Exception`` fallbacks around model calls in the services do not
swallow it and carry on with a made-up answer.
"""

from __future__ import annotations

from typing import Any

from src.agents.model_clients.llm_base import BaseLlmAdapter


class Captured(BaseException):
    def __init__(self, request: dict[str, Any]) -> None:
        super().__init__(request.get("request_kind") or "captured")
        self.request = request


class CaptureLlm(BaseLlmAdapter):
    def __init__(self, *, context_window: int | None = None) -> None:
        self.context_window = context_window
        self.requests: list[dict[str, Any]] = []

    @property
    def calls(self) -> list[dict[str, Any]]:
        # Stages read ``llm.calls`` to tell a model decision from a code one.
        return self.requests

    async def model_context_window(self, model: str) -> int | None:
        return self.context_window

    def _capture(self, request: dict[str, Any]) -> None:
        self.requests.append(request)
        raise Captured(request)

    async def chat(
        self,
        model: str,
        messages: list[dict] | None = None,
        *,
        stream: bool = False,
        think: bool | None = None,
        format: Any = None,  # noqa: A002 - the keyword gMART call sites pass
        options: dict[str, Any] | None = None,
        request_kind: str | None = None,
        **kwargs: Any,
    ):
        self._capture(
            {
                "api": "chat",
                "request_kind": request_kind,
                "messages": messages or [],
                "format": format,
                "options": dict(options or {}),
                "think": think,
                "stream": stream,
                "kwargs": kwargs,
            }
        )

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        stream: bool = False,
        request_kind: str | None = None,
        **kwargs: Any,
    ):
        options = kwargs.pop("options", None)
        think = kwargs.pop("think", None)
        self._capture(
            {
                "api": "generate",
                "request_kind": request_kind,
                "messages": [{"role": "user", "content": prompt}],
                "format": kwargs.pop("format", None),
                "options": dict(options or {}),
                "think": think,
                "stream": stream,
                "kwargs": kwargs,
            }
        )

    async def list(self) -> dict[str, list[dict[str, Any]]]:
        return {"models": []}

    async def ps(self) -> dict[str, list[dict[str, Any]]]:
        return {"models": []}
