"""Backend-neutral contract for the LLM clients the agents use.

Every agent talks to the model through :class:`BaseLlmAdapter`, so the inference
engine is a deployment choice rather than something baked into the services.
Two backends implement it: Ollama (the historical one, still the default) and
any OpenAI-compatible server such as vLLM.

The response objects deliberately mimic Ollama's: the call sites read them both
as attributes (``part.message.content``, ``part.done``, ``title.response``) and
as mappings (``response["message"]["content"]``), so keeping that shape lets the
two backends be swapped without touching a single agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel, ConfigDict


class LlmResponseError(Exception):
    """Backend-independent transport error.

    Carries ``status_code`` because the services map a 404 from the model server
    onto the REST-facing ``ModelNotFound``.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class _SubscriptableModel(BaseModel):
    """Model allowing both attribute and key access, as Ollama's types do."""

    model_config = ConfigDict(extra="allow")

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class LlmMessage(_SubscriptableModel):
    role: str = "assistant"
    content: str = ""
    thinking: str | None = None


class LlmChatResponse(_SubscriptableModel):
    model: str = ""
    message: LlmMessage = LlmMessage()
    done: bool = True
    done_reason: str | None = None


class LlmGenerateResponse(_SubscriptableModel):
    model: str = ""
    response: str = ""
    done: bool = True


class BaseLlmAdapter(ABC):
    """The whole surface the agents use — nothing else may be added lightly.

    ``chat`` returns a single :class:`LlmChatResponse` when ``stream`` is false
    and an async iterator of them when it is true; the call sites therefore do
    ``async for part in await client.chat(..., stream=True)``, which mirrors the
    Ollama client and must be preserved by every implementation.
    """

    @abstractmethod
    async def chat(
        self,
        model: str,
        messages: list[dict] | None = None,
        *,
        stream: bool = False,
        think: bool | None = None,
        format: Any = None,  # noqa: A002 — the Ollama keyword the call sites pass
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LlmChatResponse | AsyncIterator[LlmChatResponse]:
        """Chat completion. ``format`` takes a JSON schema for structured output."""

    @abstractmethod
    async def generate(
        self, model: str, prompt: str, *, stream: bool = False, **kwargs: Any
    ) -> LlmGenerateResponse:
        """Single-prompt completion (used for chat titles)."""

    @abstractmethod
    async def list(self) -> dict[str, list[dict[str, Any]]]:
        """``{"models": [{"model": name}, ...]}`` — the shape get_models expects."""

    @abstractmethod
    async def ps(self) -> dict[str, list[dict[str, Any]]]:
        """Models currently loaded, same shape as :meth:`list`."""
