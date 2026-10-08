"""LLM calls and tokens spent by one pipeline run.

A run opens a :class:`LlmUsage` with :func:`meter`; every completion the adapters
finish while it is active (including calls made from tasks the run spawns, which
copy the context) is added to it by :func:`record`. Calls outside a metered run are
not counted anywhere.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass

_USAGE: ContextVar[LlmUsage | None] = ContextVar("llm_usage", default=None)


@dataclass
class LlmUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Calls whose server reported no token counts (counted, not measured).
    unmeasured_calls: int = 0
    tokenize_calls: int = 0
    tokenize_cache_hits: int = 0

    def snapshot(self) -> dict[str, int]:
        return asdict(self)


def meter() -> LlmUsage:
    """Start counting the current context's LLM calls into a new :class:`LlmUsage`.

    The value stays set for the rest of the task, deliberately without a reset:
    an async generator may be resumed from another context, where a reset fails.
    """
    usage = LlmUsage()
    _USAGE.set(usage)
    return usage


def current() -> LlmUsage | None:
    return _USAGE.get()


def record(
    *, prompt_tokens: int | None = None, completion_tokens: int | None = None
) -> None:
    usage = _USAGE.get()
    if usage is None:
        return
    usage.calls += 1
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        usage.input_tokens += prompt_tokens
        usage.output_tokens += completion_tokens
    else:
        usage.unmeasured_calls += 1


def record_tokenize(*, cached: bool = False) -> None:
    usage = _USAGE.get()
    if usage is None:
        return
    if cached:
        usage.tokenize_cache_hits += 1
    else:
        usage.tokenize_calls += 1
