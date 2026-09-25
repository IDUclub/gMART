"""Allocate generation tokens from the context window, in proportion to the input.

A call without an ``output`` share may use the whole remaining window. With a share,
generation is limited to ``floor + ratio * input_tokens``: a larger context to
work through gets a larger answer, while a model that never stops (runaway
reasoning or a repeating JSON value) ends early instead of occupying the shared
server for the rest of the window.
"""

import json
from dataclasses import dataclass

from loguru import logger


@dataclass(frozen=True)
class OutputShare:
    """Output tokens allowed per input token, plus a fixed allowance for reasoning."""

    ratio: float
    floor: int

    def limit(self, input_tokens: int) -> int:
        return self.floor + int(self.ratio * max(0, input_tokens))


# Structured JSON (plans, audits, selections): the reply restates part of the
# input at most; the floor covers gpt-oss reasoning, which counts as output.
STRUCTURED_OUTPUT = OutputShare(ratio=0.5, floor=4096)
# An answer audit restates every answer line with its supporting quotes: on
# gpt-oss-20b it measured 1.2-1.8 output tokens per input token.
AUDIT_OUTPUT = OutputShare(ratio=2.0, floor=4096)
# Verbatim evidence extraction may quote every relevant input sentence.
EVIDENCE_OUTPUT = OutputShare(ratio=1.0, floor=2048)
# A drafted answer; a truncated draft is continued by the caller.
ANSWER_OUTPUT = OutputShare(ratio=0.5, floor=4096)


def estimated_input_tokens(messages, schema=None):
    # Unknown tokenizers: UTF-8 bytes are a conservative upper estimate.
    result = 256 + sum(
        64 + len(str(m.get("content", "")).encode("utf-8")) for m in messages
    )
    if schema is not None:
        result += len(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
    return result


@dataclass(frozen=True)
class OutputBudget:
    """``window_rest``: tokens left in the window; ``tokens``: what to request."""

    window_rest: int
    tokens: int

    @property
    def limited(self) -> bool:
        """Whether a truncated reply may still fit with a wider proportional limit."""
        return self.tokens < self.window_rest


async def output_budget(
    llm,
    model,
    messages,
    window,
    *,
    schema=None,
    reasoning_effort=None,
    output: OutputShare | None = None,
    scale: float = 1.0,
) -> OutputBudget:
    """Output tokens for ``messages``: the window remainder, limited by ``output``.

    ``scale`` widens the proportional limit, e.g. after a reply was truncated by it.
    """
    counter = getattr(llm, "model_input_tokens", None)
    count = (
        await counter(model, messages, reasoning_effort=reasoning_effort)
        if counter
        else None
    )
    if type(count) is int and count >= 0:
        # The server renders its own chat template; leave a small safety margin
        # for provider-specific completion framing. Schemas constrain decoding.
        available = window - count - 256
        source = "server"
    else:
        count = estimated_input_tokens(messages, schema)
        available = window - count
        source = "estimate"
    limit = None if output is None else int(output.limit(count) * scale)
    tokens = available if limit is None else min(available, limit)
    logger.info(
        "Model context model={} input_tokens={} counting={} output_tokens={} "
        "output_limit={} window={}",
        model,
        count,
        source,
        tokens,
        limit,
        window,
    )
    return OutputBudget(window_rest=available, tokens=tokens)


async def remaining_output_tokens(llm, model, messages, window, **kwargs) -> int:
    """Output tokens to request; see :func:`output_budget`."""
    return (await output_budget(llm, model, messages, window, **kwargs)).tokens
