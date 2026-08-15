"""How much context a served model will accept.

The restrictions context grows with the scenario: on the largest ones the final
answer's prompt reached 380 000 tokens against a 16 384 limit, and the request
came back 400 with the whole pipeline row lost. Knowing the ceiling lets the
caller fold the context into parts that fit instead.

``MODEL_CONTEXT_TOKENS`` carries the per-model ceilings in the same shape as
``OPENAI_MODEL_ROUTES``::

    MODEL_CONTEXT_TOKENS="gpt-oss-20b=16384,gemma-3-27b=32768"

An unlisted model falls back to ``MODEL_CONTEXT_TOKENS_DEFAULT``. The value must
match what the server was started with (vLLM's ``--max-model-len``); when it
drifts, the 400 is still caught and the context folded harder, so a stale
setting costs a retry rather than the row.
"""

from __future__ import annotations

import os

DEFAULT_CONTEXT_TOKENS = 16384
# Room for the system prompt, the question, the history and the answer itself.
DEFAULT_RESERVE_TOKENS = 3072
# Russian prose and Cyrillic JSON keys tokenise at roughly three characters per
# token — deliberately pessimistic, since overshooting costs a failed request.
DEFAULT_CHARS_PER_TOKEN = 3


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, "") or default))
    except ValueError:
        return default


def context_tokens(model: str) -> int:
    """The model's context window, in tokens."""

    default = _int_env("MODEL_CONTEXT_TOKENS_DEFAULT", DEFAULT_CONTEXT_TOKENS)
    for item in (os.getenv("MODEL_CONTEXT_TOKENS", "") or "").split(","):
        name, _, value = item.strip().partition("=")
        if name.strip() == model and value.strip():
            try:
                return max(1, int(value))
            except ValueError:
                return default
    return default


def context_budget_chars(model: str) -> int:
    """Characters of context this model can be given in one call."""

    reserve = _int_env("MODEL_CONTEXT_RESERVE_TOKENS", DEFAULT_RESERVE_TOKENS)
    chars_per_token = _int_env("MODEL_CHARS_PER_TOKEN", DEFAULT_CHARS_PER_TOKEN)
    usable = max(context_tokens(model) - reserve, reserve)
    return usable * chars_per_token
