"""Allocate the remaining context to generation, with no independent output cap."""

import json

from loguru import logger


def estimated_input_tokens(messages, schema=None):
    # Unknown tokenizers: UTF-8 bytes are a conservative upper estimate.
    result = 256 + sum(
        64 + len(str(m.get("content", "")).encode("utf-8")) for m in messages
    )
    if schema is not None:
        result += len(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
    return result


async def remaining_output_tokens(
    llm, model, messages, window, *, schema=None, reasoning_effort=None
):
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
    logger.info(
        "Model context model={} input_tokens={} counting={} output_tokens={} window={}",
        model,
        count,
        source,
        available,
        window,
    )
    return available
