"""Character limit for the complete textual model input, separate from tokens."""

import json
import os


def request_limit() -> int:
    value = int(os.getenv("DVD_REQUEST_MAX_CHARS", "32000"))
    if value < 4096:
        raise ValueError("DVD_REQUEST_MAX_CHARS must be at least 4096")
    return value


def request_chars(messages, schema=None) -> int:
    payload = {"messages": messages}
    if schema is not None:
        payload["format"] = schema
    return len(json.dumps(payload, ensure_ascii=False))


def check_request(messages, schema=None):
    if request_chars(messages, schema) > request_limit():
        raise ValueError("model_request_exceeds_character_limit")
