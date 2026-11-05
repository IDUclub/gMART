"""Helpers shared by the pydantic-ai and ADK models: reading adapter replies.

The adapters (and every test double standing in for them) answer in Ollama's
shape, reachable both as attributes and as mapping keys; doubles sometimes return
plain dicts or ``SimpleNamespace``. These helpers read either form so the agent
frameworks sit on any of them.
"""

from __future__ import annotations

import json
from typing import Any


def field(obj: Any, name: str, default: Any = None) -> Any:
    """``obj.name`` or ``obj[name]``, whichever the reply supports."""

    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    value = getattr(obj, name, None)
    if value is not None:
        return value
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(name, default)
        except TypeError:
            return default
    return default


def message_of(reply: Any) -> Any:
    return field(reply, "message")


def content_of(reply: Any) -> str:
    return field(message_of(reply), "content") or ""


def thinking_of(reply: Any) -> str | None:
    return field(message_of(reply), "thinking")


def done_reason_of(reply: Any) -> str | None:
    return field(reply, "done_reason")


def tool_calls_of(reply: Any) -> list[dict[str, Any]]:
    """Tool calls as ``[{"id", "name", "arguments": <json str>}]``.

    Accepts the OpenAI wire form (``{"id", "function": {"name", "arguments"}}``)
    and Ollama's (arguments already a dict).
    """

    calls = field(message_of(reply), "tool_calls") or []
    result: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        function = field(call, "function") or call
        name = field(function, "name")
        if not name:
            continue
        arguments = field(function, "arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False)
        result.append(
            {
                "id": field(call, "id") or f"call_{index}",
                "name": name,
                "arguments": arguments,
            }
        )
    return result


def strip_json_fence(content: str) -> str:
    """Drop a Markdown code fence some models wrap around JSON."""

    content = (content or "").strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```")
        content = content.removesuffix("```")
    return content.strip()
