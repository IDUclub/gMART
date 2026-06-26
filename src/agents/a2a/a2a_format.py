from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

A2AData = dict[str, Any]

# Stable identifier of the A2A profile extension that declares a project scenario id
# (scenario_id) as a required, discoverable input on the geospatial agents (restriction +
# provision). The URI is an opaque, globally-unique namespace string — it is never fetched
# over HTTP; clients only match it. Bump the trailing version when the contract changes.
SCENARIO_CONTEXT_EXTENSION_URI = (
    "https://github.com/IDUclub/gMART/a2a/extensions/scenario-context/v1"
)


def scenario_context_extension() -> A2AData:
    """
    Function builds the required scenario-context AgentExtension for an AgentCard.
    Returns:
        A2AData: AgentExtension declaration (uri, required flag, JSON Schema params).
    """

    return {
        "uri": SCENARIO_CONTEXT_EXTENSION_URI,
        "description": (
            "Requires a project scenario id (scenario_id) on every incoming message. "
            'Pass it in a DataPart ({"kind": "data", "data": {"scenario_id": 772}}), '
            "in message.metadata under the extension uri, or inline as 'scenario_id=772' "
            "in the message text. Activate via the 'A2A-Extensions' header."
        ),
        "required": True,
        "params": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["scenario_id"],
            "properties": {
                "scenario_id": {
                    "type": "integer",
                    "description": "Urban API project scenario id.",
                }
            },
        },
    }


def utc_now_rfc3339() -> str:
    """
    Function returns the current UTC time as an RFC3339 timestamp with a 'Z' offset.
    Returns:
        str: RFC3339 timestamp, e.g. '2026-06-26T11:48:08.022614Z'.
    """

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def normalize_response(obj: Any) -> Any:
    """
    Function rewrites the legacy ``type`` part discriminator to the A2A 0.3 ``kind``.

    ``python_a2a`` serialises message/artifact parts as ``{"type": ...}``; A2A 0.3 mandates
    ``{"kind": ...}``. Only dict elements of a ``parts`` list are rewritten, so GeoJSON
    payloads nested under a part's ``data`` (which legitimately carry ``type``) stay intact.
    Args:
        obj (Any): Outgoing A2A structure (task, event, list, ...). Mutated in place.
    Returns:
        Any: The same object, with part discriminators normalised.
    """

    if isinstance(obj, dict):
        parts = obj.get("parts")
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict) and "type" in part and "kind" not in part:
                    part["kind"] = part.pop("type")
        for value in obj.values():
            normalize_response(value)
    elif isinstance(obj, list):
        for item in obj:
            normalize_response(item)
    return obj


def sanitized_user_message(
    parts: list[A2AData],
    incoming_message_id: Any = None,
) -> A2AData:
    """
    Function builds a spec-compliant echo of the user message for task history.

    Every A2A Message must carry ``messageId`` and ``kind`` — including the echoed user
    message stored in ``history``. The client's original ``messageId`` is preserved when
    present, otherwise a new one is generated.
    Args:
        parts (list[A2AData]): Sanitized message parts.
        incoming_message_id (Any): Original client messageId, if any.
    Returns:
        A2AData: User message with role, kind, messageId and parts.
    """

    return {
        "role": "user",
        "kind": "message",
        "messageId": str(incoming_message_id) if incoming_message_id else str(uuid4()),
        "parts": parts,
    }


def apply_history_length(task: A2AData, length: Any) -> A2AData:
    """
    Function trims ``task.history`` to the last ``length`` messages.

    Honours the A2A ``configuration.historyLength`` / ``historyLength`` request field:
    ``0`` drops the history entirely, a positive value keeps the most recent N messages,
    and an absent/invalid value leaves the task untouched.
    Args:
        task (A2AData): Serialized A2A task. Mutated in place.
        length (Any): Requested history length.
    Returns:
        A2AData: The same task, with history trimmed when a valid limit is supplied.
    """

    if not isinstance(task, dict) or length is None:
        return task
    try:
        limit = int(length)
    except (TypeError, ValueError):
        return task
    if limit < 0:
        return task
    history = task.get("history")
    if isinstance(history, list):
        task["history"] = history[-limit:] if limit else []
    return task
