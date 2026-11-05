"""A2A protocol types, from the official ``a2a-sdk`` (protocol v0.3).

gMART serves A2A v0.3 JSON-RPC; the SDK's v0.3 models are the reference the
responses are already validated against (``tests/integration/test_a2a_sdk_validation.py``),
so agent cards are built from them and task states use their enum.
"""

from __future__ import annotations

from typing import Any

from a2a.compat.v0_3.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentSkill,
    TaskState,
)

__all__ = [
    "AgentCapabilities",
    "AgentCard",
    "AgentExtension",
    "AgentSkill",
    "TaskState",
    "TERMINAL_STATES",
    "card_to_dict",
]

# States after which a task accepts no further updates.
TERMINAL_STATES = frozenset({TaskState.completed, TaskState.failed, TaskState.canceled})


def card_to_dict(card: AgentCard) -> dict[str, Any]:
    """The card as served: camelCase keys, unset optional fields left out."""

    return card.model_dump(mode="json", by_alias=True, exclude_none=True)
