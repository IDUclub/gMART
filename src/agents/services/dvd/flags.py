"""Switches of the document-QA latency optimisations.

Each optimisation stage reads its own ``DVD_*`` variable on every use, so it can be
turned off (A/B comparison, rollback) without a release. All default to on.
"""

import os

_OFF = {"0", "false", "no", "off"}


def enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in _OFF


ADAPTIVE_CRITIC = "DVD_ADAPTIVE_CRITIC"
PLANNER_FAST_PATH = "DVD_PLANNER_FAST_PATH"
DEFERRED_CHAT_SETUP = "DVD_DEFERRED_CHAT_SETUP"
SMALL_FIRST_RETRIEVAL = "DVD_SMALL_FIRST_RETRIEVAL"
PERSISTENT_MCP_SESSION = "DVD_PERSISTENT_MCP_SESSION"
KNOWLEDGE_FALLBACK = "DVD_KNOWLEDGE_FALLBACK"
