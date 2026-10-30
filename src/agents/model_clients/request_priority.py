"""Per-call vLLM scheduling priorities, with live environment overrides."""

import os
from functools import lru_cache

from loguru import logger

DEFAULT_PRIORITIES = {
    "dvd.planner": -100,
    "dvd.conversation": -100,
    "dvd.context_reducer": -100,
    "dvd.answer": -100,
    "dvd.critic": -100,
    "dvd.revision": -100,
    "norms.planner": -100,
    "norms.answer": -100,
    "norms.review": -100,
    "restriction.plan": -60,
    "restriction.explain": -60,
    "restriction.answer": -60,
    "pzz.inputs": -60,
    "pzz.columns": -60,
    "pzz.answer": -100,
    "provision.plan": -60,
    "provision.answer": -100,
    "scenario.plan": -60,
    "scenario.mapping": -60,
    "scenario.answer": -100,
    "scenario.judge": -100,
    "orchestrator.plan": -100,
    "simple_llm": -100,
    "chat_title": -60,
    "context_summary": 20,
    "compliance.scope": -60,
    "compliance.mapping": -60,
    "project.inputs": -60,
}


@lru_cache(maxsize=None)
def _warn_once(entry: str) -> None:
    logger.warning("Invalid LLM request priority entry {!r}; ignoring it", entry)


def priority_enabled() -> bool:
    return os.getenv("LLM_REQUEST_PRIORITY_ENABLED", "true").strip().lower() not in (
        "false",
        "0",
        "no",
        "off",
    )


def request_priority(key: str | None = None) -> int | None:
    """Override > table default; no key uses OPENAI_REQUEST_PRIORITY. Zero is off."""
    if not priority_enabled():
        return None
    overrides = {}
    for entry in os.getenv("LLM_REQUEST_PRIORITIES", "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, separator, value = entry.partition("=")
        name = name.strip()
        try:
            if not separator or name not in DEFAULT_PRIORITIES:
                raise ValueError
            overrides[name] = int(value.strip())
        except ValueError:
            _warn_once(entry)
    if key is not None:
        if key not in DEFAULT_PRIORITIES:
            _warn_once(key)
            return None
        return overrides.get(key, DEFAULT_PRIORITIES[key]) or None
    value = os.getenv("OPENAI_REQUEST_PRIORITY", "").strip()
    try:
        return (int(value) or None) if value else None
    except ValueError:
        _warn_once("OPENAI_REQUEST_PRIORITY=" + value)
        return None
