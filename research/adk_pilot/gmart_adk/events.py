"""Bridge between ADK events and gMART's pipeline event stream.

The REST/SSE controllers and the A2A executors consume the same stream of
plain dicts (``status``, ``chunk``, ``tool_call``, ``feature_collection``,
``clarification``). A workflow keeps that contract by

* attaching those dicts to its own events: function nodes yield
  :func:`gmart_event`, which carries them in ``Event.custom_metadata``;
* streaming the text of the agents named in ``streamed_agents`` as ``chunk``
  items.

:class:`LegacyEventAdapter` turns the ADK event stream back into those dicts.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from google.adk import Event

GMART_KEY = "gmart"


def gmart_event(*items: dict[str, Any], **kwargs: Any) -> Event:
    """An ADK event carrying gMART pipeline items (``kwargs`` go to ``Event``)."""
    return Event(custom_metadata={GMART_KEY: list(items)}, **kwargs)


def chunk(text: str, done: bool) -> dict[str, Any]:
    return {"type": "chunk", "content": {"text": text, "done": done}}


class LegacyEventAdapter:
    """Converts the events of one run into gMART pipeline items."""

    def __init__(self, streamed_agents: Iterable[str]) -> None:
        self.streamed_agents = set(streamed_agents)
        # Node runs that already streamed partial text: their closing
        # aggregate repeats it and must not be sent again.
        self._streamed_runs: set[str] = set()

    def convert(self, event: Event) -> Iterator[dict[str, Any]]:
        metadata = event.custom_metadata or {}
        yield from metadata.get(GMART_KEY, [])

        if event.author not in self.streamed_agents:
            return
        text = _text(event)
        if not text:
            return
        run = event.node_info.path if event.node_info else event.author
        if event.partial:
            self._streamed_runs.add(run)
            yield chunk(text, done=False)
        elif run not in self._streamed_runs:
            # Non-streaming run (StreamingMode.NONE): the aggregate is all we get.
            yield chunk(text, done=False)


def _text(event: Event) -> str:
    if not event.content or not event.content.parts:
        return ""
    return "".join(
        part.text for part in event.content.parts if part.text and not part.thought
    )
