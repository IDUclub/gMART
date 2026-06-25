"""Shared helpers for the DVD-agent tests: in-memory doubles, JSON builders, event utils.

Kept free of any ``src`` import so it is safe to import before the environment is configured.
"""

from __future__ import annotations

import json
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# In-memory doubles
# ---------------------------------------------------------------------------
class _Part:
    """Mimics one ollama streaming ChatResponse part."""

    def __init__(self, content: str, done: bool) -> None:
        self.message = SimpleNamespace(content=content)
        self.done = done


class FakeLlmClient:
    """Stand-in for the ollama AsyncClient exposed as ``self.llm_client``.

    - non-stream ``chat`` (planner / critic): returns ``{"message": {"content": <json>}}``
      popped from ``json_responses``.
    - stream ``chat`` (answer drafting): returns an async iterator of parts built from the
      next string in ``answer_texts``.
    - ``generate`` (chat-title generation): returns ``SimpleNamespace(response=self.title)``.
    """

    def __init__(self) -> None:
        self.json_responses: list[str] = []
        self.answer_texts: list[str] = []
        self.title = "Тестовый чат"
        self.chat_calls: list[SimpleNamespace] = []

    async def chat(
        self, model=None, messages=None, options=None, stream=False, **kwargs
    ):
        self.chat_calls.append(
            SimpleNamespace(
                model=model, messages=messages, options=options, stream=stream
            )
        )
        if stream:
            text = self.answer_texts.pop(0) if self.answer_texts else ""
            return self._stream(text)
        content = self.json_responses.pop(0) if self.json_responses else "{}"
        return {"message": {"content": content}}

    @staticmethod
    async def _stream(text: str):
        # two parts to exercise chunk accumulation; the per-part done flag is ignored by
        # the service (it emits its own terminal done chunk after the critic accepts).
        if text:
            mid = len(text) // 2
            yield _Part(text[:mid], done=False)
            yield _Part(text[mid:], done=True)
        else:
            yield _Part("", done=True)

    async def generate(self, model=None, prompt=None, stream=False, **kwargs):
        return SimpleNamespace(response=self.title)


class FakeDvdMcpClient:
    """Stand-in for ``DvdMcpClient``: records searches and returns programmed hits."""

    def __init__(
        self,
        hits_per_call: list[list[dict]] | None = None,
        default_hits: list[dict] | None = None,
    ) -> None:
        self.search_calls: list[SimpleNamespace] = []
        self._hits_per_call = list(hits_per_call) if hits_per_call is not None else None
        self.default_hits = (
            default_hits
            if default_hits is not None
            else [
                {
                    "name": "СП 42.13330.2016",
                    "version": "ред. 2018",
                    "numbering": "7.5",
                    "breadcrumb": "Раздел 7. Озеленение",
                    "text": "Площадь озеленённых территорий — не менее 6 м² на человека.",
                }
            ]
        )

    @staticmethod
    def tool_name_for_kind(kind: str) -> str:
        return {
            "text": "search_texts",
            "table": "search_tables",
            "all": "search_all",
        }.get(str(kind), "search_all")

    async def search(self, query, kind="all", limit=10, context_height=0, **kwargs):
        self.search_calls.append(
            SimpleNamespace(
                query=query,
                kind=str(kind),
                limit=limit,
                context_height=context_height,
            )
        )
        if self._hits_per_call is not None:
            hits = self._hits_per_call.pop(0) if self._hits_per_call else []
        else:
            hits = self.default_hits
        return {"count": len(hits), "hits": hits}


class FakeUrbanApiClient:
    """Stand-in for ``UrbanApiClient.get_project_by_scenario``."""

    def __init__(
        self, project_id: int = 4242, raise_exc: Exception | None = None
    ) -> None:
        self.project_id = project_id
        self.raise_exc = raise_exc
        self.calls: list[tuple] = []

    async def get_project_by_scenario(self, token, scenario_id):
        self.calls.append((token, scenario_id))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.project_id


# ---------------------------------------------------------------------------
# JSON builders (what the planner / critic LLM would return)
# ---------------------------------------------------------------------------
def plan_json(
    search_query: str = "нормы озеленения",
    kind: str = "all",
    limit: int = 5,
    context_height: int = 1,
) -> str:
    return json.dumps(
        {
            "search_query": search_query,
            "kind": kind,
            "limit": limit,
            "context_height": context_height,
        },
        ensure_ascii=False,
    )


def verdict_json(
    satisfied: bool = True,
    critique: str = "",
    refined_search_query: str | None = None,
) -> str:
    return json.dumps(
        {
            "satisfied": satisfied,
            "critique": critique,
            "refined_search_query": refined_search_query,
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Event inspection
# ---------------------------------------------------------------------------
def types_of(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


def events_of_type(events: list[dict], event_type: str) -> list[dict]:
    return [event for event in events if event["type"] == event_type]


def statuses(events: list[dict], status: str | None = None) -> list[str]:
    out = [
        event["content"]["text"]
        for event in events
        if event["type"] == "status"
        and (status is None or event["content"]["status"] == status)
    ]
    return out


def answer_text(events: list[dict]) -> str:
    """Concatenate the text of every chunk event (across all iterations)."""
    return "".join(
        event["content"]["text"] for event in events if event["type"] == "chunk"
    )


def final_chunk(events: list[dict]) -> dict | None:
    """The terminal chunk (done=True), if present."""
    done = [
        event["content"]
        for event in events
        if event["type"] == "chunk" and event["content"].get("done")
    ]
    return done[-1] if done else None
