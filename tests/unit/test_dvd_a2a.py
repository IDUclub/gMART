"""Unit tests for the A2A surface — agent card, request parsing, event→A2A mapping, lifecycle."""

from __future__ import annotations

import pytest

from src.agents.a2a.dvd_agent import DocumentQaA2AAgent
from src.agents.a2a.dvd_executor import DocumentQaAgentExecutor
from src.agents.a2a.task_store import A2ATaskStore
from src.agents.common.exceptions.a2a_exceptions import A2AInvalidParamsError


class FakeRag:
    """Minimal stand-in for DvdRagService: yields a fixed list of pipeline events."""

    def __init__(self, events: list[dict]) -> None:
        self._events = events
        self.calls: list[dict] = []

    async def run_document_qa_pipeline(self, **kwargs):
        self.calls.append(kwargs)
        for event in self._events:
            yield event


def _executor(rag=None) -> DocumentQaAgentExecutor:
    return DocumentQaAgentExecutor(rag, A2ATaskStore())


# ---------------------------------------------------------------------------
# Agent card
# ---------------------------------------------------------------------------
def test_agent_card_shape():
    card = DocumentQaA2AAgent().get_agent_card("http://host:80")
    assert card["name"] == "document-qa-agent"
    assert card["url"] == "http://host:80/documents/a2a"
    assert card["capabilities"]["streaming"] is True
    assert card["skills"][0]["id"] == "answer-normative-questions"


def test_agent_card_relative_url_without_base():
    assert DocumentQaA2AAgent().get_agent_card("")["url"] == "/documents/a2a"


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------
def test_prepare_execution_extracts_text_and_metadata():
    out = _executor()._prepare_execution(
        {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "Какие нормы?"}],
                "metadata": {
                    "model": "m2",
                    "temperature": 0.3,
                    "scenario_id": 5,
                    "chat_id": "c9",
                },
            }
        }
    )
    assert out["user_query"] == "Какие нормы?"
    assert out["model"] == "m2"
    assert out["temperature"] == 0.3
    assert out["scenario_id"] == 5
    assert out["chat_id"] == "c9"


def test_prepare_execution_requires_text():
    with pytest.raises(A2AInvalidParamsError):
        _executor()._prepare_execution({"message": {"role": "user", "parts": []}})


def test_prepare_execution_applies_defaults():
    out = _executor()._prepare_execution(
        {"message": {"role": "user", "parts": [{"type": "text", "text": "q"}]}}
    )
    assert out["model"] == DocumentQaAgentExecutor.DEFAULT_MODEL
    assert out["scenario_id"] is None
    assert out["chat_id"] is None


# ---------------------------------------------------------------------------
# Pipeline event → A2A event mapping
# ---------------------------------------------------------------------------
class TestEventMapping:
    def _ex_with_task(self):
        ex = _executor()
        ex.task_store.create_task("t1", "c1", {"role": "user", "parts": []}, {})
        return ex

    def test_status_maps_to_working_status_update(self):
        ex = self._ex_with_task()
        ev = ex._pipeline_item_to_event(
            "t1",
            "c1",
            {"type": "status", "content": {"status": "searching", "text": "ищу"}},
        )
        assert ev["kind"] == "status-update" and ev["final"] is False

    def test_chunk_maps_to_per_iteration_artifact(self):
        ex = self._ex_with_task()
        ev = ex._pipeline_item_to_event(
            "t1",
            "c1",
            {"type": "chunk", "content": {"text": "hi", "done": False, "iteration": 2}},
        )
        assert ev["kind"] == "artifact-update"
        assert ev["artifact"]["artifactId"] == "document-qa-answer-2"

    def test_empty_chunk_is_dropped(self):
        ex = self._ex_with_task()
        ev = ex._pipeline_item_to_event(
            "t1",
            "c1",
            {"type": "chunk", "content": {"text": "", "done": True, "iteration": 1}},
        )
        assert ev is None

    def test_warning_maps_to_status_update(self):
        ex = self._ex_with_task()
        ev = ex._pipeline_item_to_event(
            "t1", "c1", {"type": "warning", "content": {"message": "no project id"}}
        )
        assert ev["kind"] == "status-update" and ev["final"] is False

    def test_error_maps_to_final_status_update(self):
        ex = self._ex_with_task()
        ev = ex._pipeline_item_to_event(
            "t1", "c1", {"type": "error", "content": {"message": "boom"}}
        )
        assert ev["kind"] == "status-update" and ev["final"] is True

    @pytest.mark.parametrize(
        "internal", ["tool_call", "service_event", "pipeline_started"]
    )
    def test_internal_events_not_surfaced(self, internal):
        ex = self._ex_with_task()
        assert (
            ex._pipeline_item_to_event("t1", "c1", {"type": internal, "content": {}})
            is None
        )


# ---------------------------------------------------------------------------
# Executor lifecycle
# ---------------------------------------------------------------------------
async def test_stream_emits_task_then_terminal_completed():
    rag = FakeRag(
        [
            {"type": "status", "content": {"status": "searching", "text": "ищу"}},
            {
                "type": "chunk",
                "content": {"text": "Ответ", "done": False, "iteration": 1},
            },
            {"type": "chunk", "content": {"text": "", "done": True, "iteration": 1}},
        ]
    )
    ex = DocumentQaAgentExecutor(rag, A2ATaskStore())
    params = {"message": {"role": "user", "parts": [{"type": "text", "text": "q"}]}}

    events = [e async for e in ex.stream(params, dvd_mcp_client=object(), token="t")]

    assert events[0]["kind"] == "task"
    assert any(e.get("kind") == "artifact-update" for e in events)
    assert any(
        e.get("kind") == "status-update"
        and e.get("final")
        and e["status"]["state"] == "completed"
        for e in events
    )


async def test_execute_returns_completed_task_with_artifact():
    rag = FakeRag(
        [{"type": "chunk", "content": {"text": "A", "done": False, "iteration": 1}}]
    )
    ex = DocumentQaAgentExecutor(rag, A2ATaskStore())
    task = await ex.execute(
        {"message": {"role": "user", "parts": [{"type": "text", "text": "q"}]}},
        dvd_mcp_client=object(),
        token="t",
    )
    assert task["status"]["state"] == "completed"
    assert task["artifacts"]


async def test_stream_disables_history_persistence():
    # A2A runs must leave no trace in ChatStorage: chat_id is forwarded only for
    # read-only history context, persist_history is always False.
    rag = FakeRag(
        [{"type": "chunk", "content": {"text": "A", "done": False, "iteration": 1}}]
    )
    ex = DocumentQaAgentExecutor(rag, A2ATaskStore())
    params = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": "q"}],
            "metadata": {"chat_id": "c9"},
        }
    }

    async for _ in ex.stream(params, dvd_mcp_client=object(), token="t"):
        pass

    (call,) = rag.calls
    assert call["persist_history"] is False
    assert call["chat_id"] == "c9"


async def test_stream_failure_emits_failed_status():
    class BoomRag:
        async def run_document_qa_pipeline(self, **kwargs):
            raise RuntimeError("pipeline boom")
            yield  # pragma: no cover

    ex = DocumentQaAgentExecutor(BoomRag(), A2ATaskStore())
    params = {"message": {"role": "user", "parts": [{"type": "text", "text": "q"}]}}

    events = [e async for e in ex.stream(params, dvd_mcp_client=object(), token="t")]

    assert any(
        e.get("kind") == "status-update"
        and e.get("final")
        and e["status"]["state"] == "failed"
        for e in events
    )
