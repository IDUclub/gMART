"""Unit tests for the urban-data QA A2A surface — agent card, request parsing,
event→A2A mapping, lifecycle. Mirrors test_dvd_a2a.py; adds coverage for
feature_collection/token_expired/pipeline_suspended, which this agent's authenticated
Urban MCP client can actually hit (unlike DVD's unauthenticated one)."""

from __future__ import annotations

import pytest

from src.agents.a2a.task_store import A2ATaskStore
from src.agents.a2a.urban_data_agent import UrbanDataA2AAgent
from src.agents.a2a.urban_data_executor import UrbanDataAgentExecutor
from src.agents.common.exceptions.a2a_exceptions import A2AInvalidParamsError


class FakeQaService:
    """Minimal stand-in for UrbanDataQaService: yields a fixed list of pipeline events."""

    def __init__(self, events: list[dict]) -> None:
        self._events = events
        self.calls: list[dict] = []

    async def run_urban_data_qa_pipeline(self, **kwargs):
        self.calls.append(kwargs)
        for event in self._events:
            yield event


def _executor(service=None) -> UrbanDataAgentExecutor:
    return UrbanDataAgentExecutor(service, A2ATaskStore())


# ---------------------------------------------------------------------------
# Agent card
# ---------------------------------------------------------------------------
def test_agent_card_shape():
    card = UrbanDataA2AAgent().get_agent_card("http://host:80")
    assert card["name"] == "urban-data-qa-agent"
    assert card["url"] == "http://host:80/urban-data/a2a"
    assert card["capabilities"]["streaming"] is True
    assert card["skills"][0]["id"] == "answer-urban-data-questions"


def test_agent_card_relative_url_without_base():
    assert UrbanDataA2AAgent().get_agent_card("")["url"] == "/urban-data/a2a"


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------
def test_prepare_execution_extracts_text_and_metadata():
    out = _executor()._prepare_execution(
        {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "Какие территории в проекте?"}],
                "metadata": {
                    "model": "m2",
                    "temperature": 0.3,
                    "scenario_id": 5,
                    "chat_id": "c9",
                },
            }
        }
    )
    assert out["user_query"] == "Какие территории в проекте?"
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
    assert out["model"] == UrbanDataAgentExecutor.DEFAULT_MODEL
    # scenario_id is optional for this agent (unlike Provision, where it's required).
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
            {"type": "status", "content": {"status": "executing", "text": "ищу"}},
        )
        assert ev["kind"] == "status-update" and ev["final"] is False

    def test_chunk_maps_to_text_artifact(self):
        ex = self._ex_with_task()
        ev = ex._pipeline_item_to_event(
            "t1", "c1", {"type": "chunk", "content": {"text": "hi", "done": False}}
        )
        assert ev["kind"] == "artifact-update"
        assert ev["artifact"]["artifactId"] == "urban-data-agent-text"

    def test_empty_chunk_is_dropped(self):
        ex = self._ex_with_task()
        ev = ex._pipeline_item_to_event(
            "t1", "c1", {"type": "chunk", "content": {"text": "", "done": True}}
        )
        assert ev is None

    def test_feature_collection_maps_to_geojson_artifact(self):
        ex = self._ex_with_task()
        fc = {"type": "FeatureCollection", "features": []}
        ev = ex._pipeline_item_to_event(
            "t1",
            "c1",
            {
                "type": "feature_collection",
                "content": {"name": "GetTerritories", "feature_collection": fc},
            },
        )
        assert ev["kind"] == "artifact-update"
        assert ev["artifact"]["artifactId"] == "geojson-getterritories"
        assert ev["artifact"]["parts"][0]["data"] == fc

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

    def test_token_expired_maps_to_non_final_status_update(self):
        ex = self._ex_with_task()
        ev = ex._pipeline_item_to_event(
            "t1", "c1", {"type": "token_expired", "content": {"message": "expired"}}
        )
        assert ev["kind"] == "status-update" and ev["final"] is False

    def test_pipeline_suspended_maps_to_final_failed_status_update(self):
        ex = self._ex_with_task()
        ev = ex._pipeline_item_to_event(
            "t1",
            "c1",
            {"type": "pipeline_suspended", "content": {"message": "suspended"}},
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
    service = FakeQaService(
        [
            {"type": "status", "content": {"status": "executing", "text": "ищу"}},
            {"type": "chunk", "content": {"text": "Ответ", "done": False}},
            {"type": "chunk", "content": {"text": "", "done": True}},
        ]
    )
    ex = UrbanDataAgentExecutor(service, A2ATaskStore())
    params = {"message": {"role": "user", "parts": [{"type": "text", "text": "q"}]}}

    events = [
        e async for e in ex.stream(params, urban_data_mcp_client=object(), token="t")
    ]

    assert events[0]["kind"] == "task"
    assert any(e.get("kind") == "artifact-update" for e in events)
    assert any(
        e.get("kind") == "status-update"
        and e.get("final")
        and e["status"]["state"] == "completed"
        for e in events
    )


async def test_execute_returns_completed_task_with_artifact():
    service = FakeQaService(
        [{"type": "chunk", "content": {"text": "A", "done": False}}]
    )
    ex = UrbanDataAgentExecutor(service, A2ATaskStore())
    task = await ex.execute(
        {"message": {"role": "user", "parts": [{"type": "text", "text": "q"}]}},
        urban_data_mcp_client=object(),
        token="t",
    )
    assert task["status"]["state"] == "completed"
    assert task["artifacts"]


async def test_stream_disables_history_persistence():
    # A2A runs must leave no trace in ChatStorage: chat_id is forwarded only for
    # read-only history context, persist_history is always False.
    service = FakeQaService(
        [{"type": "chunk", "content": {"text": "A", "done": False}}]
    )
    ex = UrbanDataAgentExecutor(service, A2ATaskStore())
    params = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": "q"}],
            "metadata": {"chat_id": "c9"},
        }
    }

    async for _ in ex.stream(params, urban_data_mcp_client=object(), token="t"):
        pass

    (call,) = service.calls
    assert call["persist_history"] is False
    assert call["chat_id"] == "c9"


async def test_stream_suspended_stops_without_completed_status():
    service = FakeQaService(
        [
            {"type": "chunk", "content": {"text": "A", "done": False}},
            {
                "type": "pipeline_suspended",
                "content": {"message": "token not refreshed in time"},
            },
        ]
    )
    ex = UrbanDataAgentExecutor(service, A2ATaskStore())
    params = {"message": {"role": "user", "parts": [{"type": "text", "text": "q"}]}}

    events = [
        e async for e in ex.stream(params, urban_data_mcp_client=object(), token="t")
    ]

    assert not any(
        e.get("kind") == "status-update" and e["status"]["state"] == "completed"
        for e in events
    )
    assert any(
        e.get("kind") == "status-update"
        and e.get("final")
        and e["status"]["state"] == "failed"
        for e in events
    )


async def test_stream_failure_emits_failed_status():
    class BoomService:
        async def run_urban_data_qa_pipeline(self, **kwargs):
            raise RuntimeError("pipeline boom")
            yield  # pragma: no cover

    ex = UrbanDataAgentExecutor(BoomService(), A2ATaskStore())
    params = {"message": {"role": "user", "parts": [{"type": "text", "text": "q"}]}}

    events = [
        e async for e in ex.stream(params, urban_data_mcp_client=object(), token="t")
    ]

    assert any(
        e.get("kind") == "status-update"
        and e.get("final")
        and e["status"]["state"] == "failed"
        for e in events
    )
