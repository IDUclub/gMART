"""Unit tests for the DVD HTTP layer — SSE QA endpoint and the A2A endpoints.

A fresh FastAPI app is built from the routers with every per-request dependency overridden by
a fake, so the routing / DTO parsing / SSE serialization / DI wiring are exercised in isolation
(no Ollama / IDU_DVD MCP / Redis / ChatStorage).
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents.common.auth.auth import optional_bearer_token, verify_bearer_token
from src.agents.common.middlewares.exception_handler import ExceptionHandlerMiddleware
from src.agents.dependencies.dependencies import (
    get_dvd_a2a_service,
    get_dvd_mcp_client,
    get_dvd_rag_service,
)
from src.agents.routers.dvd_a2a_controller import dvd_a2a_router
from src.agents.routers.dvd_controller import dvd_router
from src.agents.services.dvd_a2a_service import DocumentQaA2AService


class FakeStateStore:
    """Serves the pipeline states the public-access guard inspects on reconnect."""

    def __init__(self, states: dict[str, dict] | None = None) -> None:
        self.states = states or {}

    async def get_state(self, request_id: str) -> dict | None:
        return self.states.get(request_id)


class FakeRagService:
    """Yields a small, valid event sequence regardless of inputs."""

    def __init__(self, states: dict[str, dict] | None = None) -> None:
        self.state_store = FakeStateStore(states)
        self.calls: list[dict] = []

    async def run_document_qa_pipeline(self, model=None, **kwargs):
        self.calls.append(kwargs)
        yield {"type": "pipeline_started", "content": {"request_id": "rid-1"}}
        yield {"type": "status", "content": {"status": "searching", "text": "ищу"}}
        yield {
            "type": "chunk",
            "content": {"text": "Ответ", "done": True, "iteration": 1},
        }


def _parse_sse(text: str) -> list[dict]:
    return [
        json.loads(line[len("data:") :].strip())
        for line in text.splitlines()
        if line.startswith("data:")
    ]


def _build_client(token: str | None, rag_service: FakeRagService) -> TestClient:
    app = FastAPI()
    app.include_router(dvd_router)
    app.include_router(dvd_a2a_router)
    app.add_middleware(ExceptionHandlerMiddleware)
    app.dependency_overrides[verify_bearer_token] = lambda: "test-token"
    app.dependency_overrides[optional_bearer_token] = lambda: token
    app.dependency_overrides[get_dvd_mcp_client] = lambda: object()
    app.dependency_overrides[get_dvd_rag_service] = lambda: rag_service
    app.dependency_overrides[get_dvd_a2a_service] = lambda: DocumentQaA2AService(
        rag_service
    )
    return TestClient(app)


@pytest.fixture
def rag_service():
    return FakeRagService()


@pytest.fixture
def client(rag_service):
    """Authorized caller — a bearer token reached the front door."""

    with _build_client("test-token", rag_service) as c:
        yield c


@pytest.fixture
def anonymous_client(rag_service):
    """Public caller — no Authorization header at all."""

    with _build_client(None, rag_service) as c:
        yield c


class TestQaStream:
    def test_streams_events_as_sse(self, client):
        resp = client.get("/documents/qa/stream", params={"request": "Какие нормы?"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(resp.text)
        types = [e["type"] for e in events]
        assert types[0] == "pipeline_started"
        assert "chunk" in types

        chunk = next(e for e in events if e["type"] == "chunk")
        assert chunk["content"]["text"] == "Ответ"
        assert chunk["content"]["iteration"] == 1  # validated through DvdResponse

    def test_request_query_param_is_required(self, client):
        # `request` has no default in SimpleRequestDTO → 422 when missing
        assert client.get("/documents/qa/stream").status_code == 422

    def test_persists_history_for_an_authorized_request(self, client, rag_service):
        client.get("/documents/qa/stream", params={"request": "Какие нормы?"})
        assert rag_service.calls[0]["persist_history"] is True


class TestPublicQaStream:
    """The shared index is answerable without a bearer token; user scope is not."""

    def test_anonymous_question_is_answered(self, anonymous_client, rag_service):
        resp = anonymous_client.get(
            "/documents/qa/stream", params={"request": "Какие нормы?"}
        )
        assert resp.status_code == 200

        events = _parse_sse(resp.text)
        assert [e["type"] for e in events][0] == "pipeline_started"
        # No user JWT → nothing may be written to Chat Storage.
        assert rag_service.calls[0]["persist_history"] is False
        assert rag_service.calls[0]["token"] is None

    def test_anonymous_scenario_id_is_rejected(self, anonymous_client):
        resp = anonymous_client.get(
            "/documents/qa/stream",
            params={"request": "Какие нормы?", "scenario_id": 772},
        )
        assert resp.status_code == 401

    def test_anonymous_chat_id_is_rejected(self, anonymous_client):
        resp = anonymous_client.get(
            "/documents/qa/stream",
            params={"request": "Какие нормы?", "chat_id": "c" * 36},
        )
        assert resp.status_code == 401

    def test_anonymous_reconnect_to_a_user_pipeline_is_rejected(self, rag_service):
        request_id = "r" * 36
        rag_service.state_store.states[request_id] = {"scenario_id": 772}
        with _build_client(None, rag_service) as anonymous_client:
            resp = anonymous_client.get(
                "/documents/qa/stream",
                params={"request": "Какие нормы?", "request_id": request_id},
            )
        assert resp.status_code == 401

    def test_anonymous_reconnect_to_a_public_pipeline_is_allowed(self, rag_service):
        request_id = "r" * 36
        rag_service.state_store.states[request_id] = {
            "scenario_id": None,
            "chat_id": None,
        }
        with _build_client(None, rag_service) as anonymous_client:
            resp = anonymous_client.get(
                "/documents/qa/stream",
                params={"request": "Какие нормы?", "request_id": request_id},
            )
        assert resp.status_code == 200


class TestA2A:
    def test_agent_card_endpoint(self, client):
        resp = client.get("/documents/.well-known/agent-card.json")
        assert resp.status_code == 200
        assert resp.json()["name"] == "document-qa-agent"

    def test_tasks_list_returns_empty(self, client):
        resp = client.post(
            "/documents/a2a",
            json={"jsonrpc": "2.0", "id": "1", "method": "tasks/list", "params": {}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["jsonrpc"] == "2.0"
        assert body["result"] == []

    def test_invalid_jsonrpc_returns_error(self, client):
        resp = client.post(
            "/documents/a2a",
            json={"jsonrpc": "1.0", "id": "1", "method": "tasks/list", "params": {}},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()
