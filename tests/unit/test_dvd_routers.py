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

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.dependencies.dependencies import (
    get_dvd_a2a_service,
    get_dvd_mcp_client,
    get_dvd_rag_service,
)
from src.agents.routers.dvd_a2a_controller import dvd_a2a_router
from src.agents.routers.dvd_controller import dvd_router
from src.agents.services.dvd_a2a_service import DocumentQaA2AService


class FakeRagService:
    """Yields a small, valid event sequence regardless of inputs."""

    async def run_document_qa_pipeline(self, model=None, **kwargs):
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


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(dvd_router)
    app.include_router(dvd_a2a_router)
    app.dependency_overrides[verify_bearer_token] = lambda: "test-token"
    app.dependency_overrides[get_dvd_mcp_client] = lambda: object()
    app.dependency_overrides[get_dvd_rag_service] = lambda: FakeRagService()
    app.dependency_overrides[get_dvd_a2a_service] = lambda: DocumentQaA2AService(
        FakeRagService()
    )
    with TestClient(app) as c:
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
