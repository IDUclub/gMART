"""Integration: exercise the real ``/restriction/a2a`` (and deprecated ``/a2a``) HTTP endpoints
end to end — router mounting, request-DTO validation, dependency injection, bearer-token auth,
JSON-RPC dispatch and SSE streaming — through the actual FastAPI app wiring.

Only the restriction pipeline itself is faked (mirrors the approach in
``tests/unit/test_dvd_routers.py`` for the document-QA agent), so no Ollama / IDU MCP / Urban
API calls happen. Complements ``tests/integration/test_a2a_sdk_validation.py`` (calls
``A2AService`` directly, bypassing HTTP/DI) and ``tests/unit/test_a2a_spec_compliance.py``
(service-level unit coverage).
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents.a2a.task_store import A2ATaskStore
from src.agents.common.auth.auth import verify_bearer_token
from src.agents.dependencies.dependencies import get_a2a_service, get_idu_mcp_client
from src.agents.routers.a2a_controller import a2a_router, restriction_a2a_router
from src.agents.services.a2a_service import A2AService

pytestmark = pytest.mark.integration

FEATURE_COLLECTION = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [30.3, 59.9]},
            "properties": {"name": "school"},
        }
    ],
}


class FakeRestrictionService:
    """Representative restriction pipeline: status, text chunk, GeoJSON layer."""

    async def run_restriction_execution_pipline(self, **kwargs):
        yield {"type": "status", "content": {"text": "working"}}
        yield {"type": "chunk", "content": {"text": "Зона ограничения построена."}}
        yield {
            "type": "feature_collection",
            "content": {"name": "schools", "feature_collection": FEATURE_COLLECTION},
        }


def _parse_sse(text: str) -> list[dict]:
    return [
        json.loads(line[len("data:") :].strip())
        for line in text.splitlines()
        if line.startswith("data:")
    ]


def _send_payload(text: str, task_id: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": task_id,
        "method": "message/send",
        "params": {
            "id": task_id,
            "message": {
                "role": "user",
                "parts": [
                    {"kind": "data", "data": {"scenario_id": 772}},
                    {"kind": "text", "text": text},
                ],
            },
        },
    }


@pytest.fixture
def client():
    """Full router wiring (canonical + deprecated) with the pipeline faked but auth real.

    The A2AService (and its task store) is built once and reused across requests within a
    test — a fresh instance per request would give every ``tasks/get`` an empty store.
    """
    app = FastAPI()
    app.include_router(restriction_a2a_router)
    app.include_router(a2a_router)
    a2a_service = A2AService(FakeRestrictionService(), task_store=A2ATaskStore())
    app.dependency_overrides[verify_bearer_token] = lambda: "test-token"
    app.dependency_overrides[get_idu_mcp_client] = lambda: object()
    app.dependency_overrides[get_a2a_service] = lambda: a2a_service
    with TestClient(app) as c:
        yield c


@pytest.fixture
def unauthenticated_client():
    """Same wiring, but bearer-token auth is NOT overridden — exercises the real auth gate."""
    app = FastAPI()
    app.include_router(restriction_a2a_router)
    app.include_router(a2a_router)
    app.dependency_overrides[get_a2a_service] = lambda: A2AService(
        FakeRestrictionService(), task_store=A2ATaskStore()
    )
    with TestClient(app) as c:
        yield c


class TestAgentCardDiscovery:
    def test_canonical_agent_card(self, client):
        resp = client.get("/restriction/.well-known/agent-card.json")
        assert resp.status_code == 200
        card = resp.json()
        assert card["name"] == "restriction-creation-agent"
        assert card["protocolVersion"] == "0.3.0"
        assert card["preferredTransport"] == "JSONRPC"
        assert card["url"].endswith("/a2a")

    def test_legacy_agent_card_matches_canonical(self, client):
        canonical = client.get("/restriction/.well-known/agent-card.json").json()
        legacy = client.get("/.well-known/agent.json").json()
        assert canonical == legacy


class TestMessageSend:
    def test_send_completes_with_geojson_artifact(self, client):
        resp = client.post(
            "/restriction/a2a",
            json=_send_payload("построй зону вокруг школ 200 м", "it-send-1"),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["jsonrpc"] == "2.0"
        task = body["result"]
        assert task["status"]["state"] == "completed"
        assert task["history"], "history must include the echoed user message"
        for message in task["history"]:
            assert message.get("messageId")

        geojson_parts = [
            part
            for artifact in task["artifacts"]
            for part in artifact["parts"]
            if part.get("kind") == "data"
        ]
        assert geojson_parts, "expected a GeoJSON data artifact"
        assert geojson_parts[0]["data"]["type"] == "FeatureCollection"

    def test_deprecated_root_endpoint_behaves_like_canonical(self, client):
        resp = client.post(
            "/a2a", json=_send_payload("построй зону вокруг школ", "it-send-legacy")
        )
        assert resp.status_code == 200
        assert resp.json()["result"]["status"]["state"] == "completed"

    def test_missing_scenario_id_returns_json_rpc_invalid_params(self, client):
        payload = {
            "jsonrpc": "2.0",
            "id": "it-send-2",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "build zone"}],
                }
            },
        }
        resp = client.post("/restriction/a2a", json=payload)
        assert resp.status_code == 200
        assert resp.json()["error"]["code"] == -32602


class TestTaskLookup:
    def test_get_task_after_send_round_trips(self, client):
        send_resp = client.post(
            "/restriction/a2a",
            json=_send_payload("построй зону вокруг школ", "it-gettask-1"),
        )
        assert send_resp.status_code == 200

        get_resp = client.post(
            "/restriction/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "it-gettask-2",
                "method": "tasks/get",
                "params": {"id": "it-gettask-1"},
            },
        )
        assert get_resp.status_code == 200
        task = get_resp.json()["result"]
        assert task["id"] == "it-gettask-1"
        assert task["status"]["state"] == "completed"


class TestStreaming:
    def test_message_stream_emits_task_then_completed_terminal_event(self, client):
        payload = _send_payload("построй зону вокруг школ", "it-stream-1")
        payload["method"] = "message/stream"
        resp = client.post("/restriction/a2a", json=payload)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(resp.text)
        assert events, "expected at least one SSE event"
        # A2A 0.3: first frame is the Task itself, follow-ups are flat
        # kind-discriminated status-update / artifact-update events.
        assert events[0]["result"]["kind"] == "task"
        terminal = [
            e
            for e in events
            if e.get("result", {}).get("kind") == "status-update"
            and e["result"].get("final")
            and e["result"]["status"]["state"] == "completed"
        ]
        assert terminal, "expected a terminal completed status-update event"


class TestAuth:
    def test_missing_bearer_token_is_rejected(self, unauthenticated_client):
        resp = unauthenticated_client.post(
            "/restriction/a2a",
            json=_send_payload("построй зону", "it-noauth-1"),
        )
        assert resp.status_code in (401, 403)
