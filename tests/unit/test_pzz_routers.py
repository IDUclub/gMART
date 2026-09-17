import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.dependencies.dependencies import (
    get_pzz_a2a_service,
    get_pzz_mcp_client,
    get_pzz_service,
)
from src.agents.routers.pzz_a2a_controller import pzz_a2a_router
from src.agents.routers.pzz_controller import pzz_router
from src.agents.services.pzz.pzz_a2a_service import PzzA2AService


class Service:
    def __init__(self):
        self.calls = []

    async def run_pzz_pipeline(self, **kwargs):
        self.calls.append(kwargs)
        yield {"type": "chunk", "content": {"text": "Ответ ПЗЗ", "done": True}}


def build_client():
    service = Service()
    app = FastAPI()
    app.include_router(pzz_router)
    app.include_router(pzz_a2a_router)
    app.dependency_overrides[verify_bearer_token] = lambda: "token"
    app.dependency_overrides[get_pzz_service] = lambda: service
    app.dependency_overrides[get_pzz_mcp_client] = lambda: SimpleNamespace(
        api_client=None
    )
    app.dependency_overrides[get_pzz_a2a_service] = lambda: PzzA2AService(service)
    return TestClient(app), service


def test_post_sse_and_input_validation():
    client, service = build_client()
    response = client.post(
        "/pzz/check/stream",
        json={
            "request": "Проверь",
            "scenario_id": 42,
            "inputs": {"mode": "scenario", "year": 2026, "source": "PZZ"},
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    event = json.loads(
        next(
            line[5:] for line in response.text.splitlines() if line.startswith("data:")
        )
    )
    assert event["type"] == "chunk"
    assert service.calls[0]["inputs"].year == 2026
    assert (
        client.post(
            "/pzz/check/stream",
            json={"request": "Проверь", "inputs": {"source": "invented"}},
        ).status_code
        == 422
    )


def test_get_sse_and_public_card():
    client, service = build_client()
    assert (
        client.get(
            "/pzz/check/stream", params={"request": "Проверь", "scenario_id": 42}
        ).status_code
        == 200
    )
    assert service.calls[0]["scenario_id"] == 42
    card = client.get("/pzz/.well-known/agent-card.json").json()
    assert card["url"] == "http://testserver/pzz/a2a"
    assert client.get("/pzz/agent.json").json() == card


def test_upload_requires_configured_rest_url():
    client, _ = build_client()
    assert (
        client.post("/pzz/uploads", files={"file": ("layer.json", b"{}")}).status_code
        == 503
    )


def test_a2a_http_stream_is_json_rpc_sse():
    client, _ = build_client()
    response = client.post(
        "/pzz/a2a",
        json={
            "jsonrpc": "2.0",
            "id": "a2a-test",
            "method": "message/stream",
            "params": {"message": {"parts": [{"kind": "text", "text": "Проверь"}]}},
        },
    )
    assert response.status_code == 200
    events = [
        json.loads(line[5:])
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]
    assert events[0]["result"]["kind"] == "task"
    assert events[-1]["result"]["status"]["state"] == "completed"
    assert events[-1]["result"]["final"] is True
