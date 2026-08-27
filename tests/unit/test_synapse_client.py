import json

import httpx
import pytest

from src.agents.api_clients.synapse_client import SynapseApiClient


def _jwt(exp: int = 4_102_444_800) -> str:
    import base64

    payload = (
        base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    )
    return f"header.{payload}.signature"


@pytest.mark.asyncio
async def test_client_logs_in_and_uses_configured_project_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/auth/login":
            return httpx.Response(
                200,
                json={"access_token": _jwt(), "refresh_token": "refresh"},
            )
        if request.url.path == "/api/projects":
            return httpx.Response(
                200, json={"project_id": "p1", "status": "running", "user_prompt": "x"}
            )
        raise AssertionError(request.url)

    http = httpx.AsyncClient(
        base_url="http://synapse.test", transport=httpx.MockTransport(handler)
    )
    client = SynapseApiClient(
        "http://synapse.test",
        "service@example.test",
        "secret",
        workflow_id="wf-id",
        run_config_id="run-id",
        client=http,
    )

    result = await client.create_project("prompt")

    assert result["project_id"] == "p1"
    body = json.loads(requests[-1].content)
    assert body == {
        "user_prompt": "prompt",
        "approval_mode": "auto",
        "workflow_id": "wf-id",
        "run_config_id": "run-id",
    }
    assert requests[-1].headers["authorization"].startswith("Bearer ")
    await http.aclose()


@pytest.mark.asyncio
async def test_sse_parser_supports_multiline_data() -> None:
    async def lines():
        for line in (
            "event: project_completed",
            "id: event-1",
            'data: {"status":',
            'data: "completed"}',
            "",
        ):
            yield line

    events = [event async for event in SynapseApiClient._parse_sse(lines())]

    assert events[0].event == "project_completed"
    assert events[0].event_id == "event-1"
    assert events[0].data == {"status": "completed"}


@pytest.mark.asyncio
async def test_client_refreshes_once_after_project_401() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/api/auth/login":
            return httpx.Response(
                200,
                json={"access_token": _jwt(), "refresh_token": "refresh-1"},
            )
        if request.url.path == "/api/auth/refresh":
            return httpx.Response(
                200,
                json={"access_token": _jwt(), "refresh_token": "refresh-2"},
            )
        if request.url.path == "/api/projects" and calls.count("/api/projects") == 1:
            return httpx.Response(401)
        if request.url.path == "/api/projects":
            return httpx.Response(200, json={"project_id": "p1"})
        raise AssertionError(request.url)

    http = httpx.AsyncClient(
        base_url="http://synapse.test", transport=httpx.MockTransport(handler)
    )
    client = SynapseApiClient(
        "http://synapse.test",
        "service@example.test",
        "secret",
        workflow_id="wf-id",
        run_config_id="run-id",
        client=http,
    )

    assert (await client.create_project("prompt"))["project_id"] == "p1"
    assert calls == [
        "/api/auth/login",
        "/api/projects",
        "/api/auth/refresh",
        "/api/projects",
    ]
    await http.aclose()
