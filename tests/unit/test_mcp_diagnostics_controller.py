"""Unit tests for the MCP diagnostics API used by the embedded UI console."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents.dependencies.dependencies import get_idu_mcp_client
from src.agents.routers.mcp_diagnostics_controller import mcp_diagnostics_router


def _client(mcp_client: AsyncMock) -> TestClient:
    app = FastAPI()
    app.include_router(mcp_diagnostics_router)
    app.dependency_overrides[get_idu_mcp_client] = lambda: mcp_client
    return TestClient(app)


def test_lists_current_mcp_tools():
    mcp_client = AsyncMock()
    mcp_client.load_ollama_tools.return_value = [
        {
            "type": "function",
            "function": {
                "name": "GetServices",
                "description": "Services",
                "parameters": {"type": "object"},
            },
        }
    ]

    response = _client(mcp_client).get("/mcp-diagnostics/tools")

    assert response.status_code == 200
    assert response.json()[0]["function"]["name"] == "GetServices"
    mcp_client.load_ollama_tools.assert_awaited_once_with()


def test_lists_serialized_prompts():
    prompt = SimpleNamespace(
        model_dump=lambda **kwargs: {
            "name": "GetAvailableServices",
            "arguments": [{"name": "scenario_id", "required": True}],
        }
    )
    mcp_client = AsyncMock()
    mcp_client.get_prompts.return_value = [prompt]

    response = _client(mcp_client).get("/mcp-diagnostics/prompts")

    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "GetAvailableServices",
            "arguments": [{"name": "scenario_id", "required": True}],
        }
    ]
    mcp_client.get_prompts.assert_awaited_once_with()


def test_calls_tool_with_arguments_and_meta():
    mcp_client = AsyncMock()
    mcp_client.execute_tool.return_value = {"type": "FeatureCollection", "features": []}

    response = _client(mcp_client).post(
        "/mcp-diagnostics/tools/call",
        json={
            "name": "GetServices",
            "arguments": {"scenario_id": 772},
            "meta": {"source": "diagnostics"},
        },
    )

    assert response.status_code == 200
    assert response.json() == {"result": {"type": "FeatureCollection", "features": []}}
    mcp_client.execute_tool.assert_awaited_once_with(
        "GetServices",
        {"scenario_id": 772},
        meta={"source": "diagnostics"},
        log=True,
    )


def test_rejects_non_object_arguments():
    response = _client(AsyncMock()).post(
        "/mcp-diagnostics/tools/call",
        json={"name": "GetServices", "arguments": []},
    )

    assert response.status_code == 422
