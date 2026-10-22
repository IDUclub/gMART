"""Exception responses and server logs must not echo request credentials."""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from src.agents.common.exceptions.a2a_exceptions import A2AJsonRpcError
from src.agents.common.middlewares.exception_handler import ExceptionHandlerMiddleware


@pytest.fixture
def logs():
    captured = []
    sink = logger.add(
        captured.append,
        diagnose=False,
        format="{message}",
        filter="src.agents.common.middlewares.exception_handler",
    )
    try:
        yield captured
    finally:
        logger.remove(sink)


@pytest.fixture
def sensitive_headers():
    return {
        "Authorization": "Bearer synthetic-test-token",
        "Cookie": "session=synthetic-cookie",
        "Set-Cookie": "session=synthetic-set-cookie",
        "Proxy-Authorization": "Bearer synthetic-proxy-token",
        "X-Auth-Helper-Api-Key": "synthetic-helper-key",
        "X-Api-Key": "synthetic-api-key",
        "X-Custom-ToKeN": "synthetic-custom-token",
        "X-Custom-SeCrEt": "synthetic-custom-secret",
        "X-Custom-PaSsWoRd": "synthetic-custom-password",
        "X-Custom-AuTh": "synthetic-custom-auth",
        "X-Custom-KeY": "synthetic-custom-key",
    }


def _client(path, error):
    app = FastAPI()
    app.add_middleware(ExceptionHandlerMiddleware)

    @app.post(path)
    async def fail():
        raise error

    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("path", ["/failure", "/auth/token"])
def test_unhandled_error_redacts_response_and_logs(
    path, logs, sensitive_headers, monkeypatch
):
    read_id = AsyncMock(side_effect=AssertionError("Body must not be read"))
    monkeypatch.setattr(ExceptionHandlerMiddleware, "request_id", read_id)
    response = _client(path, RuntimeError("synthetic failure")).post(
        path + "?password=synthetic-query-password",
        headers={**sensitive_headers, "X-Request-Id": "synthetic-correlation"},
        json={"username": "u", "password": "synthetic-pass"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "message": "Internal server error",
        "error_type": "RuntimeError",
        "detail": "synthetic failure",
    }
    read_id.assert_not_awaited()
    output = "".join(logs)
    for value in [
        *sensitive_headers.values(),
        "synthetic-pass",
        "synthetic-query-password",
    ]:
        assert value not in response.text
        assert value not in output
    for value in sensitive_headers.values():
        assert value.removeprefix("Bearer ") not in output
    assert "method=POST" in output
    assert f"path={path}" in output
    assert "synthetic-correlation" in output
    assert "RuntimeError: synthetic failure" in output
    assert any(message.record["exception"] is not None for message in logs)


@pytest.mark.parametrize("request_id", ["synthetic-rpc-id", 0, 42, None])
def test_a2a_error_preserves_request_id(request_id, logs, sensitive_headers):
    response = _client("/rpc", A2AJsonRpcError(-32602, "Invalid params")).post(
        "/rpc",
        headers=sensitive_headers,
        json={"id": request_id, "params": {"password": "synthetic-pass"}},
    )

    assert response.status_code == 200
    assert response.json() == {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32602, "message": "Invalid params"},
    }
    for value in [*sensitive_headers.values(), "synthetic-pass"]:
        assert value not in response.text
        assert value not in "".join(logs)


@pytest.mark.parametrize("body", ["not-json", "[]", "null", "{}"])
def test_a2a_error_without_request_id(body, logs):
    response = _client("/rpc", A2AJsonRpcError(-32600, "Invalid Request")).post(
        "/rpc", content=body
    )

    assert response.status_code == 200
    assert response.json()["id"] is None
    assert not logs
