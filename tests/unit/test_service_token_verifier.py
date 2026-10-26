"""Service-only JWT filtering follows the MCP unauthenticated contract."""

from unittest.mock import AsyncMock

import pytest
from fastmcp.server.auth import AccessToken
from fastmcp.server.auth.providers.jwt import JWTVerifier
from loguru import logger
from mcp.server.auth.middleware.bearer_auth import (
    BearerAuthBackend,
    RequireAuthMiddleware,
)
from starlette.applications import Starlette
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.idu_mcp.common.auth.token_verifier import ServiceTokenVerifier


@pytest.fixture
def verifier(monkeypatch):
    monkeypatch.setenv("SERVICE_AUTH_SERVER_URL", "https://synthetic-auth.invalid")
    monkeypatch.setenv("SERVICE_AUTH_REALM", "synthetic-realm")
    return ServiceTokenVerifier()


def _token(username):
    return AccessToken(
        token="synthetic-test-token",
        client_id="synthetic-client",
        scopes=[],
        claims={"preferred_username": username},
    )


@pytest.mark.parametrize("username", ["regular-user", "", None, 123])
async def test_non_service_token_returns_none(verifier, monkeypatch, username):
    parent = AsyncMock(return_value=_token(username))
    monkeypatch.setattr(JWTVerifier, "verify_token", parent)
    captured = []
    sink = logger.add(captured.append, diagnose=False)
    try:
        assert await verifier.verify_token("synthetic-test-token") is None
    finally:
        logger.remove(sink)

    parent.assert_awaited_once_with("synthetic-test-token")
    assert any(message.record["level"].name == "WARNING" for message in captured)
    assert "synthetic-test-token" not in "".join(captured)
    assert "service-account token is required" in "".join(captured)


async def test_service_account_token_is_returned(verifier, monkeypatch):
    token = _token("service-account-foo")
    monkeypatch.setattr(JWTVerifier, "verify_token", AsyncMock(return_value=token))

    assert await verifier.verify_token("synthetic-test-token") is token


async def test_invalid_token_returns_none(verifier, monkeypatch):
    monkeypatch.setattr(JWTVerifier, "verify_token", AsyncMock(return_value=None))

    assert await verifier.verify_token("synthetic-test-token") is None


def test_non_service_token_is_http_401(verifier, monkeypatch):
    parent = AsyncMock(return_value=_token("regular-user"))
    monkeypatch.setattr(JWTVerifier, "verify_token", parent)

    async def protected(request):
        return JSONResponse({"ok": True})

    app = AuthenticationMiddleware(
        RequireAuthMiddleware(
            Starlette(routes=[Route("/mcp", protected)]), required_scopes=[]
        ),
        backend=BearerAuthBackend(verifier),
    )
    response = TestClient(app).get(
        "/mcp", headers={"Authorization": "Bearer synthetic-test-token"}
    )

    assert response.status_code == 401
    assert response.json() == {
        "error": "invalid_token",
        "error_description": "Authentication required",
    }
    assert response.headers["www-authenticate"].startswith("Bearer ")
    assert "traceback" not in response.text.lower()
    assert "synthetic-test-token" not in response.text
    parent.assert_awaited_once_with("synthetic-test-token")
