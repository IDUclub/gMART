"""Unit tests for the /auth router — availability flag and the token proxy.

A fresh FastAPI app is built from the router with the app-config dependency overridden,
so the endpoint logic is exercised without a real auth helper (its HTTP call is mocked).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents.common.middlewares.exception_handler import (
    ExceptionHandlerMiddleware,
)
from src.agents.dependencies.dependencies import get_app_config
from src.agents.routers.auth_controller import auth_router


def _client(auth_helper_url: str | None, auth_helper_api_key: str | None) -> TestClient:
    app = FastAPI()
    app.add_middleware(ExceptionHandlerMiddleware)
    app.include_router(auth_router)
    app.dependency_overrides[get_app_config] = lambda: SimpleNamespace(
        AUTH_HELPER_URL=auth_helper_url,
        AUTH_HELPER_API_KEY=auth_helper_api_key,
    )
    return TestClient(app, raise_server_exceptions=False)


class TestAuthAvailable:
    def test_enabled_when_both_vars_set(self):
        resp = _client("http://helper", "key").get("/auth/available")
        assert resp.status_code == 200
        assert resp.json() == {"enabled": True}

    @pytest.mark.parametrize(
        "url,key", [(None, None), ("http://helper", None), (None, "key")]
    )
    def test_disabled_without_full_config(self, url, key):
        resp = _client(url, key).get("/auth/available")
        assert resp.status_code == 200
        assert resp.json() == {"enabled": False}


class TestIssueToken:
    def test_proxies_credentials_and_returns_helper_response(self):
        helper_response = {
            "access_token": "jwt",
            "expires_in": 300,
            "token_type": "Bearer",
        }
        with patch(
            "src.agents.routers.auth_controller.JsonApiHandler.post",
            new=AsyncMock(return_value=helper_response),
        ) as post:
            resp = _client("http://helper", "key").post(
                "/auth/token", json={"username": "user", "password": "pass"}
            )
        assert resp.status_code == 200
        assert resp.json() == helper_response
        post.assert_awaited_once_with(
            "/api/token",
            headers={"X-Auth-Helper-Api-Key": "key"},
            data={
                "username": "user",
                "password": "pass",
                "scope": "openid profile email",
            },
        )

    def test_not_configured_raises_404(self):
        resp = _client(None, None).post(
            "/auth/token", json={"username": "user", "password": "pass"}
        )
        assert resp.status_code == 404

    def test_validates_body(self):
        resp = _client("http://helper", "key").post(
            "/auth/token", json={"username": "user"}
        )
        assert resp.status_code == 422
