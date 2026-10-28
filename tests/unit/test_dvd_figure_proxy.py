"""The real DVD REST transport behind the public figure route, without network."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents.common.middlewares.exception_handler import ExceptionHandlerMiddleware
from src.agents.dependencies import dependencies
from src.agents.routers.dvd_controller import dvd_router

PNG = b"\x89PNG\r\n\x1a\nfigure"


@pytest.fixture
def proxy(monkeypatch):
    calls = []
    upstream = SimpleNamespace(status=200, unavailable=False)

    def respond(request):
        calls.append(request)
        if upstream.unavailable:
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(
            upstream.status,
            content=PNG if upstream.status == 200 else b'{"detail":"missing"}',
            headers={"Content-Type": "image/png"},
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "src.agents.api_clients.dvd_api_client.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    service_auth = SimpleNamespace(
        get_authorization_headers=AsyncMock(
            return_value={"Authorization": "Bearer service-token"}
        )
    )
    monkeypatch.setattr(dependencies, "get_service_auth", lambda: service_auth)
    config = SimpleNamespace(DVD_API_URL="http://dvd")
    monkeypatch.setitem(dependencies.app_deps, "app_config", config)
    app = FastAPI()
    app.include_router(dvd_router)
    app.add_middleware(ExceptionHandlerMiddleware)
    with TestClient(app) as client:
        yield client, calls, upstream, config


def test_token_subject_and_scope_forwarded_with_service_token(proxy):
    client, calls, _, _ = proxy
    token = "e30.eyJzdWIiOiJ1c2VyLXN1YmplY3QifQ.sig"
    response = client.get(
        "/documents/figures/node-1?project_id=42&scenario_id=772",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.content == PNG
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "private"
    assert response.headers["content-disposition"].startswith("inline;")
    request = calls[0]
    assert request.url.path == "/library/figures/node-1/image"
    assert dict(request.url.params) == {"project_id": "42", "scenario_id": "772"}
    assert request.headers["X-User-Id"] == "user-subject"
    assert request.headers["Authorization"] == "Bearer service-token"


def test_anonymous_corpus_image_and_download_disposition(proxy):
    client, calls, _, _ = proxy
    response = client.get("/documents/figures/node-1?download=1")
    assert response.status_code == 200
    assert response.content == PNG
    assert response.headers["content-disposition"].startswith("attachment;")
    assert calls[0].headers["X-User-Id"] == "anonymous"
    assert not calls[0].url.params


@pytest.mark.parametrize("query", ["project_id=42", "scenario_id=772", "project_id="])
def test_anonymous_project_scope_rejected_before_fetch(proxy, query):
    client, calls, _, _ = proxy
    assert client.get(f"/documents/figures/node-1?{query}").status_code == 401
    assert not calls


@pytest.mark.parametrize(
    "status,expected", [(404, 404), (403, 502), (500, 502), (400, 502)]
)
def test_upstream_status_mapping(proxy, status, expected):
    client, _, upstream, _ = proxy
    upstream.status = status
    assert client.get("/documents/figures/node-1").status_code == expected


def test_transport_failure_uses_downstream_error(proxy):
    client, _, upstream, _ = proxy
    upstream.unavailable = True
    assert client.get("/documents/figures/node-1").status_code == 502


def test_disabled_proxy_is_503_without_fetch(proxy):
    client, calls, _, config = proxy
    config.DVD_API_URL = None
    assert client.get("/documents/figures/node-1").status_code == 503
    assert not calls
