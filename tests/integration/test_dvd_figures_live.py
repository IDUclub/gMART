"""A live DVD figure becomes a permanent file and is served by the gMART proxy.

Uses DVD_MCP_SERVER, optional DVD_API_URL and the service identity. No model or
Redis is required; an unavailable DVD or a corpus without figures skips the test.
"""

import os

import httpx
import pytest
from fastapi import FastAPI
from fastmcp import Client

from src.agents.api_clients.dvd_api_client import DvdApiClient
from src.agents.dependencies.dependencies import get_dvd_figure_client
from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient
from src.agents.routers.dvd_controller import dvd_router
from src.agents.services.dvd.dvd_context import DvdContextBuilder
from src.agents.services.dvd.figures import is_figure
from src.common.service_auth import (
    ANONYMOUS_USER_ID,
    ServiceTokenAuth,
    build_service_auth,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def live_figure():
    mcp_url = os.environ.get("DVD_MCP_SERVER", "").rstrip("/")
    api_url = (os.environ.get("DVD_API_URL") or mcp_url.removesuffix("/mcp")).rstrip(
        "/"
    )
    if not api_url:
        pytest.skip("IDU_DVD URLs are not set for the live figure check")
    auth = build_service_auth()
    async with auth:
        try:
            await auth.get_access_token()
            url = os.environ["DVD_MCP_SERVER"]
            mcp = DvdMcpClient(
                Client(url, auth=ServiceTokenAuth(auth, ANONYMOUS_USER_ID)),
                mcp_url=url,
            )
            result = await mcp.search_fragments(
                {"query": "рисунок", "types": ["figure"], "limit": 20},
                mode="filtered",
            )
            hits = [h for h in result.get("hits", []) if is_figure(h)]
            if not hits:
                pytest.skip("The live IDU_DVD corpus has no available figure hits")
            hit = DvdContextBuilder.ordered_hits(hits)[0]
            client = DvdApiClient(api_url, auth, ANONYMOUS_USER_ID)
            image = await client.get_figure_image(hit.get("node_id") or hit["id"])
        except Exception as exc:
            pytest.skip(f"IDU_DVD figure services unavailable: {exc}")
        yield api_url, hit, client, image


async def test_live_figure_file_event_and_proxy(live_figure, service, monkeypatch):
    api_url, hit, client, image = live_figure
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    service.dvd_api_url = api_url
    collected = {"citation_hits": [hit], "tool_calls": []}
    monkeypatch.setenv("DVD_READABLE_CITATIONS", "true")
    events = [
        e
        async for e in service._finish_retrieval(
            "live-figure", collected, "Рисунок [1].", 1
        )
    ]
    file = next(e["content"] for e in events if e["type"] == "file")
    app = FastAPI()
    app.include_router(dvd_router)
    app.dependency_overrides[get_dvd_figure_client] = lambda: client
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agents"
    ) as browser:
        response = await browser.get(file["url"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == image
