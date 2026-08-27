"""Unit tests for DvdMcpClient — kind→tool mapping, argument building, result normalization."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient
from src.common.service_auth import (
    ANONYMOUS_USER_ID,
    USER_ID_HEADER,
    service_headers,
)


def _client() -> DvdMcpClient:
    return DvdMcpClient(Mock(), mcp_url="http://dvd/mcp")


def test_tool_name_for_kind():
    assert DvdMcpClient.tool_name_for_kind("text") == "search_texts"
    assert DvdMcpClient.tool_name_for_kind("table") == "search_tables"
    assert DvdMcpClient.tool_name_for_kind("all") == "search_all"
    assert DvdMcpClient.tool_name_for_kind("anything-else") == "search_all"


async def test_search_calls_correct_tool_with_args():
    c = _client()
    c.execute_tool = AsyncMock(return_value={"count": 1, "hits": [{"text": "x"}]})
    out = await c.search("озеленение", kind="text", limit=7, context_height=2)
    c.execute_tool.assert_awaited_once()
    name, args = c.execute_tool.await_args.args
    assert name == "search_texts"
    assert args == {"query": "озеленение", "limit": 7, "context_height": 2}
    assert out == {"count": 1, "hits": [{"text": "x"}]}


async def test_search_includes_optional_filters():
    c = _client()
    c.execute_tool = AsyncMock(return_value={"hits": []})
    await c.search("q", name="СП 1", version="ред.2", tags=["a", "b"])
    _, args = c.execute_tool.await_args.args
    assert args["name"] == "СП 1"
    assert args["version"] == "ред.2"
    assert args["tags"] == ["a", "b"]


async def test_search_includes_structural_filters():
    c = _client()
    c.execute_tool = AsyncMock(return_value={"hits": []})
    await c.search(
        "q",
        document_names=["СП 42.13330", "ГОСТ 21.501"],
        block="amendment",
        types=["clause", "table"],
    )
    _, args = c.execute_tool.await_args.args
    assert args["document_names"] == ["СП 42.13330", "ГОСТ 21.501"]
    assert args["block"] == "amendment"
    assert args["types"] == ["clause", "table"]


async def test_search_includes_user_index_scope():
    c = _client()
    c.execute_tool = AsyncMock(return_value={"hits": []})
    await c.search("q", scenario_id=772)
    _, args = c.execute_tool.await_args.args
    assert args["scenario_id"] == "772"
    assert args["include_shared"] is True
    assert args["include_inherited"] is True


async def test_search_omits_empty_filters():
    c = _client()
    c.execute_tool = AsyncMock(return_value={"hits": []})
    await c.search("q")
    _, args = c.execute_tool.await_args.args
    assert set(args) == {"query", "limit", "context_height"}


def test_normalize_handles_none():
    assert DvdMcpClient._normalize(None) == {"count": 0, "hits": []}


def test_normalize_unwraps_pydantic_like_objects():
    class Hit:
        def model_dump(self):
            return {"text": "h"}

    class Resp:
        def model_dump(self):
            return {"count": 1, "hits": [Hit()]}

    out = DvdMcpClient._normalize(Resp())
    assert out["hits"] == [{"text": "h"}]


def test_normalize_fills_missing_count():
    out = DvdMcpClient._normalize({"hits": [{"a": 1}, {"b": 2}]})
    assert out["count"] == 2


class TestAnonymousTransportHeaders:
    """A public document-QA question still reaches IDU_DVD as an authorized service call."""

    @pytest.mark.asyncio
    async def test_anonymous_caller_sends_service_token_and_placeholder_user_id(
        self, monkeypatch
    ):
        # IDU_DVD rejects every search tool call without X-User-Id, so the anonymous
        # path must announce a placeholder subject rather than omit the header.
        from src.agents.dependencies import dependencies

        class FakeServiceAuth:
            async def get_authorization_headers(self):
                return {"Authorization": "Bearer service-token"}

        monkeypatch.setitem(
            dependencies.app_deps,
            "app_config",
            SimpleNamespace(DVD_MCP_URL="http://dvd:8000/mcp"),
        )
        monkeypatch.setattr(dependencies, "get_service_auth", FakeServiceAuth)

        client = await dependencies.get_dvd_mcp_client(token=None)

        auth = client.mcp_client.transport.auth
        headers = await service_headers(auth.auth, auth.user_id)
        assert headers["Authorization"] == "Bearer service-token"
        assert headers[USER_ID_HEADER] == ANONYMOUS_USER_ID
