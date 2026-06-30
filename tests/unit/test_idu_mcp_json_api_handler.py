"""Unit tests for the idu_mcp ``JsonApiHandler`` — bounded retry + ToolError mapping.

The MCP-side handler shares the agents fix (bounded retry instead of recursion) but,
being an MCP tool boundary, surfaces failures as ``fastmcp.exceptions.ToolError`` so the
calling LLM agent gets an actionable message rather than an unhandled crash.
"""

from __future__ import annotations

import pytest
from fastmcp.exceptions import ToolError

from src.idu_mcp.common.api_handlers.json_api_handler import JsonApiHandler


class FakeResponse:
    def __init__(
        self,
        status: int,
        json_body=None,
        text_body: str = "",
        content_type: str = "application/json",
    ) -> None:
        self.status = status
        self._json = json_body
        self._text = text_body
        self.content_type = content_type

    async def json(self, content_type=None):
        return self._json

    async def text(self):
        return self._text


class FakeReqCtx:
    def __init__(self, outcome) -> None:
        self._outcome = outcome

    async def __aenter__(self):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def get(self, url=None, headers=None, params=None):
        self.calls += 1
        return FakeReqCtx(self._outcomes.pop(0))


def _handler(max_retries: int = 3) -> JsonApiHandler:
    return JsonApiHandler("http://urban", max_retries=max_retries, backoff_base=0)


async def test_get_success_returns_json():
    session = FakeSession([FakeResponse(200, json_body=[{"name": "ok"}])])
    result = await _handler().get("v1/service_types", session=session)
    assert result == [{"name": "ok"}]
    assert session.calls == 1


async def test_get_404_raises_tool_error_without_retry():
    session = FakeSession([FakeResponse(404, text_body="missing")])
    with pytest.raises(ToolError):
        await _handler().get("v1/scenarios/999", session=session)
    assert session.calls == 1


async def test_get_non_transient_500_raises_tool_error():
    session = FakeSession([FakeResponse(500, json_body={"error": "boom"})])
    with pytest.raises(ToolError):
        await _handler().get("v1/x", session=session)
    assert session.calls == 1


async def test_get_reset_by_peer_retries_then_succeeds():
    session = FakeSession(
        [
            FakeResponse(500, json_body={"error": "reset by peer"}),
            FakeResponse(200, json_body=[{"name": "ok"}]),
        ]
    )
    result = await _handler().get("v1/x", session=session)
    assert result == [{"name": "ok"}]
    assert session.calls == 2


async def test_get_retries_exhausted_raises_tool_error():
    session = FakeSession([ConnectionResetError("reset")] * 3)
    with pytest.raises(ToolError):
        await _handler(max_retries=3).get("v1/x", session=session)
    assert session.calls == 3
