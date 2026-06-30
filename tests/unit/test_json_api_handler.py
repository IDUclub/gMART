"""Unit tests for the agents ``JsonApiHandler`` — bounded retry + status mapping.

These pin the two behaviours behind the production RecursionError incident:
- a non-2xx status (404, 400, ...) must *raise* immediately, not recurse forever;
- transient failures (network errors, 500 "reset by peer") retry a bounded number
  of times, then surface a ``DownstreamServiceError`` instead of looping.

The HTTP layer is faked: ``FakeSession.get/post`` returns an async context manager
that yields a programmed response or raises a programmed exception, so no socket
is ever opened.
"""

from __future__ import annotations

import pytest

from src.agents.common.api_handlers.json_api_handler import JsonApiHandler
from src.agents.common.exceptions.api_exceptions import DownstreamServiceError
from src.agents.common.exceptions.base_exceptions import (
    AgentsInputException,
    AgentsNotFound,
    AgentsUnauthorizedException,
)
from src.agents.common.exceptions.token_exceptions import TokenExpiredError


class FakeResponse:
    """Minimal stand-in for ``aiohttp.ClientResponse``."""

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
    """Async context manager returned by ``FakeSession.get/post``.

    The programmed ``outcome`` is either a ``FakeResponse`` (yielded on enter) or an
    ``Exception`` instance (raised on enter, to emulate a transport failure).
    """

    def __init__(self, outcome) -> None:
        self._outcome = outcome

    async def __aenter__(self):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """Records each call's HTTP method and replays programmed outcomes in order."""

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.methods: list[str] = []

    def get(self, url=None, headers=None, params=None):
        self.methods.append("get")
        return FakeReqCtx(self._outcomes.pop(0))

    def post(self, url=None, headers=None, params=None, json=None):
        self.methods.append("post")
        return FakeReqCtx(self._outcomes.pop(0))


def _handler(max_retries: int = 3) -> JsonApiHandler:
    # backoff_base=0 keeps the retry sleeps instantaneous.
    return JsonApiHandler("http://urban", max_retries=max_retries, backoff_base=0)


# --------------------------------------------------------------------------- #
# Success
# --------------------------------------------------------------------------- #
async def test_get_success_returns_json():
    session = FakeSession([FakeResponse(200, json_body={"project": {"project_id": 7}})])
    result = await _handler().get("/v1/scenarios/1", session=session)
    assert result == {"project": {"project_id": 7}}
    assert session.methods == ["get"]


# --------------------------------------------------------------------------- #
# Terminal statuses raise immediately (no recursion) — the incident regression
# --------------------------------------------------------------------------- #
async def test_get_404_raises_not_found_without_retry():
    session = FakeSession([FakeResponse(404, text_body="missing")])
    with pytest.raises(AgentsNotFound):
        await _handler().get("/v1/scenarios/999", session=session)
    # exactly one call: the old code recursed ~1000 times here before crashing
    assert session.methods == ["get"]


async def test_get_400_raises_input_exception():
    session = FakeSession([FakeResponse(400, text_body="bad")])
    with pytest.raises(AgentsInputException):
        await _handler().get("/v1/x", session=session)


async def test_get_401_token_expired():
    session = FakeSession([FakeResponse(401, json_body={"message": "Token expired."})])
    with pytest.raises(TokenExpiredError):
        await _handler().get("/v1/x", session=session)


async def test_get_401_other_message_unauthorized():
    session = FakeSession([FakeResponse(401, json_body={"message": "nope"})])
    with pytest.raises(AgentsUnauthorizedException):
        await _handler().get("/v1/x", session=session)


async def test_get_unexpected_status_raises_downstream():
    session = FakeSession([FakeResponse(503, text_body="overloaded")])
    with pytest.raises(DownstreamServiceError) as ei:
        await _handler().get("/v1/x", session=session)
    assert ei.value.downstream_status == 503
    assert ei.value.status_code == 502


async def test_get_non_transient_500_raises_downstream():
    session = FakeSession([FakeResponse(500, json_body={"error": "boom"})])
    with pytest.raises(DownstreamServiceError) as ei:
        await _handler().get("/v1/x", session=session)
    assert ei.value.downstream_status == 500
    assert session.methods == ["get"]


# --------------------------------------------------------------------------- #
# Transient failures: bounded retry
# --------------------------------------------------------------------------- #
async def test_get_reset_by_peer_retries_then_succeeds():
    session = FakeSession(
        [
            FakeResponse(500, json_body={"error": "connection reset by peer"}),
            FakeResponse(200, json_body=[{"name": "ok"}]),
        ]
    )
    result = await _handler().get("/v1/x", session=session)
    assert result == [{"name": "ok"}]
    assert session.methods == ["get", "get"]


async def test_get_reset_by_peer_exhausts_retries_and_raises():
    session = FakeSession([FakeResponse(500, json_body={"error": "reset by peer"})] * 3)
    with pytest.raises(DownstreamServiceError) as ei:
        await _handler(max_retries=3).get("/v1/x", session=session)
    assert ei.value.downstream_status is None
    assert len(session.methods) == 3


async def test_get_network_error_retried_then_succeeds():
    session = FakeSession(
        [ConnectionResetError("reset"), FakeResponse(200, json_body={"ok": True})]
    )
    result = await _handler().get("/v1/x", session=session)
    assert result == {"ok": True}
    assert session.methods == ["get", "get"]


async def test_get_network_error_exhausted_raises_downstream():
    session = FakeSession([ConnectionResetError("reset")] * 3)
    with pytest.raises(DownstreamServiceError) as ei:
        await _handler(max_retries=3).get("/v1/x", session=session)
    assert ei.value.downstream_status is None
    assert "ConnectionResetError" in str(ei.value.error_input)


# --------------------------------------------------------------------------- #
# POST uses POST on retry (regression: the old retry path called self.get)
# --------------------------------------------------------------------------- #
async def test_post_retries_with_post_not_get():
    session = FakeSession(
        [
            FakeResponse(500, json_body={"error": "reset by peer"}),
            FakeResponse(201, json_body={"created": True}),
        ]
    )
    result = await _handler().post("/v1/x", data={"a": 1}, session=session)
    assert result == {"created": True}
    assert session.methods == ["post", "post"]
