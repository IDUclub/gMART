from unittest.mock import AsyncMock, patch

import pytest
from loguru import logger

from src.agents.common.exceptions.api_exceptions import DownstreamServiceError
from src.agents.common.exceptions.base_exceptions import AgentsInputException
from tests.unit.test_auth_controller import _client


@pytest.fixture
def records(monkeypatch):
    monkeypatch.setenv("AUTH_LOG_USERNAME", "false")
    captured = []
    sink = logger.add(lambda message: captured.append(message.record))
    yield captured
    logger.remove(sink)


def _audit(records):
    return [r["extra"]["login"] for r in records if "login" in r["extra"]]


def _post(*, result=None, error=None, username="private-user"):
    with patch(
        "src.agents.routers.auth_controller.JsonApiHandler.post",
        new=AsyncMock(return_value=result, side_effect=error),
    ):
        return _client("http://helper", "private-key").post(
            "/auth/token",
            json={"username": username, "password": "private-password"},
            headers={"Authorization": "Bearer private-header"},
        )


def test_success_correlates_attempt_and_result_without_secrets(records):
    response = _post(
        result={"access_token": "private-token", "refresh_token": "private-refresh"}
    )
    assert response.status_code == 200
    start, end = _audit(records)
    assert start["event"] == "login_attempt"
    assert start["outcome"] == "started"
    assert end["event"] == "login_result"
    assert end["outcome"] == "success"
    assert start["attempt_id"] == end["attempt_id"]
    assert end["status"] == 200
    assert end["duration_ms"] >= 0
    assert "private-" not in str(records)


@pytest.mark.parametrize(
    "error,status,reason,downstream",
    [
        (
            AgentsInputException(
                "private-error",
                {
                    "detail": {
                        "error": "invalid_grant",
                        "error_description": "private-detail",
                    }
                },
            ),
            401,
            "invalid_credentials",
            "-",
        ),
        (
            DownstreamServiceError(
                "private-helper", 503, "private-error", "private-detail"
            ),
            502,
            "auth_helper_unavailable",
            503,
        ),
        (
            DownstreamServiceError("private-helper", None, "private-error"),
            502,
            "auth_helper_unavailable",
            None,
        ),
        (RuntimeError("private-error"), 500, "internal_error", "-"),
    ],
)
def test_failure_reasons_and_no_secret_payloads(
    records, error, status, reason, downstream
):
    response = _post(error=error)
    assert response.status_code == status
    start, end = _audit(records)
    assert end["reason"] == reason
    assert end["status"] == status
    assert end["downstream_status"] == downstream
    assert end["outcome"] == "failure"
    assert start["attempt_id"] == end["attempt_id"]
    assert "private-" not in str(records)
    assert "private-" not in response.text


@pytest.mark.parametrize(
    "body",
    [{"username": "private-user"}, {"password": {"private": "private-password"}}, None],
)
def test_validation_failures_logged_without_reading_body(records, body):
    response = _client("http://helper", "private-key").post("/auth/token", json=body)
    assert response.status_code == 422
    assert _audit(records)[-1]["reason"] == "invalid_request"
    assert "private" not in str(records)


def test_missing_configuration_logged(records):
    response = _client(None, None).post(
        "/auth/token", json={"username": "private-user", "password": "private-password"}
    )
    assert response.status_code == 404
    assert _audit(records)[-1]["reason"] == "login_not_configured"
    assert "private-" not in str(records)


@pytest.mark.parametrize(
    "result", [None, {}, {"access_token": ""}, {"access_token": 123}]
)
def test_missing_token_not_reported_as_success(records, result):
    response = _post(result=result)
    assert response.status_code == 502
    assert _audit(records)[-1]["reason"] == "invalid_helper_response"
    assert _audit(records)[-1]["downstream_status"] == 200


def test_username_opt_in_escapes_line_breaks(records, monkeypatch):
    monkeypatch.setenv("AUTH_LOG_USERNAME", "true")
    _post(result={"access_token": "private-token"}, username="alice\nFORGED\rline")
    assert _audit(records)[-1]["username"] == "alice\nFORGED\rline"
    message = [r["message"] for r in records if "login" in r["extra"]][-1]
    assert "\n" not in message and "\r" not in message
    assert "private-" not in str(records)


def test_availability_poll_not_a_login_attempt(records):
    assert _client(None, None).get("/auth/available").status_code == 200
    assert _audit(records) == []


def test_attempt_ids_are_unique(records):
    for _ in range(2):
        _post(result={"access_token": "private-token"})
    assert _audit(records)[0]["attempt_id"] != _audit(records)[2]["attempt_id"]
