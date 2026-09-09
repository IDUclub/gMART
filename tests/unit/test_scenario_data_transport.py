from unittest.mock import AsyncMock

import anyio
import httpx
import pytest

from agents.services.scenario_data import scenario_data_service as service_module
from agents.services.scenario_data.scenario_data_service import ScenarioDataService


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError(""),
        httpx.ReadError(""),
        httpx.RemoteProtocolError(""),
        anyio.BrokenResourceError(),
        anyio.ClosedResourceError(),
        anyio.EndOfStream(),
    ],
)
async def test_read_transport_errors_retry_boundedly(monkeypatch, state_store, error):
    service = ScenarioDataService.__new__(ScenarioDataService)
    service.state_store = state_store
    monkeypatch.setattr(service_module.asyncio, "sleep", AsyncMock())
    operation = AsyncMock(side_effect=error)
    events = []
    with pytest.raises(type(error)):
        async for event in service._retryable_operation(
            "request", None, [], operation, [], retry_transient=True
        ):
            events.append(event)
    assert operation.await_count == 3
    assert len(events) == 2
    assert all(e["content"]["status"] == "tool_retry" for e in events)


def test_wrapped_transport_error_is_retryable_without_message_matching():
    error = ValueError("Read failed")
    error.__cause__ = httpx.ConnectError("")
    assert service_module._is_transient_tool_error(error)


async def test_success_after_transport_retry_preserves_result(monkeypatch, state_store):
    service = ScenarioDataService.__new__(ScenarioDataService)
    service.state_store = state_store
    monkeypatch.setattr(service_module.asyncio, "sleep", AsyncMock())
    expected = {"scenario_id": 848, "value": 0}
    operation = AsyncMock(side_effect=[httpx.ConnectError(""), expected])
    result = []
    events = [
        e
        async for e in service._retryable_operation(
            "request", None, [], operation, result, retry_transient=True
        )
    ]
    assert result == [expected]
    assert operation.await_count == 2
    assert len(events) == 1


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Bad identifier"),
        httpx.HTTPStatusError(
            "Forbidden",
            request=httpx.Request("GET", "http://localhost"),
            response=httpx.Response(403),
        ),
    ],
)
async def test_permanent_errors_are_not_retried(monkeypatch, state_store, error):
    service = ScenarioDataService.__new__(ScenarioDataService)
    service.state_store = state_store
    monkeypatch.setattr(service_module.asyncio, "sleep", AsyncMock())
    operation = AsyncMock(side_effect=error)
    with pytest.raises(type(error)):
        async for _ in service._retryable_operation(
            "request", None, [], operation, [], retry_transient=True
        ):
            pytest.fail("Permanent errors must not produce a retry")
    assert operation.await_count == 1
