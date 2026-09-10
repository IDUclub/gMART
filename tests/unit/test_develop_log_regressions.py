"""Failures observed on develop: connection loss, empty JSON and auth/config handling."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from redis.exceptions import ConnectionError

from src.agents.services.pipeline_state import PipelineStateStore, PipelineStatus


async def test_completion_recovers_from_one_lost_redis_connection(monkeypatch):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    store = PipelineStateStore(redis)
    await store.create(
        "r", chat_id=None, user_query="q", scenario_id=845, model="m", temperature=0
    )
    original = redis.get
    attempts = 0

    async def get(key):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("Connection lost")
        return await original(key)

    monkeypatch.setattr(redis, "get", get)
    await store.set_status("r", PipelineStatus.DONE)
    assert (await store.get_state("r"))["status"] == "done"
    await redis.aclose()


async def test_event_retry_after_lost_ack_does_not_duplicate(monkeypatch):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    store = PipelineStateStore(redis)
    original = redis.pipeline
    attempts = 0

    def pipeline(*args, **kwargs):
        p = original(*args, **kwargs)
        execute = p.execute

        async def execute_then_lose_response(*args, **kwargs):
            nonlocal attempts
            result = await execute(*args, **kwargs)
            attempts += 1
            if attempts == 1:
                raise ConnectionError("Lost EXEC acknowledgement")
            return result

        p.execute = execute_then_lose_response
        return p

    monkeypatch.setattr(redis, "pipeline", pipeline)
    event = {"type": "chunk", "content": {"text": "same text"}}
    await store.buffer_event("r", event)
    await store.buffer_event("r", event)  # distinct real events must not be coalesced
    assert attempts >= 2
    assert await store.get_buffered_events("r") == [event, event]
    assert await redis.ttl("pipeline:r:events") > 0
    await redis.aclose()


async def test_persistent_redis_failure_is_bounded_and_classified():
    from src.agents.common.exceptions.base_exceptions import PipelineStorageUnavailable

    redis = SimpleNamespace(get=AsyncMock(side_effect=ConnectionError("internal host")))
    with pytest.raises(PipelineStorageUnavailable):
        await PipelineStateStore(redis).set_status("r", PipelineStatus.DONE)
    assert redis.get.await_count == 3


async def test_event_is_persisted_before_it_can_be_emitted():
    from src.agents.services.scenario_data.scenario_data_service import (
        ScenarioDataService,
    )

    service = object.__new__(ScenarioDataService)
    service.state_store = SimpleNamespace(
        buffer_event=AsyncMock(side_effect=ConnectionError("lost"))
    )
    with pytest.raises(ConnectionError):
        await service._buf("r", {"type": "chunk"})
    service.state_store.buffer_event.assert_awaited_once()


def test_env_file_detection_preserves_process_settings(tmp_path, monkeypatch):
    from src.agents.common.config import app_config_loader as loader

    path = tmp_path / ".env.agents"
    path.write_text("URBAN_API_URL=http://file-value\n", encoding="utf-8")
    monkeypatch.setenv("URBAN_API_URL", "http://process-value")
    monkeypatch.setattr(loader, "find_dotenv", lambda _: str(path))
    assert loader.try_load("agents") is True
    import os

    assert os.environ["URBAN_API_URL"] == "http://process-value"


async def test_gpt_oss_json_recovers_from_exhausted_reasoning_budget():
    from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter

    adapter = OpenAiCompatAdapter("http://example.test/v1", think_mode="off")

    def response(content, reason):
        return SimpleNamespace(
            model="gpt-oss-20b",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, role="assistant"),
                    finish_reason=reason,
                )
            ],
        )

    create = AsyncMock(
        side_effect=[response("", "length"), response('{"ok":true}', "stop")]
    )
    adapter.client.chat.completions.create = create
    result = await adapter.chat(
        "gpt-oss-20b", [], think=False, format="json", options={"num_predict": 1024}
    )
    assert json.loads(result.message.content) == {"ok": True}
    calls = create.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["reasoning_effort"] == "low"
    assert calls[1].kwargs["max_tokens"] > calls[0].kwargs["max_tokens"]
    await adapter.client.close()


async def test_exhausted_json_is_error_not_user_clarification():
    from src.agents.model_clients.llm_base import LlmResponseError
    from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter

    adapter = OpenAiCompatAdapter("http://example.test/v1")
    result = SimpleNamespace(
        model="m",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="", role="assistant"),
                finish_reason="length",
            )
        ],
    )
    create = AsyncMock(return_value=result)
    adapter.client.chat.completions.create = create
    with pytest.raises(LlmResponseError):
        await adapter.chat("m", [], format="json", options={"num_predict": 1024})
    assert create.await_count == 2
    await adapter.client.close()


def test_invalid_credentials_are_sanitized_without_retry():
    from unittest.mock import patch

    from src.agents.common.exceptions.base_exceptions import AgentsInputException
    from src.agents.routers import auth_controller
    from tests.unit.test_auth_controller import _client

    error = AgentsInputException(
        "http://internal-helper",
        json.dumps(
            {
                "detail": {
                    "error": "invalid_grant",
                    "error_description": "Invalid user credentials",
                }
            }
        ),
    )
    with patch.object(
        auth_controller.JsonApiHandler,
        "post",
        new_callable=AsyncMock,
        side_effect=error,
    ) as post:
        response = _client("http://internal-helper", "server-secret").post(
            "/auth/token", json={"username": "u", "password": "private-password"}
        )
    assert response.status_code == 401
    assert "Неверный логин или пароль" in response.json()["message"]
    assert (
        "internal-helper" not in response.text
        and "private-password" not in response.text
    )
    post.assert_awaited_once()


async def test_storage_outage_does_not_restart_the_sse_pipeline():
    from starlette.requests import Request

    from src.agents.common.exceptions.base_exceptions import PipelineStorageUnavailable
    from src.agents.common.executors.sse_executors import stream_with_error_handling

    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [],
            "query_string": b"",
            "server": ("test", 80),
        }
    )
    calls = 0

    async def pipeline(**kwargs):
        nonlocal calls
        calls += 1
        raise PipelineStorageUnavailable()
        yield {}

    events = [
        e
        async for e in stream_with_error_handling(
            pipeline, request, None, "m", rerun=True
        )
    ]
    assert calls == 1
    assert len(events) == 1 and events[0]["type"] == "error"
    assert "сохранить состояние" in events[0]["content"]["message"]


def test_helper_configuration_error_is_not_invalid_user_password():
    from unittest.mock import patch

    from src.agents.common.exceptions.base_exceptions import AgentsUnauthorizedException
    from src.agents.routers import auth_controller
    from tests.unit.test_auth_controller import _client

    with patch.object(
        auth_controller.JsonApiHandler,
        "post",
        new_callable=AsyncMock,
        side_effect=AgentsUnauthorizedException("bad helper key at internal host"),
    ):
        response = _client("http://internal-helper", "server-secret").post(
            "/auth/token", json={"username": "u", "password": "private-password"}
        )
    assert response.status_code == 502
    assert "Неверный логин" not in response.text
    assert "internal" not in response.text and "private-password" not in response.text
