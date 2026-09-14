import asyncio
import uuid
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from loguru import logger
from redis.asyncio import Redis
from redis.asyncio.client import PubSub
from redis.exceptions import ConnectionError, ResponseError

from src.agents.common.logging.redis_logging import (
    LoggedRedis,
    redis_attempt,
    redis_request_id,
)
from src.agents.services.pipeline_state import PipelineStateStore


@pytest.fixture
def records(monkeypatch):
    monkeypatch.setenv("REDIS_LOG_COMMANDS", "true")
    captured = []
    sink = logger.add(
        lambda message: captured.append(message.record),
        filter=lambda r: "redis" in r["extra"],
    )
    yield captured
    logger.remove(sink)


@pytest.fixture
async def client():
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    redis = LoggedRedis(connection_pool=fake.connection_pool)
    yield redis
    await redis.aclose()
    await fake.aclose()


async def test_success_metadata_never_contains_values_or_credentials(
    records, monkeypatch
):
    client = LoggedRedis.from_url("redis://user:private-password@gmart-redis:6379/2")
    rid = str(uuid.uuid4())
    monkeypatch.setattr(
        Redis, "execute_command", AsyncMock(return_value="private-value")
    )
    try:
        assert await client.get(f"pipeline:{rid}:state") == "private-value"
        record = records[-1]["extra"]["redis"]
        assert record["endpoint"] == "gmart-redis:6379/2"
        assert record["request_id"] == rid
        assert record["command"] == "GET"
        assert record["key_kind"] == "pipeline:state"
        assert record["outcome"] == "ok"
        assert record["duration_ms"] >= 0
        assert "private" not in str(records)
    finally:
        await client.aclose()


async def test_watch_and_transaction_logged_when_executed(client, records):
    rid = str(uuid.uuid4())
    key = f"pipeline:{rid}:state"
    async with client.pipeline() as pipe:
        await pipe.watch(key)
        assert await pipe.get(key) is None
        pipe.multi()
        pipe.set(key, "private-payload")
        pipe.expire(key, 60)
        assert [r["extra"]["redis"]["command"] for r in records] == ["WATCH", "GET"]
        assert await pipe.execute() == [True, True]
    record = records[-1]["extra"]["redis"]
    assert record["command"] == "EXEC"
    assert record["commands"] == "SET,EXPIRE"
    assert record["command_count"] == 2
    assert record["request_id"] == rid
    assert "private-payload" not in str(records)


async def test_partial_pipeline_error_is_not_logged_as_success(client, records):
    await client.set("key", "private-payload")
    async with client.pipeline() as pipe:
        pipe.lrange("key", 0, -1)
        result = await pipe.execute(raise_on_error=False)
    assert isinstance(result[0], ResponseError)
    assert records[-1]["extra"]["redis"]["outcome"] == "partial_error"


async def test_retries_have_attempt_and_request_context(client, records, monkeypatch):
    monkeypatch.setattr(
        Redis,
        "execute_command",
        AsyncMock(side_effect=[ConnectionError("private-error"), "ok"]),
    )
    rid = str(uuid.uuid4())
    assert (
        await PipelineStateStore(client)._retry(
            client.get, "unknown-key", _request_id=rid
        )
        == "ok"
    )
    assert [r["extra"]["redis"]["attempt"] for r in records] == [1, 2]
    assert [r["extra"]["redis"]["request_id"] for r in records] == [rid, rid]
    assert records[0]["level"].name == "ERROR"
    assert "private-error" not in str(records)
    assert redis_attempt.get() == 1
    assert redis_request_id.get() is None


async def test_errors_remain_when_success_logging_disabled(
    client, records, monkeypatch
):
    monkeypatch.setenv("REDIS_LOG_COMMANDS", "false")
    await client.set("private-key", "private-value")
    assert records == []
    error = ConnectionError("private-exception")
    monkeypatch.setattr(Redis, "execute_command", AsyncMock(side_effect=error))
    with pytest.raises(ConnectionError) as raised:
        await client.get("private-key")
    assert raised.value is error
    assert records[-1]["extra"]["redis"]["error_type"] == "ConnectionError"
    assert "private" not in str(records)


async def test_script_and_non_uuid_keys_are_redacted(client, records, monkeypatch):
    monkeypatch.setattr(Redis, "execute_command", AsyncMock(return_value=1))
    await client.eval("private-script", 1, "private-key", "private-token")
    await client.get("pipeline:private-token:state")
    assert all(r["extra"]["redis"]["request_id"] == "-" for r in records)
    assert "private" not in str(records)


async def test_pubsub_and_publish_do_not_log_tokens(client, records, monkeypatch):
    rid = str(uuid.uuid4())
    channel = f"pipeline:{rid}:token_channel"
    async with client.pubsub() as pubsub:
        await pubsub.subscribe(channel)
        assert records[-1]["extra"]["redis"]["outcome"] == "sent"
        await pubsub.get_message(timeout=1)
        assert await client.publish(channel, "private-token") == 1
        result = await pubsub.get_message(timeout=1)
        assert result["data"] == "private-token"
        monkeypatch.setattr(
            PubSub,
            "parse_response",
            AsyncMock(side_effect=ConnectionError("private-error")),
        )
        with pytest.raises(ConnectionError):
            await pubsub.get_message(timeout=1)
        assert records[-1]["extra"]["redis"]["command"] == "PUBSUB_READ"
        assert records[-1]["extra"]["redis"]["outcome"] == "error"
    assert "private" not in str(records)


async def test_cancellation_is_preserved(client, records, monkeypatch):
    monkeypatch.setattr(
        Redis, "execute_command", AsyncMock(side_effect=asyncio.CancelledError)
    )
    with pytest.raises(asyncio.CancelledError):
        await client.get("key")
    assert records[-1]["extra"]["redis"]["outcome"] == "cancelled"
