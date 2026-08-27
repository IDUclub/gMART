import pytest
from fakeredis.aioredis import FakeRedis
from redis.exceptions import TimeoutError as RedisTimeoutError

from src.agents.services.synapse_run_store import (
    SynapseIdempotencyConflict,
    SynapseRunStore,
)


@pytest.mark.asyncio
async def test_idempotency_reuses_same_request_and_rejects_other_payload() -> None:
    redis = FakeRedis(decode_responses=True)
    store = SynapseRunStore(redis)

    assert await store.claim_idempotency(
        user_id="u1", key="key", payload_hash="a", request_id="r1"
    ) == ("r1", True)
    assert await store.claim_idempotency(
        user_id="u1", key="key", payload_hash="a", request_id="r2"
    ) == ("r1", False)
    with pytest.raises(SynapseIdempotencyConflict):
        await store.claim_idempotency(
            user_id="u1", key="key", payload_hash="b", request_id="r3"
        )

    await redis.aclose()


@pytest.mark.asyncio
async def test_project_and_run_indexes_resolve_original_user() -> None:
    redis = FakeRedis(decode_responses=True)
    store = SynapseRunStore(redis)
    await store.create_state(
        "request-1",
        {
            "request_id": "request-1",
            "user_id": "user-1",
            "synapse_project_id": "project-1",
            "run_id": "run-1",
            "status": "running",
        },
    )
    await store.bind_project("request-1", "project-1", run_id="run-1")

    assert (
        await store.resolve_a2a_user(project_id="project-1", run_id="run-1") == "user-1"
    )
    assert await store.resolve_a2a_user(project_id="other", run_id="run-1") is None
    await redis.aclose()


@pytest.mark.asyncio
async def test_source_event_is_added_to_stream_only_once() -> None:
    redis = FakeRedis(decode_responses=True)
    store = SynapseRunStore(redis)
    await store.create_state(
        "request-1", {"request_id": "request-1", "status": "running"}
    )

    first = await store.add_event(
        "request-1", source_event_id="event-1", event={"value": 1}
    )
    repeated = await store.add_event(
        "request-1", source_event_id="event-1", event={"value": 1}
    )

    assert first is not None
    assert repeated is None
    rows = await redis.xrange("synapse:run:request-1:events")
    assert len(rows) == 1
    await redis.aclose()


@pytest.mark.asyncio
async def test_only_one_request_can_hold_an_active_chat() -> None:
    redis = FakeRedis(decode_responses=True)
    store = SynapseRunStore(redis)
    await store.create_state("request-1", {"status": "running"})
    await store.create_state("request-2", {"status": "running"})

    assert await store.claim_chat("chat-1", "request-1") is True
    assert await store.claim_chat("chat-1", "request-2") is False

    await redis.aclose()


@pytest.mark.asyncio
async def test_event_stream_treats_redis_timeout_as_active_heartbeat() -> None:
    class TimeoutRedis:
        async def xread(self, *args, **kwargs):
            raise RedisTimeoutError("socket timeout")

        async def hgetall(self, key):
            return {"status": "running"}

    stream = SynapseRunStore(TimeoutRedis()).read_events("request-1")

    assert await anext(stream) == ("", {})
    await stream.aclose()


@pytest.mark.asyncio
async def test_event_stream_stops_after_timeout_for_finished_run() -> None:
    class TimeoutRedis:
        async def xread(self, *args, **kwargs):
            raise RedisTimeoutError("socket timeout")

        async def hgetall(self, key):
            return {"status": "done"}

    stream = SynapseRunStore(TimeoutRedis()).read_events("request-1")

    with pytest.raises(StopAsyncIteration):
        await anext(stream)
