"""Opt-in fault injection against an isolated Redis, never the shared stack.

Set GMART_TEST_REDIS_URL to a disposable server. Only this client's connection
is killed, and keys use fresh UUIDs. No FLUSHDB/FLUSHALL is used.
"""

import os
import uuid

import pytest
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import ConnectionError

from src.agents.common.logging.redis_logging import LoggedRedis
from src.agents.services.pipeline_state import PipelineStateStore, PipelineStatus

pytestmark = pytest.mark.integration


@pytest.fixture
async def isolated_redis():
    url = os.getenv("GMART_TEST_REDIS_URL")
    if not url:
        pytest.skip("GMART_TEST_REDIS_URL must name an isolated test Redis")
    client = LoggedRedis.from_url(
        url,
        decode_responses=True,
        health_check_interval=30,
        socket_connect_timeout=2,
        socket_timeout=2,
        retry=Retry(NoBackoff(), 0),
    )
    await client.ping()
    yield client
    await client.aclose()


async def test_completion_after_server_kills_this_connection(isolated_redis):
    redis = isolated_redis
    rid = str(uuid.uuid4())
    store = PipelineStateStore(redis)
    await store.create(
        rid,
        chat_id=None,
        user_query="test",
        scenario_id=845,
        model="test",
        temperature=0,
    )
    connection_id = await redis.client_id()
    admin = Redis.from_url(os.environ["GMART_TEST_REDIS_URL"])
    try:
        assert await admin.client_kill_filter(_id=connection_id) == 1
        await store.set_status(rid, PipelineStatus.DONE)
        assert (await store.get_state(rid))["status"] == "done"
    finally:
        await admin.aclose()
        await redis.delete(f"pipeline:{rid}:state")


async def test_lost_transaction_ack_is_idempotent_on_real_redis(
    isolated_redis, monkeypatch
):
    redis = isolated_redis
    rid = str(uuid.uuid4())
    store = PipelineStateStore(redis)
    original = redis.pipeline
    lost = False

    def pipeline(*args, **kwargs):
        p = original(*args, **kwargs)
        execute = p.execute

        async def execute_with_lost_ack(*args, **kwargs):
            nonlocal lost
            result = await execute(*args, **kwargs)
            if not lost:
                lost = True
                await redis.connection_pool.disconnect()
                raise ConnectionError("Injected lost EXEC acknowledgement after commit")
            return result

        p.execute = execute_with_lost_ack
        return p

    monkeypatch.setattr(redis, "pipeline", pipeline)
    event = {"type": "chunk", "content": {"text": "persist exactly once"}}
    try:
        await store.buffer_event(rid, event)
        assert lost
        assert await store.get_buffered_events(rid) == [event]
        assert await redis.ttl(f"pipeline:{rid}:event_ids") > 0
    finally:
        await redis.delete(f"pipeline:{rid}:events", f"pipeline:{rid}:event_ids")


async def test_lock_retry_does_not_steal_or_release_another_owner(isolated_redis):
    store = PipelineStateStore(isolated_redis)
    cid = str(uuid.uuid4())
    try:
        assert await store.acquire_chat(cid, "first")
        assert await store.acquire_chat(cid, "first")
        assert not await store.acquire_chat(cid, "second")
        await store.release_chat(cid, "second")
        assert await isolated_redis.get(f"pipeline:{cid}:active_request") == "first"
        await store.release_chat(cid, "first")
        assert await store.acquire_chat(cid, "second")
    finally:
        await isolated_redis.delete(f"pipeline:{cid}:active_request")
