"""Pipeline state in Redis: one hash field per step, large payloads compressed,
and a null store for runs that need no state at all.

Checkpoints and buffered events hold whole GeoJSON tool results. Rewriting the
accumulated blob on every step made the traffic grow with the square of the number
of steps, and on the largest scenarios the write stalled until the socket timed
out, killing the run. Benchmarks have no use for any of it — they never reconnect
— so they switch the store off entirely.
"""

from __future__ import annotations

import json

import pytest

from src.agents.services.pipeline_state import (
    REDIS_COMPRESS_MIN_BYTES,
    NullPipelineStateStore,
    PipelineStatus,
    pipeline_state_disabled,
)


def _big_layer(n: int = 4000) -> dict:
    return {
        "objects": [
            {
                "geometry": {"type": "Point", "coordinates": [30.1 + i, 59.9]},
                "properties": {"name": "жилой дом"},
            }
            for i in range(n)
        ]
    }


async def test_checkpoint_round_trip_per_step(state_store):
    await state_store.save_checkpoint("req-1", "plan", {"mode": "restrictions"})
    await state_store.save_checkpoint("req-1", "plan_explanation", True)
    await state_store.save_checkpoint("req-1", "layers", _big_layer(50))

    checkpoint = await state_store.get_checkpoint("req-1")

    assert checkpoint["plan"] == {"mode": "restrictions"}
    assert checkpoint["plan_explanation"] is True
    assert checkpoint["layers"] == _big_layer(50)


async def test_missing_checkpoint_is_empty(state_store):
    assert await state_store.get_checkpoint("nothing-here") == {}


async def test_large_payload_is_compressed(state_store):
    payload = _big_layer()
    raw_size = len(json.dumps(payload, ensure_ascii=False))
    assert raw_size > REDIS_COMPRESS_MIN_BYTES

    await state_store.save_checkpoint("req-2", "layers", payload)

    stored = await state_store._redis.hget("pipeline:req-2:checkpoint", "layers")
    assert stored.startswith("gz:")
    assert len(stored) < raw_size / 5
    assert (await state_store.get_checkpoint("req-2"))["layers"] == payload


async def test_saving_a_step_does_not_rewrite_the_previous_ones(state_store):
    """The whole point: a small step must cost a small write, however much
    geometry an earlier step already put in the checkpoint."""

    writes: list[int] = []
    real_hset = state_store._redis.hset

    async def spy(key, field=None, value=None, **kwargs):
        writes.append(len(value))
        return await real_hset(key, field, value, **kwargs)

    state_store._redis.hset = spy

    await state_store.save_checkpoint("req-3", "layers", _big_layer())
    await state_store.save_checkpoint("req-3", "final_response", True)

    assert writes[0] > 1000
    assert writes[1] < 100


async def test_checkpoint_written_by_an_older_build_is_discarded(state_store):
    """Before this change the checkpoint was a plain string at the same key;
    hitting one mid-deploy must not break the pipeline."""

    await state_store._redis.set("pipeline:req-4:checkpoint", json.dumps({"plan": 1}))

    assert await state_store.get_checkpoint("req-4") == {}
    await state_store.save_checkpoint("req-4", "plan", {"mode": "buffers_only"})
    assert await state_store.get_checkpoint("req-4") == {
        "plan": {"mode": "buffers_only"}
    }


async def test_buffered_events_are_compressed_and_replay_intact(state_store):
    """Replay reads these back verbatim, and a feature_collection event carries the
    same geometry as the checkpoint — so it goes through Redis the same way."""

    small = {"type": "status", "content": {"text": "Строю буферы"}}
    big = {"type": "feature_collection", "content": _big_layer()}

    await state_store.buffer_event("req-6", small)
    await state_store.buffer_event("req-6", big)

    stored = await state_store._redis.lrange("pipeline:req-6:events", 0, -1)
    assert not stored[0].startswith("gz:")
    assert stored[1].startswith("gz:")
    assert len(stored[1]) < len(json.dumps(big, ensure_ascii=False)) / 5
    assert await state_store.get_buffered_events("req-6") == [small, big]


async def test_checkpoint_expires(state_store):
    await state_store.save_checkpoint("req-5", "plan", {"mode": "restrictions"})

    assert await state_store._redis.ttl("pipeline:req-5:checkpoint") > 0


@pytest.mark.parametrize(
    "value,expected",
    [("1", True), ("true", True), ("ON", True), ("0", False), ("", False)],
)
def test_disable_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("DISABLE_PIPELINE_STATE", value)
    assert pipeline_state_disabled() is expected


async def test_null_store_keeps_nothing_and_never_touches_redis():
    """It is constructed without a client at all, so any Redis call would raise."""

    store = NullPipelineStateStore()

    await store.create(
        "req",
        chat_id=None,
        user_query="q",
        scenario_id=1,
        model="gemma-3-27b",
        temperature=0.0,
    )
    await store.save_checkpoint("req", "layers", {"objects": [1, 2, 3]})
    await store.buffer_event("req", {"type": "chunk"})
    await store.set_status("req", PipelineStatus.DONE)

    assert await store.exists("req") is False
    assert await store.get_checkpoint("req") == {}
    assert await store.get_buffered_events("req") == []
    assert await store.get_state("req") is None
    assert await store.provide_token("req", "new") == 0


async def test_null_store_suspends_instead_of_waiting_for_a_token():
    """Nobody can publish a token without Redis, so the step must give up at once
    rather than sit out the refresh timeout."""

    with pytest.raises(TimeoutError):
        await NullPipelineStateStore().wait_for_token("req")
