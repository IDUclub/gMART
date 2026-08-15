from __future__ import annotations

import base64
import gzip
import json
import os
import uuid
from enum import StrEnum
from typing import Any

import redis.asyncio as aioredis
from loguru import logger

TOKEN_REFRESH_TIMEOUT: float = 360.0
PIPELINE_TTL: int = 360  # seconds

# Checkpoints and buffered events both carry whole GeoJSON tool results: on large
# scenarios a single one is hundreds of megabytes, and writing that to Redis
# uncompressed used to stall the connection until the kernel gave up on the socket.
# GeoJSON compresses by an order of magnitude, and level 1 keeps the CPU cost
# negligible next to the transfer it saves. Values below the threshold stay plain
# JSON so the common small payload remains readable with redis-cli.
REDIS_COMPRESS_MIN_BYTES: int = 32 * 1024
_GZIP_PREFIX = "gz:"


def _pack(data: Any) -> str:
    raw = json.dumps(data, ensure_ascii=False)
    if len(raw) < REDIS_COMPRESS_MIN_BYTES:
        return raw
    packed = gzip.compress(raw.encode("utf-8"), compresslevel=1)
    # The Redis client decodes responses as text, so the gzip bytes travel
    # base64-encoded; that still leaves the payload far smaller than the JSON.
    return _GZIP_PREFIX + base64.b64encode(packed).decode("ascii")


def _unpack(raw: str) -> Any:
    if raw.startswith(_GZIP_PREFIX):
        raw = gzip.decompress(base64.b64decode(raw[len(_GZIP_PREFIX) :])).decode(
            "utf-8"
        )
    return json.loads(raw)


class PipelineStatus(StrEnum):
    RUNNING = "running"
    WAITING_TOKEN = "waiting_token"
    SUSPENDED = "suspended"
    DONE = "done"
    FAILED = "failed"


class PipelineStep(StrEnum):
    NORMGRAPH = "normgraph"
    PLAN = "plan"
    PLAN_EXPLANATION = "plan_explanation"
    LAYERS = "layers"
    BUFFERS = "buffers"
    RESTRICTIONS = "restrictions"
    FINAL_RESPONSE = "final_response"
    # provision effects pipeline steps
    RESOLVE_SERVICE = "resolve_service"
    GET_SERVICE_ID = "get_service_id"
    CALCULATE_EFFECTS = "calculate_effects"
    CALCULATE_PROVISION = "calculate_provision"


class PipelineStateStore:
    """
    Redis-backed store for pipeline state, checkpoints, event buffer,
    and cross-worker token-refresh signalling via pub/sub.
    """

    _PREFIX = "pipeline"

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis

    def _key(self, request_id: str, suffix: str) -> str:
        return f"{self._PREFIX}:{request_id}:{suffix}"

    @staticmethod
    def new_request_id() -> str:
        return str(uuid.uuid4())

    async def exists(self, request_id: str) -> bool:
        return bool(await self._redis.exists(self._key(request_id, "state")))

    async def create(
        self,
        request_id: str,
        *,
        chat_id: str | None,
        user_query: str,
        scenario_id: int,
        model: str,
        temperature: float,
    ) -> None:
        state = {
            "status": PipelineStatus.RUNNING,
            "chat_id": chat_id,
            "user_query": user_query,
            "scenario_id": scenario_id,
            "model": model,
            "temperature": temperature,
        }
        await self._redis.setex(
            self._key(request_id, "state"),
            PIPELINE_TTL,
            json.dumps(state, ensure_ascii=False),
        )

    async def get_state(self, request_id: str) -> dict | None:
        raw = await self._redis.get(self._key(request_id, "state"))
        return json.loads(raw) if raw else None

    async def set_status(self, request_id: str, status: PipelineStatus) -> None:
        raw = await self._redis.get(self._key(request_id, "state"))
        if not raw:
            return
        state = json.loads(raw)
        state["status"] = status
        await self._redis.setex(
            self._key(request_id, "state"),
            PIPELINE_TTL,
            json.dumps(state, ensure_ascii=False),
        )

    async def save_checkpoint(self, request_id: str, step: str, data: Any) -> None:
        """
        Store one pipeline step. Each step is its own hash field: rewriting the
        whole checkpoint on every step made the traffic quadratic in the number
        of steps, which is what pushed large scenarios over the socket timeout.
        """

        key = self._key(request_id, "checkpoint")
        payload = _pack(data)
        try:
            await self._redis.hset(key, step, payload)
        except aioredis.ResponseError:
            # A checkpoint written by an older build is a plain string at this key;
            # it belongs to a pipeline that cannot resume across the deploy anyway.
            logger.warning(f"Dropping legacy checkpoint for {request_id}")
            await self._redis.delete(key)
            await self._redis.hset(key, step, payload)
        await self._redis.expire(key, PIPELINE_TTL)

    async def get_checkpoint(self, request_id: str) -> dict:
        try:
            raw = await self._redis.hgetall(self._key(request_id, "checkpoint"))
        except aioredis.ResponseError:
            logger.warning(f"Ignoring legacy checkpoint for {request_id}")
            return {}
        return {step: _unpack(value) for step, value in raw.items()}

    async def buffer_event(self, request_id: str, event: dict) -> None:
        # feature_collection events carry the same geometry the checkpoint does,
        # so they get the same treatment — replay only ever reads them back here.
        key = self._key(request_id, "events")
        await self._redis.rpush(key, _pack(event))
        await self._redis.expire(key, PIPELINE_TTL)

    async def get_buffered_events(self, request_id: str) -> list[dict]:
        raw_list = await self._redis.lrange(self._key(request_id, "events"), 0, -1)
        return [_unpack(r) for r in raw_list]

    async def wait_for_token(self, request_id: str) -> str:
        """
        Subscribe and block until a new token is published for this request.
        Wrap with asyncio.wait_for to enforce a timeout.
        """
        channel = self._key(request_id, "token_channel")
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    return json.loads(message["data"])["token"]
        finally:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Error closing pubsub for {request_id}: {exc}")
        raise RuntimeError("pubsub closed without receiving token")

    async def provide_token(self, request_id: str, new_token: str) -> int:
        """
        Publish a new token on the request's channel.
        Returns the number of subscribers (0 = nobody is waiting).
        """
        channel = self._key(request_id, "token_channel")
        return await self._redis.publish(channel, json.dumps({"token": new_token}))


def pipeline_state_disabled() -> bool:
    """Whether pipeline state is kept at all (evaluation only).

    Enabled when the ``DISABLE_PIPELINE_STATE`` env var is a truthy value
    (``1``/``true``/``yes``/``on``). The state exists to survive a reconnect or a
    mid-pipeline token refresh; a benchmark run does neither — it reruns the whole
    query instead — so on large scenarios it only pushes megabytes of geometry
    through Redis. Off by default in production, where reconnects are the point.
    """

    return os.getenv("DISABLE_PIPELINE_STATE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class NullPipelineStateStore(PipelineStateStore):
    """A state store that keeps nothing, so the service runs without Redis.

    Reconnect finds no state and starts a fresh run; a token that expires
    mid-pipeline can no longer be handed over, so the step suspends at once
    instead of waiting out the refresh timeout for a message nobody can send.
    """

    def __init__(self) -> None:
        super().__init__(redis=None)  # type: ignore[arg-type]

    async def exists(self, request_id: str) -> bool:
        return False

    async def create(self, request_id: str, **kwargs: Any) -> None:
        return None

    async def get_state(self, request_id: str) -> dict | None:
        return None

    async def set_status(self, request_id: str, status: PipelineStatus) -> None:
        return None

    async def save_checkpoint(self, request_id: str, step: str, data: Any) -> None:
        return None

    async def get_checkpoint(self, request_id: str) -> dict:
        return {}

    async def buffer_event(self, request_id: str, event: dict) -> None:
        return None

    async def get_buffered_events(self, request_id: str) -> list[dict]:
        return []

    async def wait_for_token(self, request_id: str) -> str:
        raise TimeoutError("pipeline state is disabled: no token can be provided")

    async def provide_token(self, request_id: str, new_token: str) -> int:
        return 0
