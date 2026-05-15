from __future__ import annotations

import json
import uuid
from enum import StrEnum
from typing import Any

import redis.asyncio as aioredis
from loguru import logger

TOKEN_REFRESH_TIMEOUT: float = 360.0
PIPELINE_TTL: int = 360  # seconds


class PipelineStatus(StrEnum):
    RUNNING = "running"
    WAITING_TOKEN = "waiting_token"
    SUSPENDED = "suspended"
    DONE = "done"
    FAILED = "failed"


class PipelineStep(StrEnum):
    PLAN = "plan"
    PLAN_EXPLANATION = "plan_explanation"
    LAYERS = "layers"
    BUFFERS = "buffers"
    RESTRICTIONS = "restrictions"
    FINAL_RESPONSE = "final_response"


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
        raw = await self._redis.get(self._key(request_id, "checkpoint"))
        checkpoint: dict = json.loads(raw) if raw else {}
        checkpoint[step] = data
        await self._redis.setex(
            self._key(request_id, "checkpoint"),
            PIPELINE_TTL,
            json.dumps(checkpoint, ensure_ascii=False),
        )

    async def get_checkpoint(self, request_id: str) -> dict:
        raw = await self._redis.get(self._key(request_id, "checkpoint"))
        return json.loads(raw) if raw else {}

    async def buffer_event(self, request_id: str, event: dict) -> None:
        key = self._key(request_id, "events")
        await self._redis.rpush(key, json.dumps(event, ensure_ascii=False))
        await self._redis.expire(key, PIPELINE_TTL)

    async def get_buffered_events(self, request_id: str) -> list[dict]:
        raw_list = await self._redis.lrange(self._key(request_id, "events"), 0, -1)
        return [json.loads(r) for r in raw_list]

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
