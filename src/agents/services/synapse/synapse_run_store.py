from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import TimeoutError as RedisTimeoutError


class SynapseIdempotencyConflict(ValueError):
    pass


class SynapseRunStore:
    """Redis persistence and correlation indexes for Synapse runs."""

    ACTIVE_STATUSES = {"starting", "running", "start_unknown"}

    def __init__(self, redis: Redis, *, ttl_seconds: int = 86400) -> None:
        self.redis = redis
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _state_key(request_id: str) -> str:
        return f"synapse:run:{request_id}:state"

    @staticmethod
    def _events_key(request_id: str) -> str:
        return f"synapse:run:{request_id}:events"

    @staticmethod
    def _seen_key(request_id: str) -> str:
        return f"synapse:run:{request_id}:seen"

    @staticmethod
    def _idempotency_key(user_id: str, key: str) -> str:
        return f"synapse:idempotency:{user_id}:{key}"

    async def claim_idempotency(
        self,
        *,
        user_id: str,
        key: str,
        payload_hash: str,
        request_id: str,
    ) -> tuple[str, bool]:
        redis_key = self._idempotency_key(user_id, key)
        value = json.dumps(
            {"request_id": request_id, "payload_hash": payload_hash},
            separators=(",", ":"),
        )
        claimed = await self.redis.set(redis_key, value, nx=True, ex=self.ttl_seconds)
        if claimed:
            return request_id, True
        existing_raw = await self.redis.get(redis_key)
        if not existing_raw:
            return await self.claim_idempotency(
                user_id=user_id,
                key=key,
                payload_hash=payload_hash,
                request_id=request_id,
            )
        existing = json.loads(existing_raw)
        if existing.get("payload_hash") != payload_hash:
            raise SynapseIdempotencyConflict(
                "Idempotency-Key was already used with another payload"
            )
        return str(existing["request_id"]), False

    async def create_state(self, request_id: str, state: dict[str, Any]) -> None:
        mapping = {key: self._encode(value) for key, value in state.items()}
        await self.redis.hset(self._state_key(request_id), mapping=mapping)
        await self.redis.expire(self._state_key(request_id), self.ttl_seconds)
        await self.redis.sadd("synapse:runs:active", request_id)

    async def update_state(self, request_id: str, **changes: Any) -> None:
        if not changes:
            return
        await self.redis.hset(
            self._state_key(request_id),
            mapping={key: self._encode(value) for key, value in changes.items()},
        )
        await self.redis.expire(self._state_key(request_id), self.ttl_seconds)
        status = changes.get("status")
        if status and status not in self.ACTIVE_STATUSES:
            await self.redis.srem("synapse:runs:active", request_id)

    async def get_state(self, request_id: str) -> dict[str, Any] | None:
        raw = await self.redis.hgetall(self._state_key(request_id))
        if not raw:
            return None
        return {key: self._decode(value) for key, value in raw.items()}

    async def bind_project(
        self,
        request_id: str,
        project_id: str,
        *,
        run_id: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        await self.redis.set(
            f"synapse:project:{project_id}", request_id, ex=self.ttl_seconds
        )
        if run_id:
            await self.redis.set(
                f"synapse:run-id:{run_id}", request_id, ex=self.ttl_seconds
            )
        if chat_id:
            await self.redis.set(
                f"synapse:chat:{chat_id}:request", request_id, ex=self.ttl_seconds
            )

    async def claim_chat(self, chat_id: str, request_id: str) -> bool:
        key = f"synapse:chat:{chat_id}:active"
        claimed = await self.redis.set(key, request_id, nx=True, ex=self.ttl_seconds)
        if claimed:
            return True
        existing = await self.redis.get(key)
        if existing == request_id:
            return True
        existing_state = await self.get_state(str(existing)) if existing else None
        if existing_state and existing_state.get("status") in self.ACTIVE_STATUSES:
            return False
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
          return redis.call('del', KEYS[1])
        end
        return 0
        """
        if existing:
            await self.redis.eval(script, 1, key, existing)
        return bool(await self.redis.set(key, request_id, nx=True, ex=self.ttl_seconds))

    async def release_chat(self, chat_id: str, request_id: str) -> None:
        key = f"synapse:chat:{chat_id}:active"
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
          return redis.call('del', KEYS[1])
        end
        return 0
        """
        await self.redis.eval(script, 1, key, request_id)

    async def resolve_a2a_user(
        self, *, project_id: str | None, run_id: str | None
    ) -> str | None:
        request_id = None
        if run_id:
            request_id = await self.redis.get(f"synapse:run-id:{run_id}")
        if not request_id and project_id:
            request_id = await self.redis.get(f"synapse:project:{project_id}")
        if not request_id:
            return None
        state = await self.get_state(str(request_id))
        if not state:
            return None
        if project_id and state.get("synapse_project_id") != project_id:
            return None
        if run_id and state.get("run_id") not in (None, "", run_id):
            return None
        user_id = state.get("user_id")
        return str(user_id) if user_id else None

    async def add_event(
        self,
        request_id: str,
        *,
        source_event_id: str,
        event: dict[str, Any],
    ) -> str | None:
        seen_key = self._seen_key(request_id)
        if not await self.redis.sadd(seen_key, source_event_id):
            return None
        await self.redis.expire(seen_key, self.ttl_seconds)
        stream_id = await self.redis.xadd(
            self._events_key(request_id),
            {"event": json.dumps(event, ensure_ascii=False, separators=(",", ":"))},
            maxlen=5000,
            approximate=True,
        )
        await self.redis.expire(self._events_key(request_id), self.ttl_seconds)
        await self.update_state(
            request_id,
            last_event_id=source_event_id,
            last_stream_id=str(stream_id),
        )
        return str(stream_id)

    async def read_events(
        self, request_id: str, *, after: str = "0-0", block_ms: int = 15000
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        cursor = after or "0-0"
        while True:
            try:
                rows = await self.redis.xread(
                    {self._events_key(request_id): cursor},
                    block=block_ms,
                    count=100,
                )
            except RedisTimeoutError:
                # A Redis socket timeout shorter than XREAD's block is an empty poll,
                # not an SSE failure. Re-check terminal state and emit a heartbeat for
                # active runs exactly as when Redis returns no rows.
                rows = []
            if not rows:
                state = await self.get_state(request_id)
                if not state or state.get("status") not in self.ACTIVE_STATUSES:
                    return
                yield "", {}
                continue
            for _, messages in rows:
                for stream_id, fields in messages:
                    cursor = str(stream_id)
                    yield cursor, json.loads(fields["event"])

    async def active_request_ids(self) -> list[str]:
        values = await self.redis.smembers("synapse:runs:active")
        return [str(value) for value in values]

    async def acquire_relay(self, request_id: str, owner: str, ttl: int = 30) -> bool:
        return bool(
            await self.redis.set(
                f"synapse:relay:{request_id}:owner", owner, nx=True, ex=ttl
            )
        )

    async def renew_relay(self, request_id: str, owner: str, ttl: int = 30) -> bool:
        key = f"synapse:relay:{request_id}:owner"
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
          return redis.call('expire', KEYS[1], ARGV[2])
        end
        return 0
        """
        return bool(await self.redis.eval(script, 1, key, owner, ttl))

    async def release_relay(self, request_id: str, owner: str) -> None:
        key = f"synapse:relay:{request_id}:owner"
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
          return redis.call('del', KEYS[1])
        end
        return 0
        """
        await self.redis.eval(script, 1, key, owner)

    @staticmethod
    def _encode(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, (dict, list, bool, int, float)):
            return json.dumps(value, separators=(",", ":"))
        return str(value)

    @staticmethod
    def _decode(value: str) -> Any:
        if value == "null":
            return None
        if (value and value[0] in "[{") or value in {"true", "false"}:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value
