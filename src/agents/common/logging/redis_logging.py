"""Redis command metadata only: never serialize arguments, replies or exceptions."""

import asyncio
import os
import time
import uuid
from contextvars import ContextVar

from loguru import logger
from redis.asyncio import Redis
from redis.asyncio.client import Pipeline, PubSub
from redis.exceptions import ResponseError

redis_attempt = ContextVar("redis_attempt", default=1)
redis_request_id = ContextVar("redis_request_id", default=None)

_COMMANDS = set(
    "GET SET SETEX EXISTS EXPIRE DEL TTL LRANGE RPUSH HGET HGETALL HSET SADD SREM SMEMBERS SISMEMBER XADD XREAD EVAL EVALSHA WATCH UNWATCH MULTI EXEC PIPELINE PING SUBSCRIBE PSUBSCRIBE UNSUBSCRIBE PUNSUBSCRIBE PUBLISH PUBSUB_READ".split()
)
_KINDS = {
    "state",
    "checkpoint",
    "events",
    "event_ids",
    "active_request",
    "token_channel",
    "seen",
}


def _command(value):
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="replace")
    return value if isinstance(value, str) and value in _COMMANDS else "OTHER"


def _request_id(value):
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return "-"


def _metadata(pool, args):
    config = pool.connection_kwargs
    # Only allowlisted connection fields; never repr(pool), URL, username or password.
    host = config.get("host", "local")
    host = (
        "".join(c for c in host if c.isalnum() or c in ".:-_")[:253]
        if isinstance(host, str)
        else "local"
    )
    record = dict(
        command=_command(args[0]) if args else "OTHER",
        endpoint=f"{host}:{config.get('port', 6379)}/{config.get('db', 0)}",
        pid=os.getpid(),
        request_id=_request_id(redis_request_id.get()),
        key_kind="other",
        attempt=redis_attempt.get(),
    )
    key = args[1] if len(args) > 1 else None
    if isinstance(key, bytes):
        key = key.decode("utf-8", errors="replace")
    if isinstance(key, str):
        parts = key.split(":")
        if len(parts) == 3 and parts[0] == "pipeline" and parts[2] in _KINDS:
            record["key_kind"] = "pipeline:" + parts[2]
            if record["request_id"] == "-" and parts[2] != "active_request":
                record["request_id"] = _request_id(parts[1])
        elif len(parts) == 4 and parts[:2] == ["synapse", "run"] and parts[3] in _KINDS:
            record["key_kind"] = "synapse:" + parts[3]
            record["request_id"] = _request_id(parts[2])
    return record


async def _observe(
    pool, args, operation, *, commands=None, success="ok", log_success=True
):
    record = _metadata(pool, args)
    if commands is not None:
        record["commands"] = ",".join(commands)
        record["command_count"] = len(commands)
    started = time.monotonic()
    level, outcome, error_type = "INFO", success, "-"
    try:
        result = await operation()
        if commands is not None and isinstance(result, list):
            errors = [item for item in result if isinstance(item, ResponseError)]
            if errors:
                level, outcome, error_type = (
                    "ERROR",
                    "partial_error",
                    type(errors[0]).__name__,
                )
        return result
    except asyncio.CancelledError:
        level, outcome, error_type = "WARNING", "cancelled", "CancelledError"
        raise
    except Exception as exc:
        level, outcome, error_type = "ERROR", "error", type(exc).__name__
        raise
    finally:
        record.update(
            outcome=outcome,
            duration_ms=round((time.monotonic() - started) * 1000, 2),
            error_type=error_type,
        )
        if level != "INFO" or (
            log_success
            and os.getenv("REDIS_LOG_COMMANDS", "true").lower()
            not in {"false", "0", "no", "off"}
        ):
            logger.bind(redis=record).log(
                level,
                "Redis " + " ".join(f"{key}={value}" for key, value in record.items()),
            )


class LoggedPipeline(Pipeline):
    async def immediate_execute_command(self, *args, **kwargs):
        return await _observe(
            self.connection_pool,
            args,
            lambda: super(LoggedPipeline, self).immediate_execute_command(
                *args, **kwargs
            ),
        )

    async def execute(self, raise_on_error=True):
        # Capture names before Redis clears command_stack. Queuing is not execution.
        commands = [_command(args[0]) for args, _options in self.command_stack]
        if not commands:
            return await super().execute(raise_on_error=raise_on_error)
        first = self.command_stack[0][0]
        operation = (
            "EXEC" if self.is_transaction or self.explicit_transaction else "PIPELINE"
        )
        return await _observe(
            self.connection_pool,
            (operation, *first[1:2]),
            lambda: super(LoggedPipeline, self).execute(raise_on_error=raise_on_error),
            commands=commands,
        )


class LoggedPubSub(PubSub):
    async def execute_command(self, *args):
        # Pub/sub acknowledges asynchronously; this record confirms sending only.
        return await _observe(
            self.connection_pool,
            args,
            lambda: super(LoggedPubSub, self).execute_command(*args),
            success="sent",
        )

    async def parse_response(self, block=True, timeout=0):
        # Report receive failures too, without logging messages or empty poll cycles.
        return await _observe(
            self.connection_pool,
            ("PUBSUB_READ",),
            lambda: super(LoggedPubSub, self).parse_response(
                block=block, timeout=timeout
            ),
            log_success=False,
        )


class LoggedRedis(Redis):
    async def execute_command(self, *args, **kwargs):
        return await _observe(
            self.connection_pool,
            args,
            lambda: super(LoggedRedis, self).execute_command(*args, **kwargs),
        )

    def pipeline(self, transaction=True, shard_hint=None):
        return LoggedPipeline(
            self.connection_pool, self.response_callbacks, transaction, shard_hint
        )

    def pubsub(self, **kwargs):
        return LoggedPubSub(
            self.connection_pool, event_dispatcher=self._event_dispatcher, **kwargs
        )
