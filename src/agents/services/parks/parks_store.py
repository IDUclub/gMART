import asyncio
import json
from contextlib import asynccontextmanager
from uuid import uuid4


class ParksStore:
    """Day-long dialogue, request checkpoints and an ordered SSE journal in Redis.

    Request snapshots and emitted events commit together. No bearer tokens are stored.
    Reads use the existing bounded retry policy of PipelineStateStore.
    """

    TTL = 86400

    def __init__(self, pipeline_store):
        self.pipeline = pipeline_store
        self.redis = pipeline_store._redis

    async def get(self, key):
        raw = await self.pipeline._retry(self.redis.get, "parks:" + key)
        return json.loads(raw) if raw else None

    async def put(self, key, value, *, create=False):
        return await self.pipeline._retry(
            self.redis.set,
            "parks:" + key,
            json.dumps(value, ensure_ascii=False),
            ex=self.TTL,
            nx=create,
        )

    async def create(self, state):
        """Publish the initial checkpoint and first event in one transaction."""
        key = "parks:run:" + state["request_id"]
        async with self.redis.pipeline(transaction=True) as transaction:
            await transaction.watch(key)
            if await transaction.exists(key):
                return False
            state["event_seq"] = 1
            event = {
                "type": "pipeline_started",
                "content": {"request_id": state["request_id"], "event_id": 1},
            }
            transaction.multi()
            transaction.set(key, json.dumps(state, ensure_ascii=False), ex=self.TTL)
            event_key = "parks:events:" + state["request_id"]
            transaction.rpush(event_key, json.dumps(event, ensure_ascii=False))
            transaction.expire(event_key, self.TTL)
            await transaction.execute()
            return True

    async def save(self, state, event=None):
        # Mutations are deliberately not retried as a group: a lost reply must
        # never append an SSE frame twice.
        async with self.redis.pipeline(transaction=True) as transaction:
            transaction.set(
                "parks:run:" + state["request_id"],
                json.dumps(state, ensure_ascii=False),
                ex=self.TTL,
            )
            if event is not None:
                key = "parks:events:" + state["request_id"]
                events = event if isinstance(event, list) else [event]
                transaction.rpush(
                    key, *(json.dumps(item, ensure_ascii=False) for item in events)
                )
                transaction.expire(key, self.TTL)
            await transaction.execute()

    async def events(self, request_id, after):
        rows = await self.pipeline._retry(
            self.redis.lrange, "parks:events:" + request_id, after, -1
        )
        return [json.loads(row) for row in rows]

    @asynccontextmanager
    async def execution_lock(self, name):
        """Renewed cross-worker lease; crash recovery waits at most 30 seconds."""
        key, lease = "parks:lease:" + name, str(uuid4())
        if not await self.pipeline._retry(self.redis.set, key, lease, nx=True, ex=30):
            raise ValueError("Park workflow already running")
        owner = asyncio.current_task()

        async def update(release=False):
            async with self.redis.pipeline(transaction=True) as transaction:
                await transaction.watch(key)
                if await transaction.get(key) != lease:
                    return False
                transaction.multi()
                if release:
                    transaction.delete(key)
                else:
                    transaction.expire(key, 30)
                await transaction.execute()
                return True

        async def renew():
            try:
                while True:
                    await asyncio.sleep(5)
                    if not await update():
                        owner.cancel()
                        return
            except asyncio.CancelledError:
                raise
            except Exception:
                owner.cancel()

        renewal = asyncio.create_task(renew())
        try:
            yield
        finally:
            renewal.cancel()
            await asyncio.gather(renewal, return_exceptions=True)
            await update(release=True)
