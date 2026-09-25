"""One document producer per request; HTTP clients only subscribe to its journal."""

import asyncio
import time

from loguru import logger

from src.agents.common.exceptions.base_exceptions import (
    AgentsInputException,
    AgentsNotFound,
    AgentsUnauthorizedException,
)
from src.agents.model_clients.llm_pace import PacedDeadline, PipelineDeadlineExceeded
from src.agents.services.pipeline_state import PIPELINE_TTL, PipelineStatus

# Seconds of work at the LLM's nominal speed; stretched while the server is slow.
RUN_DEADLINE_SECONDS = PIPELINE_TTL - 30
# The run may outlive PIPELINE_TTL, so its Redis keys are refreshed while it works.
KEEP_ALIVE_SECONDS = 60

_tasks: set[asyncio.Task] = set()


async def run_metadata(store, request_id):
    return await store.get_document_run(request_id)


async def validate_resume(store, request_id, after_event, owner):
    if not request_id:
        if after_event:
            raise AgentsInputException("after_event requires request_id")
        return
    meta = await run_metadata(store, request_id)
    if meta is None:
        raise AgentsNotFound(
            "Срок восстановления запроса истёк. Запустите новый запрос."
        )
    if meta["owner"] != owner:
        raise AgentsUnauthorizedException(
            "Этот запрос принадлежит другому пользователю."
        )
    if after_event > len(await store.get_buffered_events(request_id)):
        raise AgentsInputException("after_event exceeds the event journal")


def _producer_finished(task):
    _tasks.discard(task)
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.opt(exception=error).error("Document producer journal update failed")


async def _produce(service, request_id, owner, kwargs):
    store = service.state_store
    status = "running"

    async def metadata():
        await store.save_document_run(
            request_id, dict(owner=owner, status=status, heartbeat=time.time())
        )

    async def consume():
        async for _ in service.run_document_qa_pipeline(
            request_id=request_id, **kwargs
        ):
            pass  # The service commits each event before yielding it.

    task = asyncio.create_task(consume())
    deadline = PacedDeadline(RUN_DEADLINE_SECONDS)
    heartbeat_at = deadline.started - 5
    kept_alive_at = deadline.started
    try:
        while not task.done():
            if await store.is_cancelled(request_id):
                status = "cancelled"
                task.cancel()
                break
            if deadline.exceeded():
                raise PipelineDeadlineExceeded(
                    "document run deadline exceeded", deadline.wall_seconds
                )
            if time.monotonic() - heartbeat_at >= 5:
                await metadata()
                heartbeat_at = time.monotonic()
            if time.monotonic() - kept_alive_at >= KEEP_ALIVE_SECONDS:
                await store.keep_alive(request_id)
                kept_alive_at = time.monotonic()
            await asyncio.wait({task}, timeout=0.25)
        if status == "cancelled":
            await asyncio.gather(task, return_exceptions=True)
            await store.buffer_event(
                request_id,
                dict(
                    type="error",
                    content=dict(
                        message="Запрос остановлен пользователем.", traceback=""
                    ),
                ),
            )
        else:
            await task
            status = "done"
    except (Exception, asyncio.CancelledError) as exc:
        status = "failed"
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        logger.opt(exception=exc).error(
            "Document producer stopped request_id={} reason={} error_type={}",
            request_id,
            getattr(exc, "reason", "pipeline_error"),
            type(exc).__name__,
        )
        await store.buffer_event(
            request_id,
            dict(
                type="error",
                content=dict(
                    message=(
                        exc.user_message
                        if isinstance(exc, PipelineDeadlineExceeded)
                        else "Не удалось завершить запрос. Повторите попытку."
                    ),
                    traceback="",
                ),
            ),
        )
        await store.set_status(request_id, PipelineStatus.FAILED)
        if isinstance(exc, asyncio.CancelledError):
            raise
    finally:
        await metadata()


async def stream_document_run(
    service, *, request_id=None, after_event=0, owner=None, **kwargs
):
    store = service.state_store
    await validate_resume(store, request_id, after_event, owner)
    if request_id:
        meta = await run_metadata(store, request_id)
    else:
        request_id = store.new_request_id()
        meta = dict(owner=owner, status="running", heartbeat=time.time())
        claimed = await store.save_document_run(request_id, meta, create=True)
        if not claimed:
            raise RuntimeError("document request identity collision")
        task = asyncio.create_task(_produce(service, request_id, owner, kwargs))
        _tasks.add(task)
        task.add_done_callback(_producer_finished)
    cursor = after_event
    try:
        while True:
            meta = await run_metadata(store, request_id)
            events = await store.get_buffered_events(request_id)
            for index, event in enumerate(events[cursor:], cursor + 1):
                cursor = index
                yield {**event, "event_id": index}
            if meta is None or (
                meta["status"] == "running" and time.time() - meta["heartbeat"] > 30
            ):
                yield dict(
                    type="error",
                    content=dict(
                        message="Выполнение прервано на сервере. Запустите новый запрос.",
                        traceback="",
                    ),
                )
                return
            if meta["status"] != "running":
                return
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        logger.info(
            "Document subscriber disconnected request_id={} after_event={}; producer continues",
            request_id,
            cursor,
        )
        raise
