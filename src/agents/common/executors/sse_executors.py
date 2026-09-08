import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import Request
from loguru import logger

from src.agents.model_clients.base_client import BaseLlmClient

StreamGenerator = Callable[..., AsyncIterator[dict[str, Any]]]

# Generator kwargs that are safe and meaningful to log. Everything else passed
# to the pipeline (mcp_client, effects_mcp_client, ...) is a client object that
# may carry the auth token, so it must never be dumped into the logs.
_LOGGABLE_PARAMS = (
    "user_query",
    "scenario_id",
    "chat_id",
    "request_id",
    "temperature",
)


def _log_stream_request(
    request: Request,
    model: str,
    rerun: bool,
    kwargs: dict[str, Any],
) -> None:
    """
    Log the incoming SSE request with full, non-sensitive information.

    Captures the HTTP envelope (method, URL, client, query params) together with
    the meaningful pipeline parameters, so that if the stream dies the logs show
    exactly which request was in flight. The auth token is never logged — only
    whether an ``Authorization`` header was present.

    Args:
        request (Request): Current FastAPI Request object.
        model (str): Model the pipeline runs on.
        rerun (bool): Whether the pipeline retries on error.
        kwargs (dict[str, Any]): Keyword arguments passed to the generator.
    """

    client = request.client
    request_info = {
        "method": request.method,
        "url": str(request.url),
        "client": f"{client.host}:{client.port}" if client else None,
        "query_params": dict(request.query_params),
        "has_auth_header": "authorization" in request.headers,
        "model": model,
        "rerun": rerun,
        "params": {key: kwargs[key] for key in _LOGGABLE_PARAMS if key in kwargs},
    }
    logger.info(f"SSE stream request started: {request_info}")


async def stream_with_error_handling(
    generator: StreamGenerator,
    request: Request,
    llm_client: BaseLlmClient,
    model: str,
    rerun: bool = True,
    continue_on_disconnect: bool = False,
    *args: Any,
    **kwargs: Any,
) -> AsyncIterator[dict[str, Any]]:
    """
    Universal SSE-safe wrapper around an async generator.

    Args:
        generator: Function that returns an async iterator/generator.
        request: Current FastAPI Request object.
        llm_client (BaseLlmClient): Kept in the shared wrapper contract for callers;
            internal errors are never sent back to the model for explanation.
        model (str): Model to run generation on.
        rerun (bool): Weather try to rerun pipeline if raised error or not.
        *args: Positional arguments passed to the generator.
        **kwargs: Keyword arguments passed to the generator.

    Yields:
        Items produced by the original generator, or error/done events.
    """

    _log_stream_request(request, model, rerun, kwargs)

    try:
        stream = generator(model=model, *args, **kwargs)
        async for item in stream:
            if await request.is_disconnected():
                logger.info("Client disconnected during stream")
                if continue_on_disconnect:
                    asyncio.create_task(_drain_disconnected_stream(stream))
                return

            yield item

    except asyncio.CancelledError:
        logger.info("Stream cancelled")
        raise

    except Exception as exc:
        logger.opt(exception=exc).error("Unhandled exception while running pipeline")
        if rerun:
            logger.info("Trying to re-run pipeline")
            yield {
                "type": "status",
                "message": "При извлечении запроса произошла ошибка. Производится попытка перезапуска запроса.",
            }
            try:
                yield {"type": "status"}
                async for item in generator(model=model, *args, **kwargs):
                    if await request.is_disconnected():
                        logger.info("Client disconnected during stream, finishing")
                        return

                    yield item
                return
            except Exception as retry_exc:
                logger.opt(exception=retry_exc).error(
                    "Couldn't re-run pipeline on retry, needs manual check"
                )

        # Never ask the same model that may have caused the failure to explain it.
        # The full exception is already in server logs; clients receive neither a
        # speculative diagnosis nor internal paths and stack frames.
        yield {
            "type": "chunk",
            "content": {
                "text": (
                    "Не удалось выполнить запрос из-за внутренней ошибки сервера. "
                    "Повторите попытку позже."
                ),
                "done": False,
            },
        }
        yield {
            "type": "error",
            "content": {
                "message": "Internal stream exception",
                "traceback": "",
            },
        }
        yield {
            "type": "chunk",
            "content": {
                "text": "",
                "done": True,
            },
        }
        return


async def _drain_disconnected_stream(stream: AsyncIterator[dict[str, Any]]) -> None:
    """Keep an SSE-authoritative pipeline advancing after a network disconnect."""

    try:
        async for _ in stream:
            pass
    except asyncio.CancelledError:
        logger.info("Disconnected background pipeline was explicitly cancelled")
        return
    except Exception as exc:
        logger.opt(exception=exc).error(
            "Disconnected background pipeline failed while draining"
        )
