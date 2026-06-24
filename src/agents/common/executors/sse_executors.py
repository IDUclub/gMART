import asyncio
import traceback
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
    *args: Any,
    **kwargs: Any,
) -> AsyncIterator[dict[str, Any]]:
    """
    Universal SSE-safe wrapper around an async generator.

    Args:
        generator: Function that returns an async iterator/generator.
        request: Current FastAPI Request object.
        llm_client (BaseLlmClient): BaseLlmClient object for generating responses via llm.
        model (str): Model to run generation on.
        rerun (bool): Weather try to rerun pipeline if raised error or not.
        *args: Positional arguments passed to the generator.
        **kwargs: Keyword arguments passed to the generator.

    Yields:
        Items produced by the original generator, or error/done events.
    """

    _log_stream_request(request, model, rerun, kwargs)

    try:
        async for item in generator(model=model, *args, **kwargs):
            if await request.is_disconnected():
                logger.info("Client disconnected during stream")
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
            except Exception as retry_exc:
                logger.opt(exception=retry_exc).error(
                    "Couldn't re-run pipeline on retry, needs manual check"
                )
                exc = retry_exc

        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        logger.info("Streaming user-facing error explanation to the client")
        # TODO pass all history of generated prompts and messages.
        messages = [
            {
                "role": "system",
                "content": f"""
                В ходе выполнения пайплайна извлечения нормативных требований произошла следующая
                ошибка\n:
                {tb_str}

                Объясни пользователю что случилось простым языком, не вдаваясь в технические детали.
                Если проблема может быть исправлена более точным запросом, укажи на это и скажи, как его можно
                переформулировать.
                """,
            },
        ]
        try:
            async for chunk in llm_client.execute_request(model, messages):
                if chunk["content"]["done"]:
                    chunk["content"]["done"] = False
                yield chunk
        except Exception as explain_exc:
            # The fallback explanation itself failed — commonly the LLM backend
            # is the root cause and is unreachable. Log the full traceback and
            # still close the stream cleanly below, instead of letting the
            # exception escape the generator and silently drop the connection.
            logger.opt(exception=explain_exc).error(
                "Failed to generate user-facing error explanation via LLM"
            )
        yield {
            "type": "error",
            "content": {
                "message": "Internal stream exception",
                "traceback": tb_str,
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
