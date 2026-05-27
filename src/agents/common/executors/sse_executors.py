import asyncio
import logging
import traceback
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import Request

from src.agents.model_clients.base_client import BaseLlmClient

logger = logging.getLogger(__name__)


StreamGenerator = Callable[..., AsyncIterator[dict[str, Any]]]


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
        logger.warning("Unhandled Exception. Faced error during running pipeline")
        logger.exception(exc)
        if rerun:
            logger.info("Trying to re-run pipeline")
            yield {
                "type": "status",
                "content": {
                    "status": "data_retrievement",
                    "text": "При извлечении запроса произошла ошибка. Производится попытка перезапуска запроса.",
                },
            }
            try:
                async for item in generator(model=model, *args, **kwargs):
                    if await request.is_disconnected():
                        logger.warning("Client disconnected during stream")
                        logger.info("Finishing stream fue to client disconnected")
                        return

                    yield item
            except Exception as retry_exc:
                logger.error(
                    "Unhandled Exception. Couldn't re-un pipeline on retry, needs manual check."
                )
                logger.exception(retry_exc)
                exc = retry_exc

        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
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
        async for chunk in llm_client.execute_request(model, messages):
            if chunk["content"]["done"]:
                chunk["content"]["done"] = False
            yield chunk
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
