import asyncio
import logging
import traceback
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import Request

logger = logging.getLogger(__name__)


StreamGenerator = Callable[..., AsyncIterator[dict[str, Any]]]


async def stream_with_error_handling(
    generator: StreamGenerator,
    request: Request,
    *args: Any,
    **kwargs: Any,
) -> AsyncIterator[dict[str, Any]]:
    """
    Universal SSE-safe wrapper around an async generator.

    Args:
        generator: Function that returns an async iterator/generator.
        request: Current FastAPI Request object.
        *args: Positional arguments passed to the generator.
        **kwargs: Keyword arguments passed to the generator.

    Yields:
        Items produced by the original generator, or error/done events.
    """

    try:
        async for item in generator(*args, **kwargs):
            if await request.is_disconnected():
                logger.info("Client disconnected during stream")
                return

            yield item

    except asyncio.CancelledError:
        logger.info("Stream cancelled")
        raise

    except Exception as exc:
        tb_str = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        logger.exception("Unhandled exception during stream execution")

        yield {
            "type": "error",
            "content": {
                "message": "Internal stream exception",
                "traceback": tb_str,
            },
        }
        yield {
            "type": "done",
            "content": {
                "done": True,
                "success": False,
            },
        }
        return