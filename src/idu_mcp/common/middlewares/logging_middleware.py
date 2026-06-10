"""HTTP request logging middleware for the idu_mcp Starlette app."""

import time

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every incoming HTTP request and its outcome via loguru.

    Logs the method, path, client host, response status and elapsed time.
    Headers and query params are intentionally not logged to avoid leaking
    auth tokens.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.perf_counter()
        client = request.client.host if request.client else "unknown"
        logger.info(f"--> {request.method} {request.url.path} from {client}")
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                f"<-- {request.method} {request.url.path} raised after {elapsed_ms:.1f} ms"
            )
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"<-- {request.method} {request.url.path} {response.status_code} "
            f"in {elapsed_ms:.1f} ms"
        )
        return response
