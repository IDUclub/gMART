"""Exception handling middleware is defined here."""

from typing import Any

from fastapi import FastAPI, Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.agents.common.exceptions.a2a_exceptions import A2AJsonRpcError
from src.agents.common.exceptions.base_exceptions import AgentsBaseException

# Status codes that should not expose internal details to the client.
_CLIENT_ERROR_CODES = frozenset({400, 401, 403, 404, 409, 422})


class ExceptionHandlerMiddleware(
    BaseHTTPMiddleware
):  # pylint: disable=too-few-public-methods
    """Handle exceptions, so they become http response code 500 - Internal Server Error if not handled as HTTPException
    previously.
    Attributes:
           app (FastAPI): The FastAPI application instance.
    """

    def __init__(self, app: FastAPI):
        """
        Universal exception handler middleware init function.
        Args:
            app (FastAPI): The FastAPI application instance.
        """

        super().__init__(app)

    @staticmethod
    def redacted_headers(request: Request) -> dict[str, str]:
        """
        Remove sensitive headers before logging request metadata.
        Args:
            request (Request): Request instance.
        Returns:
            dict: Headers safe to include in logs.
        """

        return {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in {"cookie", "set-cookie"}
            and not any(
                sensitive in name.lower()
                for sensitive in ("token", "secret", "password", "auth", "key")
            )
        }

    @staticmethod
    async def request_id(request: Request) -> Any:
        """
        Extract only the JSON-RPC request id from the request body.
        Args:
            request (Request): Request instance.
        Returns:
            Any: Request id if available, otherwise None.
        """

        try:
            body = await request.json()
        except Exception:
            return None
        return body.get("id") if isinstance(body, dict) else None

    async def dispatch(self, request: Request, call_next):
        """
        Dispatch function for sending errors to user from API
        Args:
            request (Request): The incoming request object.
            call_next: function to extract.
        """

        try:
            return await call_next(request)
        except AgentsBaseException as e:
            logger.warning(
                f"{e.__class__.__name__} [{e.status_code}]: {e.message} | input={e.error_input!r}"
            )
            return JSONResponse(
                status_code=e.status_code,
                content={"message": e.message, "input": e.error_input},
            )
        except A2AJsonRpcError as e:
            request_id = await self.request_id(request)
            return JSONResponse(
                status_code=200,
                content=self.a2a_error_response(e, request_id),
            )
        except Exception as e:
            logger.exception(
                "Unhandled exception: method={} path={} headers={}",
                request.method,
                request.url.path,
                self.redacted_headers(request),
            )
            return JSONResponse(
                status_code=500,
                content={
                    "message": "Internal server error",
                    "error_type": e.__class__.__name__,
                    "detail": str(e),
                },
            )

    @staticmethod
    def a2a_error_response(
        error: A2AJsonRpcError,
        request_id,
    ) -> dict:
        """
        Convert A2A errors to JSON-RPC error envelope.
        Args:
            error (A2AJsonRpcError): A2A error.
            request_id: JSON-RPC request id if available.
        Returns:
            dict: JSON-RPC error response.
        """

        error_body = {
            "code": error.code,
            "message": error.message,
        }
        if error.data is not None:
            error_body["data"] = error.data

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": error_body,
        }
