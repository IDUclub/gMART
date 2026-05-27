"""Exception handling middleware is defined here."""

import traceback

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
    async def prepare_request_info(request: Request) -> dict:
        """
        Function prepares request input data
        Args:
            request (Request): Request instance.
        Returns:
            dict: Request input data.
        """

        request_info = {
            "method": request.method,
            "url": str(request.url),
            "path_params": dict(request.path_params),
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
        }

        try:
            request_info["body"] = await request.json()
            return request_info
        except Exception as e:
            logger.error("Faced error during request_info parsing")
            logger.exception(e)
            try:
                request_info["body"] = str(await request.body())
                return request_info
            except Exception as e:
                logger.error("Couldn't pars request body as str")
                logger.exception(e)
                request_info["body"] = "Could not read request body"
                return request_info

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
            request_info = await self.prepare_request_info(request)
            return JSONResponse(
                status_code=200,
                content=self.a2a_error_response(e, request_info.get("body")),
            )
        except Exception as e:
            request_info = await self.prepare_request_info(request)
            return JSONResponse(
                status_code=500,
                content={
                    "message": "Internal server error",
                    "error_type": e.__class__.__name__,
                    "request": request_info,
                    "detail": str(e),
                    "traceback": traceback.format_exc().splitlines(),
                },
            )

    @staticmethod
    def a2a_error_response(
        error: A2AJsonRpcError,
        request_body,
    ) -> dict:
        """
        Convert A2A errors to JSON-RPC error envelope.
        Args:
            error (A2AJsonRpcError): A2A error.
            request_body: Parsed request body if available.
        Returns:
            dict: JSON-RPC error response.
        """

        request_id = None
        if isinstance(request_body, dict):
            request_id = request_body.get("id")

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
