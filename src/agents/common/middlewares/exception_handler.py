"""Exception handling middleware is defined here."""

import traceback
from uuid import uuid4

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.agents.common.exceptions.a2a_exceptions import A2AJsonRpcError
from src.agents.common.exceptions.base_exceptions import AgentsBaseException


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
        except:
            try:
                request_info["body"] = str(await request.body())
                return request_info
            except:
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
            raise e.http_repr() from e
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
            "id": request_id if request_id is not None else str(uuid4()),
            "error": error_body,
        }
