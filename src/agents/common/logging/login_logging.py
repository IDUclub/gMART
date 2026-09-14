"""Server-side login audit, without request bodies or exception payloads."""

import asyncio
import json
import os
import time
import uuid

from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from loguru import logger
from starlette.exceptions import HTTPException

from src.agents.common.exceptions.api_exceptions import DownstreamServiceError
from src.agents.common.exceptions.base_exceptions import (
    AgentsBaseException,
    AgentsNotFound,
    AgentsUnauthorizedException,
)


class LoginAuditRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()
        if self.path != "/auth/token" or "POST" not in self.methods:
            return handler

        async def audited(request):
            record = dict(
                attempt_id=str(uuid.uuid4()),
                event="login_attempt",
                outcome="started",
                username="-",
                status="-",
                reason="-",
                duration_ms=0,
                error_type="-",
                downstream_status="-",
            )
            started = time.monotonic()
            logger.bind(login=dict(record)).info("Login " + json.dumps(record))
            level = "INFO"
            try:
                response = await handler(request)
                record.update(
                    status=response.status_code,
                    outcome="success" if response.status_code < 400 else "failure",
                )
                if response.status_code >= 400:
                    level = "ERROR" if response.status_code >= 500 else "WARNING"
                    record["reason"] = "http_error"
                return response
            except asyncio.CancelledError:
                level = "WARNING"
                record.update(
                    outcome="cancelled",
                    reason="request_cancelled",
                    error_type="CancelledError",
                )
                raise
            except Exception as exc:
                status = 500
                reason = "internal_error"
                if isinstance(exc, RequestValidationError):
                    status, reason = 422, "invalid_request"
                elif isinstance(exc, AgentsUnauthorizedException):
                    status, reason = exc.status_code, "invalid_credentials"
                elif isinstance(exc, AgentsNotFound):
                    status, reason = exc.status_code, "login_not_configured"
                elif isinstance(exc, DownstreamServiceError):
                    status, reason = exc.status_code, "auth_helper_unavailable"
                    record["downstream_status"] = exc.downstream_status
                elif isinstance(exc, (AgentsBaseException, HTTPException)):
                    status, reason = exc.status_code, "http_error"
                reason = getattr(request.state, "login_failure_reason", reason)
                record.update(
                    outcome="failure",
                    status=status,
                    reason=reason,
                    error_type=type(exc).__name__,
                )
                level = "ERROR" if status >= 500 else "WARNING"
                if reason == "internal_error":
                    # The generic middleware returns request details for unknown
                    # exceptions; credentials must never reach that response path.
                    raise AgentsBaseException(
                        "Внутренняя ошибка входа. Повторите попытку позже."
                    ) from None
                raise
            finally:
                record.update(
                    event="login_result",
                    duration_ms=round((time.monotonic() - started) * 1000, 2),
                )
                if os.getenv("AUTH_LOG_USERNAME", "false").lower() in {
                    "true",
                    "1",
                    "yes",
                    "on",
                }:
                    record["username"] = getattr(request.state, "login_username", "-")[
                        :128
                    ]
                # JSON escaping prevents usernames from injecting new log lines.
                logger.bind(login=dict(record)).log(
                    level, "Login " + json.dumps(record, ensure_ascii=True)
                )

        return audited
