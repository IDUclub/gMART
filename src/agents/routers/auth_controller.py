import json

from fastapi import APIRouter, Depends, Request

from src.agents.common.api_handlers.json_api_handler import JsonApiHandler
from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.common.exceptions.api_exceptions import DownstreamServiceError
from src.agents.common.exceptions.base_exceptions import (
    AgentsBaseException,
    AgentsInputException,
    AgentsNotFound,
    AgentsUnauthorizedException,
)
from src.agents.common.logging.login_logging import LoginAuditRoute
from src.agents.dependencies.dependencies import get_app_config
from src.agents.dto.auth_dto import LoginRequestDTO

auth_router = APIRouter(prefix="/auth", tags=["auth"], route_class=LoginAuditRoute)


@auth_router.get(
    "/available",
    summary="Whether the auth helper login proxy is configured",
    response_description="Availability flag for the /auth/token endpoint",
)
async def auth_available(
    app_config: AgentsAppConfig = Depends(get_app_config),
) -> dict:
    """
    Report whether the /auth/token proxy is enabled on this deployment.

    The UI uses this flag to decide between the in-app login form and the
    legacy redirect to the auth helper page.
    """

    return {
        "enabled": bool(app_config.AUTH_HELPER_URL and app_config.AUTH_HELPER_API_KEY)
    }


@auth_router.post(
    "/token",
    summary="Obtain an access token via the IDU auth helper",
    response_description="Token response of the auth helper (access_token, expires_in, ...)",
)
async def issue_token(
    request: LoginRequestDTO,
    http_request: Request,
    app_config: AgentsAppConfig = Depends(get_app_config),
) -> dict:
    """
    Proxy the credentials to the IDU auth helper ``POST /api/token`` endpoint.

    The helper's API key is attached server-side (``AUTH_HELPER_API_KEY`` env var),
    so it never reaches the browser. The helper response is returned as-is —
    ``access_token``, ``expires_in``, ``token_type``, ``scope``.

    Returns 404 when the deployment has no auth helper configured
    (``AUTH_HELPER_URL`` / ``AUTH_HELPER_API_KEY`` are unset).
    """

    http_request.state.login_username = request.username
    if not (app_config.AUTH_HELPER_URL and app_config.AUTH_HELPER_API_KEY):
        raise AgentsNotFound(
            "Auth helper is not configured — set AUTH_HELPER_URL and "
            "AUTH_HELPER_API_KEY to enable /auth/token"
        )
    handler = JsonApiHandler(app_config.AUTH_HELPER_URL)
    try:
        result = await handler.post(
            "/api/token",
            headers={"X-Auth-Helper-Api-Key": app_config.AUTH_HELPER_API_KEY},
            data={
                "username": request.username,
                "password": request.password,
                "scope": "openid profile email",
            },
        )
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("access_token"), str)
            or not result["access_token"].strip()
        ):
            http_request.state.login_failure_reason = "invalid_helper_response"
            raise DownstreamServiceError(
                "auth-helper",
                200,
                "Сервис входа вернул некорректный ответ. Повторите попытку позже.",
            )
        return result
    except AgentsBaseException as exc:
        payload = exc.error_input
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (ValueError, TypeError):
                payload = None
        detail = payload.get("detail", payload) if isinstance(payload, dict) else None
        if (
            isinstance(exc, AgentsInputException)
            and isinstance(detail, dict)
            and detail.get("error") == "invalid_grant"
        ):
            raise AgentsUnauthorizedException(
                "Неверный логин или пароль. Проверьте данные для входа."
            ) from None
        # A helper key/realm/configuration failure must not be blamed on the user.
        raise DownstreamServiceError(
            "auth-helper",
            (
                exc.downstream_status
                if isinstance(exc, DownstreamServiceError)
                else exc.status_code
            ),
            "Сервис входа временно недоступен. Повторите попытку позже.",
        ) from None
