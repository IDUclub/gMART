from fastapi import APIRouter, Depends

from src.agents.common.api_handlers.json_api_handler import JsonApiHandler
from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.common.exceptions.base_exceptions import AgentsNotFound
from src.agents.dependencies.dependencies import get_app_config
from src.agents.dto.auth_dto import LoginRequestDTO

auth_router = APIRouter(prefix="/auth", tags=["auth"])


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

    if not (app_config.AUTH_HELPER_URL and app_config.AUTH_HELPER_API_KEY):
        raise AgentsNotFound(
            "Auth helper is not configured — set AUTH_HELPER_URL and "
            "AUTH_HELPER_API_KEY to enable /auth/token"
        )
    handler = JsonApiHandler(app_config.AUTH_HELPER_URL)
    return await handler.post(
        "/api/token",
        headers={"X-Auth-Helper-Api-Key": app_config.AUTH_HELPER_API_KEY},
        data={
            "username": request.username,
            "password": request.password,
            "scope": "openid profile email",
        },
    )
