from __future__ import annotations

import base64
import json
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastmcp import Client
from idu_service_auth import KeycloakTokenClient, KeycloakTokenConfig

USER_ID_HEADER = "X-User-Id"
# IDU_DVD demands X-User-Id on every search tool, even for the shared index where it
# discards the value (it is only honoured together with project_id/scenario_id). Public,
# unauthenticated document-QA questions carry this placeholder instead of a Keycloak sub.
ANONYMOUS_USER_ID = "anonymous"


def build_service_auth(env_prefix: str = "") -> KeycloakTokenClient:
    """Build a service-token client from mandatory, optionally prefixed settings."""

    values = {
        name: (os.getenv(f"{env_prefix}{name}") or "").strip()
        for name in (
            "SERVICE_AUTH_SERVER_URL",
            "SERVICE_AUTH_REALM",
            "SERVICE_AUTH_CLIENT_ID",
            "SERVICE_AUTH_CLIENT_SECRET",
        )
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(
            "Missing mandatory service-auth variables: "
            + ", ".join(f"{env_prefix}{name}" for name in missing)
        )
    return KeycloakTokenClient(
        KeycloakTokenConfig(
            auth_server_url=values["SERVICE_AUTH_SERVER_URL"],
            realm=values["SERVICE_AUTH_REALM"],
            client_id=values["SERVICE_AUTH_CLIENT_ID"],
            client_secret=values["SERVICE_AUTH_CLIENT_SECRET"],
            background_refresh=True,
        )
    )


def build_optional_service_auth(env_prefix: str) -> KeycloakTokenClient | None:
    """Build a prefixed auth client when configured, otherwise return ``None``.

    A partially configured client is rejected instead of silently falling back to the
    default identity. This makes service-boundary mistakes visible during startup.
    """

    names = (
        "SERVICE_AUTH_SERVER_URL",
        "SERVICE_AUTH_REALM",
        "SERVICE_AUTH_CLIENT_ID",
        "SERVICE_AUTH_CLIENT_SECRET",
    )
    if not any((os.getenv(f"{env_prefix}{name}") or "").strip() for name in names):
        return None
    return build_service_auth(env_prefix)


@asynccontextmanager
async def service_auth_lifespan(
    auth: KeycloakTokenClient,
) -> AsyncIterator[KeycloakTokenClient]:
    """Start one auth client, fail fast, and close it with the application."""

    async with auth:
        await auth.get_access_token()
        yield auth


def user_id_from_jwt(token: str) -> str:
    """Read the stable user subject from the already accepted front-door JWT."""

    try:
        payload_segment = token.split(".", 2)[1]
        payload = json.loads(
            base64.urlsafe_b64decode(
                payload_segment + "=" * (-len(payload_segment) % 4)
            )
        )
        user_id = payload.get("sub")
    except (IndexError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Bearer token does not contain a readable user id") from exc
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("Bearer token does not contain a user id")
    return user_id


def internal_user_context_jwt(user_id: str) -> str:
    """Build a non-authorizing JWT-shaped value used only for in-process user context.

    Downstream transports replace it with the process service token; its only purpose is
    to preserve the already verified ``sub`` through legacy pipeline signatures.
    """

    header = (
        base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
    )
    payload = (
        base64.urlsafe_b64encode(
            json.dumps({"sub": user_id}, separators=(",", ":")).encode()
        )
        .decode()
        .rstrip("=")
    )
    return f"{header}.{payload}."


async def service_headers(
    auth: KeycloakTokenClient, user_id: str | None = None
) -> dict[str, str]:
    headers = dict(await auth.get_authorization_headers())
    if user_id:
        headers[USER_ID_HEADER] = user_id
    return headers


class ServiceTokenAuth(httpx.Auth):
    """Attach a freshly cached/refreshed service token to every HTTP request."""

    def __init__(self, auth: KeycloakTokenClient, user_id: str | None = None) -> None:
        self.auth = auth
        self.user_id = user_id

    async def async_auth_flow(self, request: httpx.Request):
        request.headers.update(await service_headers(self.auth, self.user_id))
        yield request


async def service_mcp_client(
    url: str, auth: KeycloakTokenClient, user_id: str
) -> Client:
    return Client(url, auth=ServiceTokenAuth(auth, user_id))
