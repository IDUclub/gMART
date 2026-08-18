import os

from fastmcp.exceptions import AuthorizationError
from fastmcp.server.auth import AccessToken
from fastmcp.server.auth.providers.jwt import JWTVerifier

SERVICE_ACCOUNT_PREFIX = "service-account-"


class ServiceTokenVerifier(JWTVerifier):
    """Verify Keycloak JWTs and accept only client-credentials accounts."""

    def __init__(self) -> None:
        server_url = (os.getenv("SERVICE_AUTH_SERVER_URL") or "").strip().rstrip("/")
        realm = (os.getenv("SERVICE_AUTH_REALM") or "").strip()
        if not server_url or not realm:
            raise ValueError(
                "SERVICE_AUTH_SERVER_URL and SERVICE_AUTH_REALM are required"
            )
        issuer = f"{server_url}/realms/{realm}"
        super().__init__(
            jwks_uri=f"{issuer}/protocol/openid-connect/certs",
            issuer=issuer,
            algorithm="RS256",
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        access_token = await super().verify_token(token)
        if access_token is None:
            return None
        username = access_token.claims.get("preferred_username", "")
        if not isinstance(username, str) or not username.startswith(
            SERVICE_ACCOUNT_PREFIX
        ):
            raise AuthorizationError("A service-account token is required")

        return access_token
