from fastmcp.exceptions import AuthorizationError
from fastmcp.server.auth import AccessToken, TokenVerifier

from src.common.auth.auth_client import AuthenticationClient
from src.common.auth.exceptions import AuthError


class KeycloakTokenVerifier(TokenVerifier):
    """FastMCP TokenVerifier backed by Keycloak JWT validation."""

    def __init__(self, auth_client: AuthenticationClient):
        self._auth_client = auth_client

    async def verify_token(self, token: str) -> AccessToken:
        if not token:
            raise AuthorizationError("Bearer token is required")
        try:
            payload = await self._auth_client.process_token(token)
        except AuthError as exc:
            raise AuthorizationError(str(exc)) from exc

        client_id = payload.get("sub", "unknown")
        return AccessToken(token=token, client_id=client_id, scopes=[], claims=payload)
