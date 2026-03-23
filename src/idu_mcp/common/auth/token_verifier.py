from fastmcp.exceptions import AuthorizationError
from fastmcp.server.auth import AccessToken, TokenVerifier


class AnyTokenVerifier(TokenVerifier):

    async def verify_token(self, token: str) -> AccessToken:
        if not token:
            raise AuthorizationError("Bearer token is required")

        return AccessToken(token=token, client_id="unknown", scopes=[], claims={})
