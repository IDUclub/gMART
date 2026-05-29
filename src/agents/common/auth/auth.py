from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.agents.common.exceptions.base_exceptions import (
    AgentsInputException,
    AgentsUnauthorizedException,
)
from src.common.auth.exceptions import AuthError

http_bearer = HTTPBearer()


async def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
) -> str:
    """
    Extract and verify the Bearer token from the Authorization header.

    Returns:
        str: Raw Bearer token string (passed downstream to MCP clients).
    Raises:
        AgentsUnauthorizedException: Missing header, invalid/expired token.
        AgentsInputException: Credentials object present but token string is empty.
    """

    if not credentials:
        raise AgentsUnauthorizedException("Authorization header missing")

    token: str = credentials.credentials
    if not token:
        raise AgentsInputException("Token is missing in the authorization header")

    # Lazy import avoids circular dependency (dependencies.py imports this module).
    from src.agents.dependencies.dependencies import app_deps  # noqa: PLC0415

    auth_client = app_deps["auth_client"]
    try:
        await auth_client.process_token(token)
    except AuthError as exc:
        raise AgentsUnauthorizedException(str(exc)) from exc

    return token
