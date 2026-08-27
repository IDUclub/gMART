from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.agents.common.exceptions.base_exceptions import (
    AgentsInputException,
    AgentsUnauthorizedException,
)

http_bearer = HTTPBearer()
optional_http_bearer = HTTPBearer(auto_error=False)


async def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
) -> str:
    """
    Retrieve the Bearer token from the Authorization header.
    Args:
        credentials (HTTPAuthorizationCredentials): Request credentials.
    Returns:
        str: Extracted Bearer token.
    Raises:
        AgentsUnauthorizedException: If no credentials are provided (401).
        AgentsInputException: If the token is missing from credentials (400).
    """

    if not credentials:
        raise AgentsUnauthorizedException("Authorization header missing")

    token: str = credentials.credentials

    if not token:
        raise AgentsInputException("Token is missing in the authorization header")

    return token


async def optional_bearer_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_http_bearer),
) -> str | None:
    """
    Retrieve the Bearer token when the caller sent one, without requiring it.

    Used by public endpoints that also serve anonymous requests and switch to the
    authorized behaviour (chat history, scenario-scoped data) once a token is present.
    Args:
        credentials (HTTPAuthorizationCredentials | None): Request credentials, if any.
    Returns:
        str | None: Extracted Bearer token, or None for an anonymous request.
    Raises:
        AgentsInputException: If the Authorization header carries an empty token.
    """

    if not credentials:
        return None

    token: str = credentials.credentials

    if not token:
        raise AgentsInputException("Token is missing in the authorization header")

    return token
