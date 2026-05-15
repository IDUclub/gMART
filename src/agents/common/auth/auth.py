from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.agents.common.exceptions.base_exceptions import (
    AgentsInputException,
    AgentsUnauthorizedException,
)

http_bearer = HTTPBearer()


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
