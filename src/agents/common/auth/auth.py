from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

http_bearer = HTTPBearer()


async def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(http_bearer)) -> str:
    """
    Function retrieves Bearer token from headers.
    Args:
        credentials (HTTPAuthorizationCredentials): Request credentials.
    Returns:
        str: Extracted Bearer token.
    Raises:
         HTTPException:
            - 401 if no credentials provided.
            - 400 if no token in credentials provided.
    """

    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing",
        )

    token = credentials.credentials

    if not token:
        raise HTTPException(
            status_code=400, detail="Token is missing in the authorization header"
        )

    return token
