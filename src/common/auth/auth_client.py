"""Keycloak JWT verification client."""

import asyncio
from typing import Any

import aiohttp
from aiohttp import ClientConnectorError
from cachetools import TTLCache
from jose import JWTError, jwt
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from src.common.auth.auth_config import AuthConfig
from src.common.auth.exceptions import (
    AuthDecodeError,
    InvalidAudienceError,
    InvalidTokenSignatureError,
    TokenExpiredError,
)

_JWKS_CACHE_KEY = "jwks"


class AuthenticationClient:
    """Validates Keycloak JWT tokens and caches results."""

    _RETRIES = 3

    def __init__(self, config: AuthConfig):
        self.config = config
        self._jwks_cache: TTLCache = TTLCache(maxsize=1, ttl=config.jwks_cache_ttl)
        self._user_cache: TTLCache = TTLCache(
            maxsize=config.user_cache_size,
            ttl=config.user_cache_ttl,
        )
        self._lock = asyncio.Lock()

    def update(self, config: AuthConfig) -> None:
        """Hot-reload configuration."""
        self.config = config
        self._jwks_cache = TTLCache(maxsize=1, ttl=config.jwks_cache_ttl)
        self._user_cache = TTLCache(
            maxsize=config.user_cache_size,
            ttl=config.user_cache_ttl,
        )

    @retry(
        stop=stop_after_attempt(_RETRIES),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(ClientConnectorError),
    )
    async def _fetch_jwks(self) -> dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.config.jwks_url,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def _get_jwks(self) -> dict[str, Any]:
        if _JWKS_CACHE_KEY in self._jwks_cache:
            return self._jwks_cache[_JWKS_CACHE_KEY]

        async with self._lock:
            if _JWKS_CACHE_KEY in self._jwks_cache:
                return self._jwks_cache[_JWKS_CACHE_KEY]
            jwks = await self._fetch_jwks()
            self._jwks_cache[_JWKS_CACHE_KEY] = jwks
            return jwks

    async def _verify_jwt(self, token: str) -> dict[str, Any]:
        try:
            jwks = await self._get_jwks()

            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            if not kid:
                raise InvalidTokenSignatureError("Token header missing 'kid'")

            key = next(
                (k for k in jwks.get("keys", []) if k.get("kid") == kid),
                None,
            )
            if not key:
                raise InvalidTokenSignatureError("No matching public key found")

            payload = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                issuer=self.config.server_url,
                options={"verify_aud": False},
            )

            if self.config.verify_aud:
                audiences = payload.get("aud", [])
                if isinstance(audiences, str):
                    audiences = [audiences]
                if not any(aud in self.config.valid_audiences for aud in audiences):
                    raise InvalidAudienceError("Token audience does not match")

            return payload

        except (TokenExpiredError, InvalidTokenSignatureError, InvalidAudienceError):
            raise
        except JWTError as exc:
            if "expired" in str(exc).lower():
                raise TokenExpiredError("Token has expired") from exc
            raise InvalidTokenSignatureError(str(exc)) from exc
        except Exception as exc:
            logger.exception(exc)
            raise AuthDecodeError("Failed to decode token") from exc

    async def process_token(self, token: str) -> dict[str, Any]:
        """Verify token (if configured) and return its claims."""
        if self.config.verify:
            return await self._verify_jwt(token)
        try:
            return jwt.get_unverified_claims(token)
        except Exception as exc:
            raise AuthDecodeError("Failed to decode token claims") from exc

    async def get_user_id(self, token: str) -> str | None:
        """Validate token and return the subject claim (user id)."""
        cached = self._user_cache.get(token)
        if cached:
            return cached
        payload = await self.process_token(token)
        user_id = payload.get("sub")
        self._user_cache[token] = user_id
        return user_id
