from __future__ import annotations

import time

from fastmcp.server.auth.providers.jwt import JWTVerifier

from src.agents.common.exceptions.base_exceptions import AgentsUnauthorizedException


class SynapseCallerVerifier:
    """Verify realm JWTs and distinguish the configured Synapse service client."""

    def __init__(
        self,
        *,
        auth_server_url: str,
        realm: str,
        service_client_id: str,
        audience: str | None = None,
    ) -> None:
        issuer = f"{auth_server_url.rstrip('/')}/realms/{realm}"
        self.service_client_id = service_client_id
        self.audience = audience
        self.verifier = JWTVerifier(
            jwks_uri=f"{issuer}/protocol/openid-connect/certs",
            issuer=issuer,
            # User tokens have different audiences. Share this JWKS cache with
            # Synapse and enforce its optional audience on verify_claims only.
            audience=None,
            algorithm="RS256",
        )

    async def verify_user(self, token: str) -> str:
        access_token = await self._verify(token)
        subject = access_token.claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise AgentsUnauthorizedException("JWT subject is missing")
        return subject

    async def verify_synapse_service(self, token: str) -> dict:
        claims = await self.verify_claims(token)
        client_id = claims.get("azp") or claims.get("client_id")
        if client_id != self.service_client_id:
            raise AgentsUnauthorizedException("Synapse service token is required")
        return dict(claims)

    async def verify_claims(self, token: str) -> dict:
        access_token = await self._verify(token)
        if self.audience:
            audience = access_token.claims.get("aud")
            audiences = audience if isinstance(audience, list) else [audience]
            if self.audience not in audiences:
                raise AgentsUnauthorizedException("Invalid access token audience")
        return dict(access_token.claims)

    async def _verify(self, token: str):
        try:
            access_token = await self.verifier.verify_token(token)
        except Exception as exc:
            raise AgentsUnauthorizedException("Invalid access token") from exc
        if access_token is None:
            raise AgentsUnauthorizedException("Invalid access token")
        expires = access_token.claims.get("exp")
        if (
            not isinstance(expires, (int, float))
            or isinstance(expires, bool)
            or not expires > time.time()
        ):
            raise AgentsUnauthorizedException("Invalid access token expiration")
        return access_token
