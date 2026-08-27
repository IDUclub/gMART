from __future__ import annotations

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
        self.verifier = JWTVerifier(
            jwks_uri=f"{issuer}/protocol/openid-connect/certs",
            issuer=issuer,
            audience=audience,
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
        return dict(access_token.claims)

    async def _verify(self, token: str):
        try:
            access_token = await self.verifier.verify_token(token)
        except Exception as exc:
            raise AgentsUnauthorizedException("Invalid access token") from exc
        if access_token is None:
            raise AgentsUnauthorizedException("Invalid access token")
        return access_token
