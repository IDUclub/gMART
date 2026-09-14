"""Local-only OIDC bridge for the Synapse integration smoke stand.

The production integration uses real confidential Keycloak clients. The local ICII
stand only has an auth-helper user credential, so this process exposes the tiny subset
of Keycloak endpoints used by ``idu-service-auth`` and FastMCP's ``JWTVerifier``:

* ``client_id=gmart`` proxies the existing auth helper and returns its real IDU token;
* every other allowed local client receives a short-lived RS256 token signed here.

The server refuses to start unless ``LOCAL_OIDC_ENABLED=true`` and is intended to be
bound to localhost / the private Docker network only.
"""

from __future__ import annotations

import base64
import os
import time
import uuid
from typing import Any

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Header, HTTPException, Request, status


def _required(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} is required by the local OIDC bridge")
    return value


if (os.getenv("LOCAL_OIDC_ENABLED") or "").lower() not in {"1", "true", "yes"}:
    raise RuntimeError("The local OIDC bridge requires LOCAL_OIDC_ENABLED=true")

REALM = (os.getenv("LOCAL_OIDC_REALM") or "local").strip()
ISSUER = (
    os.getenv("LOCAL_OIDC_ISSUER") or f"http://local_auth:8085/realms/{REALM}"
).rstrip("/")
SHARED_SECRET = _required("LOCAL_OIDC_SHARED_SECRET")
ALLOWED_CLIENTS = {
    item.strip()
    for item in (
        os.getenv("LOCAL_OIDC_ALLOWED_CLIENTS")
        or "gmart,gmart-internal,synapse,frontend"
    ).split(",")
    if item.strip()
}

_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_public_numbers = _private_key.public_key().public_numbers()
_kid = uuid.uuid4().hex

app = FastAPI(title="gMART local OIDC bridge", docs_url=None, redoc_url=None)


def _b64url(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _client_credentials(authorization: str | None, form: Any) -> tuple[str, str]:
    """Accept both RFC 6749 client_secret_basic and client_secret_post."""
    if authorization and authorization.lower().startswith("basic "):
        try:
            raw = base64.b64decode(authorization.split(" ", 1)[1]).decode()
            client_id, client_secret = raw.split(":", 1)
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(
                status_code=401, detail="Invalid client authentication"
            ) from exc
    else:
        client_id = str(form.get("client_id") or "")
        client_secret = str(form.get("client_secret") or "")
    if client_id not in ALLOWED_CLIENTS or client_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Invalid client authentication")
    return client_id, client_secret


async def _helper_token() -> dict[str, Any]:
    helper_url = _required("AUTH_HELPER_URL").rstrip("/")
    body = {
        "username": _required("AUTH_USERNAME"),
        "password": _required("AUTH_PASSWORD"),
        "scope": "openid profile email",
    }
    headers = {"X-Auth-Helper-Api-Key": _required("AUTH_HELPER_API_KEY")}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{helper_url}/api/token", json=body, headers=headers
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=502, detail="The configured auth helper rejected login"
        )
    payload = response.json()
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise HTTPException(
            status_code=502, detail="The auth helper returned no access token"
        )
    return {
        "access_token": token,
        "expires_in": int(payload.get("expires_in") or 300),
        "token_type": "Bearer",
        "scope": payload.get("scope") or "openid profile email",
    }


def _local_token(client_id: str) -> dict[str, Any]:
    now = int(time.time())
    service_client = client_id != "frontend"
    claims = {
        "iss": ISSUER,
        "sub": f"service-account-{client_id}" if service_client else "local-smoke-user",
        "azp": client_id,
        "client_id": client_id,
        "preferred_username": (
            f"service-account-{client_id}" if service_client else "local-smoke-user"
        ),
        "iat": now,
        "exp": now + 300,
        "typ": "Bearer",
    }
    token = jwt.encode(claims, _private_key, algorithm="RS256", headers={"kid": _kid})
    return {
        "access_token": token,
        "expires_in": 300,
        "token_type": "Bearer",
        "scope": "openid profile email",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/realms/{realm}/protocol/openid-connect/certs")
async def certs(realm: str) -> dict[str, list[dict[str, str]]]:
    if realm != REALM:
        raise HTTPException(status_code=404, detail="Unknown realm")
    return {
        "keys": [
            {
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "kid": _kid,
                "n": _b64url(_public_numbers.n),
                "e": _b64url(_public_numbers.e),
            }
        ]
    }


@app.post("/realms/{realm}/protocol/openid-connect/token")
async def token(
    realm: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    if realm != REALM:
        raise HTTPException(status_code=404, detail="Unknown realm")
    form = await request.form()
    client_id, _ = _client_credentials(authorization, form)
    if form.get("grant_type") != "client_credentials":
        raise HTTPException(
            status_code=400, detail="Only client_credentials is supported"
        )
    return await _helper_token() if client_id == "gmart" else _local_token(client_id)
