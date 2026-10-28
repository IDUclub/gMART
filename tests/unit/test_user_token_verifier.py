"""Realm JWT verification and the boundary for in-process user contexts."""

from unittest.mock import AsyncMock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from fastmcp.server.auth.providers.jwt import RSAKeyPair

from src.agents.common.auth.auth import optional_bearer_token, verify_bearer_token
from src.agents.common.auth.synapse_auth import SynapseCallerVerifier
from src.agents.common.exceptions.base_exceptions import AgentsUnauthorizedException
from src.agents.common.middlewares.exception_handler import ExceptionHandlerMiddleware
from src.common.service_auth import internal_user_context_jwt

ISSUER = "https://auth.example/realms/test"


@pytest.fixture
def verifier():
    return SynapseCallerVerifier(
        auth_server_url="https://auth.example/",
        realm="test",
        service_client_id="synapse",
        audience="synapse-audience",
    )


@pytest.fixture(scope="module")
def keys():
    return RSAKeyPair.generate()


@pytest.mark.parametrize("audience", [None, "account", ["other", "account"]])
async def test_user_accepts_any_audience_and_uses_realm_jwks(
    verifier, keys, audience, monkeypatch
):
    assert verifier.verifier.jwks_uri == f"{ISSUER}/protocol/openid-connect/certs"
    assert verifier.verifier.issuer == ISSUER
    assert verifier.verifier.algorithm == "RS256"
    assert verifier.verifier.audience is None
    monkeypatch.setattr(
        verifier.verifier,
        "_get_verification_key",
        AsyncMock(return_value=keys.public_key),
    )
    token = keys.create_token(subject="alice", issuer=ISSUER, audience=audience)
    assert await verifier.verify_user(token) == "alice"


@pytest.mark.parametrize(
    "invalid", ["signature", "unsigned", "issuer", "expired", "no-exp", "no-sub"]
)
async def test_invalid_user_token_rejected(verifier, keys, invalid, monkeypatch):
    monkeypatch.setattr(
        verifier.verifier,
        "_get_verification_key",
        AsyncMock(return_value=keys.public_key),
    )
    signer = RSAKeyPair.generate() if invalid == "signature" else keys
    token = signer.create_token(
        subject="alice",
        issuer="https://foreign" if invalid == "issuer" else ISSUER,
        expires_in_seconds=-30 if invalid == "expired" else 3600,
        additional_claims=(
            {"exp": None}
            if invalid == "no-exp"
            else {"sub": ""} if invalid == "no-sub" else None
        ),
    )
    if invalid == "unsigned":
        token = internal_user_context_jwt("alice")
    with pytest.raises(AgentsUnauthorizedException):
        await verifier.verify_user(token)


@pytest.mark.parametrize(
    "audience,allowed",
    [
        ("synapse-audience", True),
        (["account", "synapse-audience"], True),
        ("account", False),
        (None, False),
    ],
)
async def test_synapse_audience_still_enforced(
    verifier, keys, audience, allowed, monkeypatch
):
    monkeypatch.setattr(
        verifier.verifier,
        "_get_verification_key",
        AsyncMock(return_value=keys.public_key),
    )
    token = keys.create_token(
        subject="svc",
        issuer=ISSUER,
        audience=audience,
        additional_claims={"azp": "synapse"},
    )
    if allowed:
        assert (await verifier.verify_synapse_service(token))["azp"] == "synapse"
    else:
        with pytest.raises(AgentsUnauthorizedException):
            await verifier.verify_claims(token)


@pytest.mark.parametrize("dependency", [verify_bearer_token, optional_bearer_token])
def test_http_edge_rejects_alg_none(dependency):
    app = FastAPI()
    app.add_middleware(ExceptionHandlerMiddleware)

    @app.get("/edge")
    async def edge(token=Depends(dependency)):
        return {"token": token}

    with TestClient(app) as client:
        token = internal_user_context_jwt("victim")
        assert (
            client.get(
                "/edge", headers={"Authorization": f"Bearer {token}"}
            ).status_code
            == 401
        )
        assert (
            client.get(
                "/edge", headers={"Authorization": "Bearer external-token"}
            ).status_code
            == 200
        )
        if dependency is optional_bearer_token:
            assert client.get("/edge").json() == {"token": None}


@pytest.mark.parametrize("enabled", [False, True])
def test_user_verifier_always_built_and_shared(enabled, monkeypatch):
    from src.agents.dependencies import init_dependencies as module

    config = module.load_config()
    config.SYNAPSE_ENABLED = enabled
    config.SYNAPSE_API_URL = "http://synapse"
    monkeypatch.setattr(module, "load_config", lambda: config)
    monkeypatch.setattr(module, "config_logger", lambda: "logs")
    deps = module.init_dependencies()
    assert isinstance(deps["synapse_caller_verifier"], SynapseCallerVerifier)
    assert (
        deps["project_files_service"].user_token_verifier
        is deps["synapse_caller_verifier"]
    )
