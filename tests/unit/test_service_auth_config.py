from unittest.mock import MagicMock

import pytest

from src.common import service_auth

AUTH_VALUES = {
    "SERVICE_AUTH_SERVER_URL": "https://auth.example",
    "SERVICE_AUTH_REALM": "main",
    "SERVICE_AUTH_CLIENT_ID": "gmart",
    "SERVICE_AUTH_CLIENT_SECRET": "secret",
}


def test_optional_service_auth_is_absent_without_prefixed_settings(monkeypatch):
    for name in AUTH_VALUES:
        monkeypatch.delenv(f"IDU_MCP_{name}", raising=False)

    assert service_auth.build_optional_service_auth("IDU_MCP_") is None


def test_optional_service_auth_uses_prefixed_settings(monkeypatch):
    for name, value in AUTH_VALUES.items():
        monkeypatch.setenv(f"IDU_MCP_{name}", value)
    client = MagicMock()
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr(service_auth, "KeycloakTokenClient", constructor)

    assert service_auth.build_optional_service_auth("IDU_MCP_") is client
    config = constructor.call_args.args[0]
    assert config.auth_server_url == "https://auth.example"
    assert config.realm == "main"
    assert config.client_id == "gmart"
    assert config.client_secret == "secret"


def test_partial_optional_service_auth_is_rejected(monkeypatch):
    for name in AUTH_VALUES:
        monkeypatch.delenv(f"IDU_MCP_{name}", raising=False)
    monkeypatch.setenv("IDU_MCP_SERVICE_AUTH_CLIENT_ID", "gmart")

    with pytest.raises(ValueError, match="IDU_MCP_SERVICE_AUTH_SERVER_URL"):
        service_auth.build_optional_service_auth("IDU_MCP_")
