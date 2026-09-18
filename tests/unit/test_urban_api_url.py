from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.common.api_handlers.json_api_handler import JsonApiHandler
from src.agents.common.config.app_config import AgentsAppConfig
from src.common.urban_api_url import normalize_urban_api_url
from src.idu_mcp.common.api_handlers.json_api_handler import (
    JsonApiHandler as McpHandler,
)
from src.idu_mcp.common.config.mcp_config import IduFastMcpConfig


@pytest.mark.parametrize(
    "base, api_root",
    [
        ("https://urban.test:8443", "https://urban.test:8443/api"),
        ("https://urban.test:8443/", "https://urban.test:8443/api"),
        ("https://urban.test:8443/api", "https://urban.test:8443/api"),
        (" https://urban.test:8443/api/// ", "https://urban.test:8443/api"),
        (
            "https://prostor-api.idu.actocgnitive.org/urban_api",
            "https://prostor-api.idu.actocgnitive.org/urban_api",
        ),
        (
            "https://prostor-api.idu.actocgnitive.org/urban_api/",
            "https://prostor-api.idu.actocgnitive.org/urban_api",
        ),
        (
            "https://urban.test/gateway/urban_api/",
            "https://urban.test/gateway/urban_api",
        ),
    ],
)
@pytest.mark.parametrize("side", ["agents", "mcp"])
async def test_urban_requests_use_configured_api_root(base, api_root, side):
    if side == "agents":
        config = AgentsAppConfig(
            ollama_api_url="http://localhost:11434",
            idu_mcp_url="http://idu/mcp",
            effects_mcp_url="http://effects/mcp",
            chat_storage_url="http://chat",
            urban_api_url=base,
        )
        handler = JsonApiHandler(config.URBAN_API_URL)
        assert config.CHAT_STORAGE_URL == "http://chat"
    else:
        config = IduFastMcpConfig(urban_api_url=base)
        handler = McpHandler(config.URBAN_API_URL)
    session = MagicMock()
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value={"project": {"project_id": 42}})
    session.get.return_value.__aenter__.return_value = response
    await handler.get("/v1/scenarios/7", session=session)
    assert session.get.call_args.kwargs["url"] == f"{api_root}/v1/scenarios/7"


async def test_generic_handler_preserves_non_urban_service_urls():
    session = MagicMock()
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value={})
    session.get.return_value.__aenter__.return_value = response
    await JsonApiHandler("https://auth.test").get("/token", session=session)
    assert session.get.call_args.kwargs["url"] == "https://auth.test/token"


@pytest.mark.parametrize(
    "base, expected",
    [
        ("http://api", "http://api/api"),
        ("https://urban.test/gateway/api/", "https://urban.test/gateway/api"),
        ("https://urban.test/gateway/", "https://urban.test/gateway"),
        ("https://urban.test/api/api/", "https://urban.test/api/api"),
    ],
)
def test_normalization_preserves_authority_and_proxy_path(base, expected):
    assert normalize_urban_api_url(base) == expected
    assert normalize_urban_api_url(expected) == expected


@pytest.mark.parametrize(
    "base",
    [
        "",
        "urban.test",
        "ftp://urban.test",
        "https://urban.test?x=1",
        "https://urban.test#fragment",
    ],
)
def test_invalid_base_url_is_rejected(base):
    with pytest.raises(ValueError, match="Urban API URL"):
        normalize_urban_api_url(base)
