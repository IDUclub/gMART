"""Read-only contract smoke tests for every deployed Urban MCP group."""

from __future__ import annotations

import os

import pytest

from src.agents.mcp_clients.urban_mcp_client import (
    URBAN_MCP_GROUPS,
    UrbanMcpClient,
)

pytestmark = pytest.mark.integration


async def test_all_six_urban_mcp_groups_list_read_only_tools():
    base_url = os.environ.get("URBAN_MCP_SERVER")
    if not base_url:
        pytest.skip("URBAN_MCP_SERVER is not set")
    client = UrbanMcpClient(base_url, os.environ.get("URBAN_API_TOKEN", ""))

    try:
        tools = await client.load_tools()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Urban MCP unavailable at {base_url}: {exc}")

    assert {tool.group for tool in tools} == set(URBAN_MCP_GROUPS)
    assert all(client.endpoint_url(group).endswith("/") for group in URBAN_MCP_GROUPS)
    assert any(tool.name == "GetScenarioAllObjectsWithoutGeometry" for tool in tools)
    assert not any(tool.name == "CreateProject" for tool in tools)
