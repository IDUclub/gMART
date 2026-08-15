from types import SimpleNamespace

import pytest

from src.agents.mcp_clients.urban_mcp_client import (
    URBAN_MCP_GROUPS,
    UrbanMcpClient,
)


class FakeTransport:
    def __init__(self, tools, result=None):
        self.tools = tools
        self.result = result
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def list_tools(self):
        return self.tools

    async def call_tool(self, name, arguments, meta):
        self.calls.append((name, arguments, meta))
        return SimpleNamespace(data=self.result)


def tool(name, *, read_only=True, properties=None):
    return SimpleNamespace(
        name=name,
        title=name,
        description=f"Description for {name}",
        inputSchema={"type": "object", "properties": properties or {}},
        annotations={"readOnlyHint": read_only},
        meta={"fastmcp": {"tags": ["data"]}},
    )


@pytest.mark.asyncio
async def test_loads_all_groups_filters_mutations_and_routes_calls():
    clients = {
        group: FakeTransport([tool(f"Get_{group}")], result={"group": group})
        for group in URBAN_MCP_GROUPS
    }
    clients["projects"].tools.append(tool("CreateProject", read_only=False))
    client = UrbanMcpClient("https://urban.example/", "token", clients=clients)

    loaded = await client.load_tools()

    assert {item.group for item in loaded} == set(URBAN_MCP_GROUPS)
    assert "CreateProject" not in {item.name for item in loaded}
    assert client.endpoint_url("projects") == "https://urban.example/mcp/projects/"

    result = await client.execute_tool(
        "indicators", "Get_indicators", {"scenario_id": 42}, meta={"scenario_id": 42}
    )
    assert result == {"group": "indicators"}
    assert clients["indicators"].calls == [
        ("Get_indicators", {"scenario_id": 42}, {"scenario_id": 42})
    ]


@pytest.mark.asyncio
async def test_duplicate_tool_names_across_groups_fail_clearly():
    clients = {
        group: FakeTransport(
            [
                tool(
                    "GetShared"
                    if group in {"projects", "territories"}
                    else f"Get_{group}"
                )
            ]
        )
        for group in URBAN_MCP_GROUPS
    }
    client = UrbanMcpClient("https://urban.example", "token", clients=clients)

    with pytest.raises(ValueError, match="Duplicate Urban MCP tool name"):
        await client.load_tools()


def test_update_token_recreates_every_authenticated_transport():
    created = []

    def factory(url, auth):
        created.append((url, auth))
        return FakeTransport([])

    client = UrbanMcpClient("https://urban.example", "old", client_factory=factory)
    client.update_token("new")

    assert len(created) == len(URBAN_MCP_GROUPS) * 2
    assert {auth for _, auth in created[-len(URBAN_MCP_GROUPS) :]} == {"new"}
    assert all(url.endswith("/") for url, _ in created)
