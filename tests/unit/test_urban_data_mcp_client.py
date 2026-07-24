"""Unit tests for UrbanDataMcpClient — the multi-group aggregator.

Live-verified against the real urban-mcp server (session notes): 6 groups
(dictionaries, territories, indicators, physical_objects, projects, soc_groups — the
last served at the hyphenated ``/mcp/soc-groups`` path), only ``projects`` authenticated,
``CreateProject`` is the sole mutating tool across all of them. These tests fake the
underlying ``fastmcp.Client`` (one per group) to exercise the aggregation/dispatch logic
without a network call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.mcp_clients.urban_data_mcp_client import UrbanDataMcpClient

BASE_URL = "https://urban-mcp.example.ru/mcp"


def _tool(name: str, description: str = "", input_schema: dict | None = None):
    return SimpleNamespace(
        name=name, description=description, inputSchema=input_schema or {}
    )


class FakeGroupClient:
    """Stand-in for one group's ``fastmcp.Client``: records construction args and calls."""

    def __init__(self, url: str, auth=None) -> None:
        self.url = url
        self.auth = auth
        self.tools: list = []
        self.tool_results: dict = {}
        self.raise_on_list: Exception | None = None
        self.call_tool_calls: list[dict] = []

    async def __aenter__(self) -> "FakeGroupClient":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def list_tools(self):
        if self.raise_on_list is not None:
            raise self.raise_on_list
        return self.tools

    async def call_tool(self, name: str, arguments: dict, meta: dict | None = None):
        self.call_tool_calls.append(
            {"name": name, "arguments": arguments, "meta": meta}
        )
        return SimpleNamespace(data=self.tool_results.get(name, {}))


@pytest.fixture
def fake_group_clients(monkeypatch) -> dict[str, FakeGroupClient]:
    """
    ``{group_url: FakeGroupClient}``, populated as ``UrbanDataMcpClient`` constructs one
    per group. Tests configure ``.tools``/``.tool_results``/``.raise_on_list`` on the
    relevant entries before exercising the client under test.
    """
    created: dict[str, FakeGroupClient] = {}

    def factory(url: str, auth=None) -> FakeGroupClient:
        client = FakeGroupClient(url, auth)
        created[url] = client
        return client

    monkeypatch.setattr(
        "src.agents.mcp_clients.urban_data_mcp_client.McpClient", factory
    )
    return created


def _url(group_path: str) -> str:
    return f"{BASE_URL}/{group_path}"


class TestConstruction:
    def test_builds_one_client_per_group(self, fake_group_clients):
        UrbanDataMcpClient(base_url=BASE_URL, token="tok")

        assert set(fake_group_clients) == {
            _url("dictionaries"),
            _url("territories"),
            _url("indicators"),
            _url("physical_objects"),
            _url("projects"),
            _url("soc-groups"),
        }

    def test_only_projects_group_gets_the_token(self, fake_group_clients):
        UrbanDataMcpClient(base_url=BASE_URL, token="tok")

        assert fake_group_clients[_url("projects")].auth == "tok"
        for group_path in (
            "dictionaries",
            "territories",
            "indicators",
            "physical_objects",
            "soc-groups",
        ):
            assert fake_group_clients[_url(group_path)].auth is None

    def test_no_token_means_projects_client_is_also_unauthenticated(
        self, fake_group_clients
    ):
        UrbanDataMcpClient(base_url=BASE_URL, token=None)

        assert fake_group_clients[_url("projects")].auth is None


class TestGetTools:
    @pytest.mark.asyncio
    async def test_aggregates_tools_from_every_group(self, fake_group_clients):
        client = UrbanDataMcpClient(base_url=BASE_URL, token="tok")
        fake_group_clients[_url("dictionaries")].tools = [_tool("GetTerritoryTypes")]
        fake_group_clients[_url("territories")].tools = [_tool("GetTerritoryById")]
        fake_group_clients[_url("projects")].tools = [_tool("GetProjectById")]

        tools = await client.get_tools()

        names = {t["function"]["name"] for t in tools}
        assert names == {"GetTerritoryTypes", "GetTerritoryById", "GetProjectById"}
        assert client._tool_group["GetTerritoryById"] == "territories"

    @pytest.mark.asyncio
    async def test_create_project_is_excluded(self, fake_group_clients):
        client = UrbanDataMcpClient(base_url=BASE_URL, token="tok")
        fake_group_clients[_url("projects")].tools = [
            _tool("GetProjectById"),
            _tool("CreateProject"),
        ]

        tools = await client.get_tools()

        names = {t["function"]["name"] for t in tools}
        assert "CreateProject" not in names
        assert "CreateProject" not in client._tool_group

    @pytest.mark.asyncio
    async def test_unreachable_group_is_skipped_not_fatal(self, fake_group_clients):
        client = UrbanDataMcpClient(base_url=BASE_URL, token="tok")
        fake_group_clients[_url("territories")].tools = [_tool("GetTerritoryById")]
        fake_group_clients[_url("soc-groups")].raise_on_list = RuntimeError(
            "Session terminated"
        )

        tools = await client.get_tools()

        assert {t["function"]["name"] for t in tools} == {"GetTerritoryById"}

    @pytest.mark.asyncio
    async def test_duplicate_tool_name_keeps_first_group(self, fake_group_clients):
        client = UrbanDataMcpClient(base_url=BASE_URL, token="tok")
        # dictionaries is iterated before territories (_GROUP_PATHS insertion order).
        fake_group_clients[_url("dictionaries")].tools = [_tool("Dup")]
        fake_group_clients[_url("territories")].tools = [_tool("Dup")]

        tools = await client.get_tools()

        assert len([t for t in tools if t["function"]["name"] == "Dup"]) == 1
        assert client._tool_group["Dup"] == "dictionaries"


class TestExecuteTool:
    @pytest.mark.asyncio
    async def test_dispatches_to_the_owning_group(self, fake_group_clients):
        client = UrbanDataMcpClient(base_url=BASE_URL, token="tok")
        fake_group_clients[_url("territories")].tools = [_tool("GetTerritoryById")]
        fake_group_clients[_url("territories")].tool_results["GetTerritoryById"] = {
            "territory_id": 1
        }
        await client.get_tools()

        result = await client.execute_tool("GetTerritoryById", {"territory_id": 1})

        assert result == {"territory_id": 1}
        assert fake_group_clients[_url("territories")].call_tool_calls == [
            {"name": "GetTerritoryById", "arguments": {"territory_id": 1}, "meta": {}}
        ]

    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self, fake_group_clients):
        client = UrbanDataMcpClient(base_url=BASE_URL, token="tok")
        await client.get_tools()

        with pytest.raises(ValueError):
            await client.execute_tool("NoSuchTool", {})


class TestUpdateToken:
    def test_rebuilds_only_the_projects_client(self, fake_group_clients):
        client = UrbanDataMcpClient(base_url=BASE_URL, token="tok")
        original_territories_client = fake_group_clients[_url("territories")]

        client.update_token("new-token")

        assert fake_group_clients[_url("projects")].auth == "new-token"
        # The other groups' underlying client is untouched (same instance, same auth).
        assert fake_group_clients[_url("territories")] is original_territories_client
        assert fake_group_clients[_url("territories")].auth is None
