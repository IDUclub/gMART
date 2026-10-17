"""Documents in force where the boundary of uploaded files lies."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.agents.services.compilance.compliance_territory import (
    ComplianceTerritoryFilter,
    TerritoryDocuments,
)
from src.idu_mcp.tools_services.urb_api_tools import UrbanApiTool

GEOMETRY = {"type": "Point", "coordinates": [37.49, 55.675]}


def normgraph():
    return SimpleNamespace(
        list_restriction_documents=AsyncMock(
            return_value=[
                {"name": "СП 42.13330.2016", "doc_id": "d1"},
                {"name": "ПЗЗ Санкт-Петербурга", "doc_id": "d2"},
            ]
        )
    )


async def test_documents_of_the_territories_under_the_boundary():
    idu = SimpleNamespace(
        territories_by_geometry=AsyncMock(
            return_value={
                "territories": [
                    {"territory_id": 3299, "name": "Проспект Вернадского"},
                ],
                "complete": True,
            }
        )
    )
    dvd = SimpleNamespace(list_documents=AsyncMock(return_value=[{"doc_id": "d1"}]))
    result = await ComplianceTerritoryFilter().resolve_geometry(
        idu, dvd, normgraph(), GEOMETRY
    )
    dvd.list_documents.assert_awaited_once_with(territory_ids=[3299])
    assert result.status == "ok"
    assert result.allowed == ("СП 42.13330.2016",)
    assert result.place == "files"
    assert result.territories == ("Проспект Вернадского",)
    assert "проекта из загруженных файлов (Проспект Вернадского)" in result.note()
    assert TerritoryDocuments.from_dict(result.to_dict()) == result


async def test_no_boundary_and_no_territory_stop_the_check():
    dvd = SimpleNamespace(list_documents=AsyncMock())
    missing = await ComplianceTerritoryFilter().resolve_geometry(
        SimpleNamespace(), dvd, normgraph(), None
    )
    assert missing.status == "unavailable"
    assert "нет ни границы" in missing.message
    idu = SimpleNamespace(
        territories_by_geometry=AsyncMock(return_value={"territories": []})
    )
    nowhere = await ComplianceTerritoryFilter().resolve_geometry(
        idu, dvd, normgraph(), GEOMETRY
    )
    assert nowhere.status == "unavailable"
    assert "ни в одну территорию" in nowhere.message
    dvd.list_documents.assert_not_awaited()


class FakeUrbanClient:
    def __init__(self, tree):
        self.tree = tree
        self.calls = []

    async def get_root_territories(self, token):
        return [{"territory_id": 1, "name": "Россия", "level": 1}]

    async def get_intersecting_territories(self, parent_id, geometry, token):
        self.calls.append(parent_id)
        return [
            {
                "territory_id": child,
                "name": f"t{child}",
                "level": 2,
                "territory_type": {"name": "Округ"},
                "parent": {"id": parent_id, "name": f"t{parent_id}"},
            }
            for child in self.tree.get(parent_id, [])
        ]


async def test_descent_keeps_the_deepest_territories():
    client = FakeUrbanClient({1: [10], 10: [100, 101]})
    result = await UrbanApiTool(client).get_territories_by_geometry(GEOMETRY, "u")
    assert [item["territory_id"] for item in result["territories"]] == [100, 101]
    assert result["territories"][0]["parent_id"] == 10
    assert result["complete"] is True


async def test_spent_budget_keeps_the_coarser_frontier():
    client = FakeUrbanClient({1: [10], 10: [100]})
    result = await UrbanApiTool(client).get_territories_by_geometry(
        GEOMETRY, "u", max_requests=1
    )
    assert [item["territory_id"] for item in result["territories"]] == [10]
    assert result["complete"] is False
