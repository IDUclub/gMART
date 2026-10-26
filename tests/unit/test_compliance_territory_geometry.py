"""Documents in force where the boundary of uploaded files lies."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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


async def resolve_geometry_documents(territory, documents, *, complete=True):
    idu = SimpleNamespace(
        territories_by_geometry=AsyncMock(
            return_value={"territories": [territory], "complete": complete}
        )
    )
    dvd = SimpleNamespace(list_documents=AsyncMock(return_value=documents))
    graph = SimpleNamespace(
        list_restriction_documents=AsyncMock(
            return_value=[{"name": document["name"]} for document in documents]
        )
    )
    result = await ComplianceTerritoryFilter().resolve_geometry(
        idu, dvd, graph, GEOMETRY
    )
    dvd.list_documents.assert_awaited_once_with(
        territory_ids=[int(territory["territory_id"])]
    )
    return result


@pytest.mark.parametrize("complete", [True, False])
async def test_root_leaf_excludes_descendants_only_after_complete_descent(complete):
    result = await resolve_geometry_documents(
        {"territory_id": 12639, "name": "Россия", "parent_id": None},
        [
            {
                "name": "Федеральный документ",
                "territory_id": 12639,
                "territory_path": [12639],
            },
            {
                "name": "Документ Ленинградской области",
                "territory_id": 1,
                "territory_path": [12639, 1],
            },
            {
                "name": "Документ Гатчины",
                "territory_id": 73,
                "territory_path": [12639, 1, 67, 73],
            },
        ],
        complete=complete,
    )
    assert result.status == "ok"
    descendants = ("Документ Ленинградской области", "Документ Гатчины")
    expected = ("Федеральный документ",) + (() if complete else descendants)
    assert result.allowed == expected
    assert result.excluded == (descendants if complete else ())
    assert result.outside_loaded_regions is complete
    assert result.note().count("вне регионов, загруженных в Urban API") == int(complete)
    assert result.message is None


async def test_regional_leaf_keeps_its_own_and_federal_documents():
    result = await resolve_geometry_documents(
        {"territory_id": 13368, "name": "Республика Карелия", "parent_id": 12639},
        [
            {
                "name": "Документ Карелии",
                "territory_id": 13368,
                "territory_path": [12639, 13368],
            },
            {
                "name": "Федеральный документ",
                "territory_id": 12639,
                "territory_path": [12639],
            },
        ],
    )
    assert result.status == "ok"
    assert result.allowed == ("Документ Карелии", "Федеральный документ")
    assert result.excluded == ()
    assert result.outside_loaded_regions is False
    assert "вне регионов, загруженных в Urban API" not in result.note()
    assert result.message is None


async def test_descendant_filter_normalizes_ids_and_keeps_missing_paths():
    result = await resolve_geometry_documents(
        {"territory_id": "13368", "name": "Республика Карелия", "parent_id": 12639},
        [
            {"name": "Без пути", "territory_id": 73},
            {
                "name": "Документ Карелии",
                "territory_id": "13368",
                "territory_path": [12639, "13368", "invalid", None],
            },
            {
                "name": "Документ потомка",
                "territory_id": "73",
                "territory_path": ["12639", "13368", "invalid", None, "73"],
            },
            {
                "name": "Неизвестные идентификаторы",
                "territory_id": "invalid",
                "territory_path": ["invalid", None],
            },
        ],
    )
    assert result.status == "ok"
    assert result.allowed == (
        "Без пути",
        "Документ Карелии",
        "Неизвестные идентификаторы",
    )
    assert result.excluded == ("Документ потомка",)


async def test_root_leaf_without_applicable_documents_explains_why_once():
    result = await resolve_geometry_documents(
        {"territory_id": 12639, "name": "Россия", "parent_id": None},
        [
            {
                "name": "Документ Гатчины",
                "territory_id": 73,
                "territory_path": [12639, 1, 67, 73],
            }
        ],
    )
    assert result.status == "empty"
    assert result.allowed == ()
    assert result.excluded == ("Документ Гатчины",)
    assert result.outside_loaded_regions is True
    assert result.message.count("вне регионов, загруженных в Urban API") == 1


@pytest.mark.parametrize("outside_loaded_regions", [True, False])
def test_territory_documents_round_trip_keeps_outside_loaded_regions(
    outside_loaded_regions,
):
    result = TerritoryDocuments(
        status="ok",
        allowed=("Федеральный документ",),
        place="files",
        territories=("Россия",),
        outside_loaded_regions=outside_loaded_regions,
    )
    assert result.to_dict()["outside_loaded_regions"] is outside_loaded_regions
    assert TerritoryDocuments.from_dict(result.to_dict()) == result


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
