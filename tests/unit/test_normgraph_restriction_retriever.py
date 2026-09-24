from types import SimpleNamespace

import pytest

from src.agents.services.normgraph.normgraph_restriction_retriever import (
    NormGraphRestrictionRetriever,
)
from src.agents.services.service_entities.normgraph_plan import PrimaryTool


class FakePlanner:
    def __init__(self, plan):
        self.plan = plan

    async def build_plan(self, *args, **kwargs):
        return self.plan


class FakeClient:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    async def search_restrictions(self, **arguments):
        self.calls.append(("search_restrictions", arguments))
        return {"hits": self.hits}

    async def restrictions_applicable(self, **arguments):
        self.calls.append(("restrictions_applicable", arguments))
        return {"hits": self.hits}

    async def list_restrictions(self, limit, after_id=None):
        self.calls.append(("list_restrictions", {"limit": limit, "after_id": after_id}))
        ids = [hit["id"] for hit in self.hits]
        start = 0 if after_id is None else ids.index(after_id) + 1
        rows = self.hits[start : start + limit + 1]
        more = len(rows) > limit
        return {
            "hits": rows[:limit],
            "next_after_id": rows[limit - 1]["id"] if more else None,
        }


@pytest.mark.asyncio
async def test_retriever_keeps_only_canonical_metric_restrictions():
    hits = [
        {
            "id": "r-ok",
            "kind": "минимальное_расстояние",
            "value": {"number": 25.5, "unit": "м"},
        },
        {
            "id": "r-unit",
            "kind": "минимальное_расстояние",
            "value": {"number": 1, "unit": "км"},
        },
        {
            "id": "r-kind",
            "kind": "минимальная_ширина",
            "value": {"number": 25, "unit": "м"},
        },
    ]
    retriever = NormGraphRestrictionRetriever(llm_client=None)
    retriever.planner = FakePlanner(
        SimpleNamespace(
            primary_tool=PrimaryTool.SEARCH,
            search_query="дороги жильё",
            kind=None,
            document_names=None,
            doc_type=None,
            corpus=None,
            lang=None,
            tags=None,
            subject=None,
            object=None,
            limit=10,
        )
    )
    client = FakeClient(hits)

    result = await retriever.retrieve(client, "model", "query")

    assert [hit["id"] for hit in result.restrictions] == ["r-ok"]
    assert result.unsupported_count == 2
    assert client.calls == [
        (
            "search_restrictions",
            {"query": "дороги жильё", "limit": 10, "neighbors_depth": 0},
        )
    ]


@pytest.mark.asyncio
async def test_retriever_uses_applicable_tool_when_object_is_known():
    retriever = NormGraphRestrictionRetriever(llm_client=None)
    retriever.planner = FakePlanner(
        SimpleNamespace(
            primary_tool=PrimaryTool.APPLICABLE,
            object="жилой дом",
            subject="дорога",
            kind=None,
            document_names=None,
            limit=20,
        )
    )
    client = FakeClient([])

    await retriever.retrieve(client, "model", "query")

    assert client.calls == [
        (
            "restrictions_applicable",
            {"object": "жилой дом", "subject": "дорога", "limit": 20},
        ),
        (
            "search_restrictions",
            {"query": "query", "limit": 100, "neighbors_depth": 0},
        ),
    ]


def test_explicit_document_clause_and_distance_narrow_semantic_results():
    wanted = {
        "id": "wanted",
        "kind": "минимальное_расстояние",
        "value": {"number": 50.0, "unit": "м"},
        "provenance": {"name": "СП 4.13130.2013 (тест)", "numbering": "4.14"},
    }
    hits = [
        wanted,
        {
            "id": "wrong-distance",
            "kind": "минимальное_расстояние",
            "value": {"number": 30.0, "unit": "м"},
            "provenance": {"name": "СП 4.13130.2013", "numbering": "4.14"},
        },
        {
            "id": "wrong-clause",
            "kind": "минимальное_расстояние",
            "value": {"number": 50.0, "unit": "м"},
            "provenance": {"name": "СП 4.13130.2013", "numbering": "6.1.6"},
        },
    ]

    narrowed = NormGraphRestrictionRetriever._filter_explicit_references(
        hits,
        "По СП 4.13130.2013, пункт 4.14, проверь расстояние 50 метров.",
    )

    assert narrowed == [wanted]


@pytest.mark.asyncio
async def test_exhaustive_request_lists_all_named_documents_without_llm_ranking():
    hits = [
        {
            "id": "sp-104",
            "kind": "минимальное_расстояние",
            "value": {"number": 5.0, "unit": "м"},
            "provenance": {"name": "СП 104.13330.2016 — корпус"},
        },
        {
            "id": "sp-42",
            "kind": "минимальное_расстояние",
            "value": {"number": 500.0, "unit": "м"},
            "provenance": {"name": "СП 42.13330.2016 — корпус"},
        },
        {
            "id": "other",
            "kind": "минимальное_расстояние",
            "value": {"number": 10.0, "unit": "м"},
            "provenance": {"name": "СП 4.13130.2013"},
        },
    ]
    retriever = NormGraphRestrictionRetriever(llm_client=None)
    client = FakeClient(hits)

    result = await retriever.retrieve(
        client,
        "model",
        (
            "Проверь все ограничения из СП 104.13330.2016 и СП 42.13330.2016. "
            "Верни результат по каждой норме."
        ),
    )

    assert [hit["id"] for hit in result.restrictions] == ["sp-104", "sp-42"]
    assert client.calls == [
        ("search_restrictions", {"limit": 100, "neighbors_depth": 0})
    ]


@pytest.mark.asyncio
async def test_compliance_retrieval_fetches_all_norms_without_llm_limit():
    hits = [
        {
            "id": f"r-{index}",
            "kind": "минимальное_расстояние",
            "value": {"number": 10.0, "unit": "м"},
        }
        for index in range(300)
    ]
    retriever = NormGraphRestrictionRetriever(llm_client=None)
    retriever.planner = FakePlanner(None)
    client = FakeClient(hits)

    result = await retriever.retrieve(
        client,
        "model",
        "Проверь соответствие проекта",
        retain_unsupported=True,
        retrieve_all=True,
    )

    assert len(result.restrictions) == 300
    # keyset pages within NormGraph's 500-row cap, never one corpus-sized window
    assert client.calls == [
        ("list_restrictions", {"limit": 200, "after_id": None}),
        ("list_restrictions", {"limit": 200, "after_id": "r-199"}),
    ]
    assert result.tool_call["function"]["name"] == "list_restrictions"


@pytest.mark.asyncio
async def test_compliance_retrieval_rejects_a_cursor_that_does_not_advance():
    class StuckClient(FakeClient):
        async def list_restrictions(self, limit, after_id=None):
            return {"hits": self.hits[:1], "next_after_id": "r-0"}

    retriever = NormGraphRestrictionRetriever(llm_client=None)
    with pytest.raises(RuntimeError, match="did not advance"):
        await retriever.retrieve(
            StuckClient([{"id": "r-0"}]),
            "model",
            "Проверь соответствие проекта",
            retrieve_all=True,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"planner_status": "unsupported"},
        {"planner_status": {}},
        {"template": "unsupported"},
        {"template_version": 999},
        {"schema_version": "2.0"},
        {"params": {}},
        {"source": {}},
    ],
)
def test_compliance_filters_unusable_plans(change):
    from tests.unit.test_executable_compliance_pipeline import _plan

    valid = {"id": "valid", "check_plan": _plan()}
    invalid = {"id": "invalid", "check_plan": {**_plan(), **change}}
    hits = [valid, invalid, {"id": "missing"}, {"check_plan": None}, {"check_plan": {}}]
    result = NormGraphRestrictionRetriever._result(
        hits, {}, "search_restrictions", False, True
    )
    assert result.restrictions == [valid]
    assert result.unsupported_count == 4


def test_compliance_keeps_reviewed_plans_without_metric_restriction_fields():
    from tests.unit.test_executable_compliance_pipeline import _plan

    hit = {"check_plan": {**_plan(), "planner_status": "reviewed"}}
    assert NormGraphRestrictionRetriever.has_executable_plan(hit)
