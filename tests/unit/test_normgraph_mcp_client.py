"""Normalization of NormGraph MCP results — FastMCP synthetic objects must become plain JSON.

FastMCP rehydrates structured tool output into synthetic types built from the output schema
(``Root`` objects without ``model_dump``); nested fields like ``provenance`` and ``value``
must be converted to dicts too, or the context builder crashes on ``provenance.get(...)``.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient


def _synthetic_hit():
    """Mimic a FastMCP synthetic object: attribute access only, no model_dump/dict."""
    return SimpleNamespace(
        id="382bfe03",
        subject="[не указано]",
        object="строительство",
        kind="запрет_размещения",
        value=SimpleNamespace(operator=">=", number=3.75, unit="м", condition=None),
        provenance=SimpleNamespace(
            doc_id="0e864faf",
            name="СП 30-102-99",
            version="1999",
            tags=["застройка"],
        ),
    )


class TestToDict:
    def test_converts_nested_objects_recursively(self):
        hit = NormGraphMcpClient._to_dict(_synthetic_hit())
        assert isinstance(hit, dict)
        assert isinstance(hit["provenance"], dict)
        assert hit["provenance"]["name"] == "СП 30-102-99"
        assert isinstance(hit["value"], dict)
        assert hit["value"]["number"] == 3.75

    def test_converts_lists_and_keeps_scalars(self):
        converted = NormGraphMcpClient._to_dict(
            {"hits": [_synthetic_hit()], "count": 1, "note": None}
        )
        assert converted["count"] == 1
        assert converted["note"] is None
        assert isinstance(converted["hits"][0]["provenance"], dict)
        assert converted["hits"][0]["provenance"]["tags"] == ["застройка"]

    def test_normalize_search_produces_plain_hits(self):
        result = SimpleNamespace(
            count=1, hits=[_synthetic_hit()], neighbors=[], dvd_fallback=[]
        )
        normalized = NormGraphMcpClient._normalize_search(result)
        assert normalized["count"] == 1
        provenance = normalized["hits"][0]["provenance"]
        assert provenance.get("name") == "СП 30-102-99"


class RecordingClient(NormGraphMcpClient):
    def __init__(self, result) -> None:
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    async def execute_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.result


async def test_resolve_entities_returns_plain_candidates():
    client = RecordingClient(
        [
            SimpleNamespace(
                term="школы",
                candidates=[SimpleNamespace(normalized="школа", match="alias")],
            )
        ]
    )

    result = await client.resolve_entities(["школы"], limit=5)

    assert client.calls == [("resolve_entities", {"terms": ["школы"], "limit": 5})]
    assert result == [
        {"term": "школы", "candidates": [{"normalized": "школа", "match": "alias"}]}
    ]


async def test_document_listing_sends_only_active_filters():
    client = RecordingClient(
        SimpleNamespace(
            count=1,
            documents=[SimpleNamespace(name="СП 42.13330.2016", executable_count=4)],
        )
    )

    documents = await client.list_restriction_documents(
        executable_only=True, limit=10, entities=["школа"], document_names=None
    )

    assert client.calls == [
        (
            "list_restriction_documents",
            {"executable_only": True, "limit": 10, "entities": ["школа"]},
        )
    ]
    assert documents == [{"name": "СП 42.13330.2016", "executable_count": 4}]
