import json

import pytest

from src.agents.services.compilance.compliance_scope import (
    ComplianceScope,
    ComplianceScopeResolver,
    designates,
    render_choice,
    scope_for_choice,
)


class ScriptedLlm:
    """Returns the queued JSON answers in order and records every prompt."""

    def __init__(self, *answers: dict) -> None:
        self.answers = list(answers)
        self.calls: list[list[dict]] = []

    async def chat(self, *, model, messages, **kwargs):
        self.calls.append(messages)
        if not self.answers:
            raise AssertionError("unexpected LLM call")
        return {"message": {"content": json.dumps(self.answers.pop(0))}}


class FakeNormGraph:
    def __init__(self, candidates=None, pool=None, catalogue=None) -> None:
        self.candidates = candidates or {}
        self.pool = pool or []
        self.catalogue = catalogue if catalogue is not None else list(self.pool)
        self.calls: list[tuple[str, dict]] = []

    async def resolve_entities(self, terms, limit=10):
        self.calls.append(("resolve_entities", {"terms": terms, "limit": limit}))
        return [
            {"term": term, "candidates": self.candidates.get(term, [])}
            for term in terms
        ]

    async def list_restriction_documents(
        self, executable_only=False, limit=200, **filters
    ):
        self.calls.append(
            (
                "list_restriction_documents",
                {"executable_only": executable_only, "limit": limit, **filters},
            )
        )
        return self.pool if executable_only else self.catalogue


def _entity(normalized, match="text", executable=1):
    return {
        "normalized": normalized,
        "aliases": [],
        "restriction_count": executable,
        "executable_count": executable,
        "match": match,
    }


def _doc(name, executable=1):
    return {"name": name, "executable_count": executable, "restriction_count": 3}


@pytest.mark.parametrize(
    ("reference", "name", "expected"),
    [
        ("СП 42", "СП 42.13330.2016", True),
        ("сп42.13330", "СП 42.13330.2016 Градостроительство", True),
        ("СП 42", "СП 421.1325800.2018", False),
        ("СП 42.13330.2016", "СП 42.13330.2011", False),
        ("СанПиН 2.2.1", "СанПиН 2.2.1/2.1.1.1200-03", True),
    ],
)
def test_designation_ignores_spacing_but_not_a_continued_number(
    reference, name, expected
):
    assert designates(reference, name) is expected


async def test_request_without_topic_or_document_keeps_the_full_audit():
    llm = ScriptedLlm({"topics": [], "documents": []})
    client = FakeNormGraph()

    outcome = await ComplianceScopeResolver(llm).resolve(
        client, "m", "Проверь соответствие нормам"
    )

    assert outcome.kind == "scoped"
    assert not outcome.scope.is_filtered
    assert client.calls == []


async def test_topic_maps_only_to_offered_entities():
    llm = ScriptedLlm(
        {"topics": ["школа"], "documents": []},
        {
            "selections": [
                {
                    "topic": "школа",
                    "entities": ["школа", "общеобразовательная школа", "детский сад"],
                }
            ]
        },
    )
    client = FakeNormGraph(
        candidates={
            "школа": [
                _entity("школа", "exact"),
                _entity("общеобразовательная школа"),
                _entity("спортивная школа"),
            ]
        }
    )

    outcome = await ComplianceScopeResolver(llm).resolve(
        client, "m", "Проверь нормы касательно школ"
    )

    assert outcome.kind == "scoped"
    assert outcome.scope.entities == ("общеобразовательная школа", "школа")
    assert outcome.scope.filters() == {
        "entities": ["общеобразовательная школа", "школа"]
    }
    # The invented «детский сад» never reaches NormGraph filters.
    assert "детский сад" not in outcome.scope.entities


async def test_empty_selection_falls_back_to_the_topics_own_entity():
    llm = ScriptedLlm(
        {"topics": ["школа"], "documents": []},
        {"selections": []},
    )
    client = FakeNormGraph(
        candidates={"школа": [_entity("школа", "alias"), _entity("школа искусств")]}
    )

    outcome = await ComplianceScopeResolver(llm).resolve(client, "m", "по школам")

    assert outcome.scope.entities == ("школа",)


async def test_unknown_topic_stops_with_an_explanation():
    llm = ScriptedLlm({"topics": ["космодром"], "documents": []})
    client = FakeNormGraph()

    outcome = await ComplianceScopeResolver(llm).resolve(
        client, "m", "Проверь нормы по космодромам"
    )

    assert outcome.kind == "empty"
    assert "«космодром»" in outcome.message


async def test_exact_designation_of_one_document_needs_no_choice():
    llm = ScriptedLlm({"topics": [], "documents": ["СП 42.13330"]})
    client = FakeNormGraph(
        pool=[_doc("СП 42.13330.2016", 12), _doc("СанПиН 2.2.1/2.1.1.1200-03", 3)]
    )

    outcome = await ComplianceScopeResolver(llm).resolve(
        client, "m", "Проверь нормы из СП 42.13330"
    )

    assert outcome.kind == "scoped"
    assert outcome.scope.documents == ("СП 42.13330.2016",)


async def test_designation_matching_several_documents_asks_to_choose():
    llm = ScriptedLlm(
        {"topics": ["школа"], "documents": ["СП 42"]},
        {"selections": [{"topic": "школа", "entities": ["школа"]}]},
    )
    client = FakeNormGraph(
        candidates={"школа": [_entity("школа", "exact")]},
        pool=[_doc("СП 42.13330.2016", 4), _doc("СанПиН 1.2", 2)],
        catalogue=[
            _doc("СП 42.13330.2011"),
            _doc("СП 42.13330.2016"),
            _doc("СанПиН 1.2"),
        ],
    )

    outcome = await ComplianceScopeResolver(llm).resolve(
        client, "m", "Проверь нормы по школам из СП 42"
    )

    assert outcome.kind == "choice"
    assert [c["name"] for c in outcome.choice["candidates"]] == [
        "СП 42.13330.2011",
        "СП 42.13330.2016",
    ]
    assert outcome.choice["candidates"][1]["executable_count"] == 4
    assert outcome.choice["entities"] == ["школа"]
    assert "1. СП 42.13330.2011 — исполнимых норм: 0" in outcome.message
    # Candidates are limited to documents with executable norms on the topic.
    assert (
        "list_restriction_documents",
        {"executable_only": True, "limit": 500, "entities": ["школа"]},
    ) in client.calls


async def test_described_document_is_ranked_by_matching_words():
    llm = ScriptedLlm({"topics": [], "documents": ["санпин о санитарных разрывах"]})
    client = FakeNormGraph(
        pool=[
            _doc("СП 42.13330.2016 Градостроительство", 20),
            _doc("СанПиН 2.2.1/2.1.1.1200-03 Санитарно-защитные зоны", 5),
            _doc("СП 4.13130.2013 Пожарная безопасность", 9),
        ]
    )

    outcome = await ComplianceScopeResolver(llm).resolve(
        client, "m", "Проверь по санпину о санитарных разрывах"
    )

    assert outcome.kind == "choice"
    assert outcome.choice["matched"] is True
    assert [c["name"] for c in outcome.choice["candidates"]] == [
        "СанПиН 2.2.1/2.1.1.1200-03 Санитарно-защитные зоны"
    ]


async def test_unmatched_description_offers_the_documents_with_most_norms():
    llm = ScriptedLlm({"topics": [], "documents": ["региональные нормативы"]})
    client = FakeNormGraph(pool=[_doc("СП 4.13130.2013", 9), _doc("СП 42", 20)])

    outcome = await ComplianceScopeResolver(llm).resolve(
        client, "m", "Проверь по региональным нормативам"
    )

    assert outcome.kind == "choice"
    assert outcome.choice["matched"] is False
    assert [c["name"] for c in outcome.choice["candidates"]] == [
        "СП 42",
        "СП 4.13130.2013",
    ]
    assert "не найден среди документов" in outcome.message


async def test_no_documents_with_executable_norms_stops():
    llm = ScriptedLlm({"topics": [], "documents": ["какой-нибудь свод правил"]})

    outcome = await ComplianceScopeResolver(llm).resolve(
        FakeNormGraph(), "m", "Проверь по какому-нибудь своду правил"
    )

    assert outcome.kind == "empty"


def _choice():
    return {
        "query": "Проверь нормы по школам из СП 42",
        "topics": ["школа"],
        "entities": ["школа"],
        "documents": [],
        "references": ["СП 42"],
        "matched": True,
        "candidates": [
            {"name": "СП 42.13330.2011", "executable_count": 1},
            {"name": "СП 42.13330.2016", "executable_count": 4},
            {"name": "СанПиН 1.2", "executable_count": 2},
        ],
    }


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("2", ("СП 42.13330.2016",)),
        ("Вариант 1", ("СП 42.13330.2011",)),
        ("1, 3", ("СП 42.13330.2011", "СанПиН 1.2")),
        ("1-2", ("СП 42.13330.2011", "СП 42.13330.2016")),
        ("все", ("СП 42.13330.2011", "СП 42.13330.2016", "СанПиН 1.2")),
        ("второй", ("СП 42.13330.2016",)),
        ("СП 42.13330.2016", ("СП 42.13330.2016",)),
    ],
)
async def test_choice_reply_is_parsed_without_the_model(reply, expected):
    llm = ScriptedLlm()

    result = await ComplianceScopeResolver(llm).resolve_choice("m", reply, _choice())

    assert result.kind == "selected"
    assert result.documents == expected


@pytest.mark.parametrize("reply", ["5", "СП 42"])
async def test_reply_that_picks_no_single_option_is_unresolved(reply):
    result = await ComplianceScopeResolver(ScriptedLlm()).resolve_choice(
        "m", reply, _choice()
    )

    assert result.kind == "unresolved"


async def test_free_text_reply_is_classified_by_the_model():
    llm = ScriptedLlm({"is_choice": False, "numbers": []})

    result = await ComplianceScopeResolver(llm).resolve_choice(
        "m",
        "А теперь проверь нормы по детским садам из СП 42.13330.2016 в этом сценарии",
        _choice(),
    )

    assert result.kind == "not_choice"
    assert len(llm.calls) == 1


def test_selected_documents_join_the_original_scope():
    choice = {**_choice(), "documents": ["СанПиН 1.2"]}

    scope = scope_for_choice(choice, ("СП 42.13330.2016",))

    assert scope == ComplianceScope(
        topics=("школа",),
        entities=("школа",),
        documents=("СанПиН 1.2", "СП 42.13330.2016"),
    )
    assert scope.label() == "темы: «школа»; документы: СанПиН 1.2, СП 42.13330.2016"


def test_rendered_choice_lists_numbered_options_and_how_to_answer():
    text = render_choice(_choice())

    assert text.startswith("Не удалось однозначно определить документ «СП 42»")
    assert "по теме «школа»" in text
    assert "2. СП 42.13330.2016 — исполнимых норм: 4" in text
    assert "«все»" in text


async def test_empty_scope_message_names_documents_with_executable_norms():
    client = FakeNormGraph(pool=[_doc("СП 42.13330.2016", 4)])
    scope = ComplianceScope(
        topics=("школа",), entities=("школа",), documents=("СанПиН 1.2",)
    )

    text = await ComplianceScopeResolver(ScriptedLlm()).empty_scope_message(
        client, scope, found=3
    )

    assert "Найдено норм по условиям: 3, исполнимых из них: 0." in text
    assert "- СП 42.13330.2016 — 4" in text
    assert (
        "list_restriction_documents",
        {"executable_only": True, "limit": 10, "entities": ["школа"]},
    ) in client.calls
