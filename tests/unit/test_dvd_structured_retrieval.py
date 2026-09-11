import copy
import json
from types import SimpleNamespace

import pytest

from src.agents.services.dvd.dvd_reasoning import RetrievalPlanner
from src.agents.services.service_entities.dvd_plan import (
    SemanticRetrievalPlan,
    StructureRetrievalPlan,
)
from tests.helpers import answer_text, plan_json, verdict_json

CODE = "Градостроительный кодекс Российской Федерации"
EDITION = "N\u202f190‑ФЗ (ред. от\u00a030.01.2026, с изм. и доп., вступ. в силу с\u00a001.07.2026)"
CHOICE = f"{CODE}, редакция {EDITION}: 3.3"


def test_clarification_ranks_russian_inflections_in_clause_excerpt():
    from src.agents.services.dvd.clarification import ranked_choices

    base = {"name": CODE, "version": EDITION}
    candidates = [
        {
            **base,
            "id": "article49",
            "selection_path": ["статья 49", "пункт 3.3"],
            "excerpt": "Проектная документация объектов капитального строительства",
        },
        {
            **base,
            "id": "article52",
            "selection_path": ["статья 52", "пункт 3.3"],
            "excerpt": "По решению застройщика или технического заказчика этапы строительства",
        },
    ]
    choices = ranked_choices(
        candidates,
        "Что говорится о выделении этапов строительства в пункте 3.3 " + CODE + "?",
    )
    assert "статья 52" in choices[0]


def test_dates_in_copied_candidate_are_not_document_designations():
    plan = RetrievalPlanner._clamp(
        SemanticRetrievalPlan(retrieval_mode="semantic"), CHOICE
    )
    assert plan.retrieval_mode == "structure"
    assert plan.pattern == "3.3"
    assert plan.document_names == [CODE]
    assert plan.version == EDITION


async def test_clarification_deduplicates_and_ranks_by_question(service, fake_llm):
    fake_llm.json_responses = [plan_json()]
    candidate = {"name": CODE, "version": EDITION, "structure_path": ["3.3"]}
    client = Pages(
        [
            {
                "ambiguous": True,
                "candidates": [
                    {
                        "name": "СП 309.1325800.2017",
                        "version": "2017",
                        "structure_path": ["3", "3.3 аппаратная"],
                    },
                    {**candidate, "id": "one"},
                    {**candidate, "id": "one"},
                    {**candidate, "structure_path": ["52", "3.3"]},
                ],
            }
        ]
    )
    events = await run(
        service, client, "Расскажи о пункте 3.3 Градостроительного кодекса"
    )
    options = [
        line for line in answer_text(events).splitlines() if line.startswith("- ")
    ]
    assert len(options) == 3
    assert CODE in options[0]
    assert options[0].endswith(": 52 / 3.3")


@pytest.mark.parametrize(
    "reply", [CHOICE.replace("\u00a0", " ").replace("\u202f", " "), "первый вариант"]
)
async def test_selected_candidate_reaches_structural_answer(service, fake_llm, reply):
    service.get_chat_messages.return_value = SimpleNamespace(
        messages=[
            {"role": "user", "content": "О чём пункт 3.3?"},
            {
                "role": "assistant",
                "content": "Нашлось несколько подходящих элементов. Уточните документ, редакцию или структурный путь:\n\n- "
                + CHOICE,
            },
        ]
    )
    # Selection needs no LLM planning; the sole JSON request is the answer critic.
    fake_llm.json_responses = [verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Пункт устанавливает порядок действий [1]."]

    class Selected(Pages):
        tool_name_for_kind = staticmethod(lambda kind: "search_all")

        async def search(self, *args, **kwargs):
            return {"hits": []}

    client = Selected(
        [
            {
                "hits": [
                    {
                        "id": "one",
                        "name": CODE,
                        "text": "Пункт устанавливает порядок действий.",
                    }
                ],
                "total": 1,
                "complete": True,
            }
        ]
    )
    events = await run(service, client, reply)
    assert answer_text(events) == "Пункт устанавливает порядок действий [1]."
    mode, request = client.calls[0]
    assert mode == "structure" and request["pattern"] == "3.3"
    assert request["document_names"] == [CODE] and request["version"] == EDITION


async def test_two_turn_choice_keeps_only_selected_roots_and_children(
    service, fake_llm
):
    # Keep this retrieval regression independent of the separately tested reducer.
    service.context_reducer.configured_window = 32768
    root = {
        "id": "root",
        "name": CODE,
        "version": EDITION,
        "structure_path": ["3.3"],
        "parent_id": "parent",
        "content_digest": "identical-subtrees",
    }
    duplicate = {**root, "id": "duplicate"}
    other = {**root, "id": "other", "structure_path": ["52", "3.3"]}
    candidates = [other, *([root] * 25), duplicate]
    fake_llm.json_responses = [plan_json()]
    events = await run(
        service,
        Pages([{"ambiguous": True, "candidates": candidates}]),
        "О чём пункт 3.3?",
    )
    clarification = answer_text(events)
    assert (
        len([line for line in clarification.splitlines() if line.startswith("- ")]) == 2
    )
    assert "52 / 3.3" in clarification
    service.get_chat_messages.return_value = SimpleNamespace(
        messages=[
            {"role": "user", "content": "О чём пункт 3.3?"},
            {
                "role": "assistant",
                "parts": [{"kind": "text", "payload": {"text": clarification}}],
            },
        ]
    )
    fake_llm.json_responses = [verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Выбранный пункт с дочерним уточнением [1] [3]."]
    # A bare 3.3 also matches 52 / 3.3 upstream. Filter by the actual offered
    # identity after fetching every page, preserving children of duplicate roots.
    client = Pages(
        [
            {
                "ambiguous": True,
                "candidates": [other, root, duplicate],
                "hits": [
                    {**other, "text": "WRONG SECTION", "matched": True},
                    {**root, "text": "SELECTED ROOT", "matched": True},
                ],
                "total": 4,
                "complete": False,
                "next_cursor": "next",
            },
            {
                "ambiguous": True,
                "candidates": [other, root, duplicate],
                "hits": [
                    {**duplicate, "text": "SELECTED ROOT", "matched": True},
                    {
                        "id": "child",
                        "name": CODE,
                        "text": "CHILD EXCEPTION",
                        "matched": False,
                        "matched_ancestor_ids": ["duplicate"],
                    },
                ],
                "total": 4,
                "complete": True,
            },
        ]
    )
    events = await run(service, client, "второй вариант")
    assert answer_text(events) == "Выбранный пункт с дочерним уточнением [1] [3]."
    context = next(c.messages[0]["content"] for c in fake_llm.chat_calls if c.stream)
    assert "SELECTED ROOT" in context and "SELECTED ROOT" in context
    assert "CHILD EXCEPTION" in context and "WRONG SECTION" not in context
    assert len(client.calls) == 2


async def test_only_duplicate_candidates_do_not_require_clarification(
    service, fake_llm
):
    candidate = {
        "name": CODE,
        "version": EDITION,
        "structure_path": ["3.3"],
        "parent_id": "parent",
        "content_digest": "same-complete-text",
    }
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Ответ по пункту [1]."]
    client = Pages(
        [
            {
                "ambiguous": True,
                "candidates": [{**candidate, "id": "a"}, {**candidate, "id": "b"}],
                "hits": [
                    {"id": "a", "name": CODE, "text": "Текст пункта"},
                    {"id": "b", "name": CODE, "text": "Текст пункта"},
                ],
                "total": 2,
                "complete": True,
            }
        ]
    )
    events = await run(service, client, "Расскажи о пункте 3.3")
    assert answer_text(events) == "Ответ по пункту [1]."


async def test_incomplete_candidate_list_is_not_treated_as_unique(service, fake_llm):
    fake_llm.json_responses = [plan_json()]
    client = Pages(
        [
            {
                "ambiguous": True,
                "candidates_complete": False,
                "candidates": [
                    {"name": CODE, "version": EDITION, "structure_path": ["3.3"]},
                ],
            }
        ]
    )
    events = await run(service, client)
    assert "Уточните" in answer_text(events)
    assert "полный список" in answer_text(events)
    assert not any(c.stream for c in fake_llm.chat_calls)


def test_edition_dates_do_not_replace_explicit_document_filter():
    plan = RetrievalPlanner._clamp(
        SemanticRetrievalPlan(retrieval_mode="semantic", document_names=[CODE]),
        f"Что изменилось в {CODE} в редакции от 30.01.2026?",
    )
    assert plan.document_names == [CODE]


def test_choices_preserve_editions_ancestors_and_amendments():
    from src.agents.services.dvd.clarification import parse_choice, ranked_choices

    root = {"name": CODE, "version": EDITION, "structure_path": ["3.3"]}
    options = ranked_choices(
        [
            root,
            {**root, "structure_path": ["52", "3.3"]},
            {**root, "version": "2020"},
            {**root, "block": "amendment"},
        ],
        "",
    )
    assert len(options) == 4
    assert (
        parse_choice(next(o for o in options if "52 / 3.3" in o))["pattern"]
        == "52 / 3.3"
    )
    assert (
        parse_choice(next(o for o in options if o.endswith("[изменения]")))["block"]
        == "amendment"
    )


def test_ordinal_selection_does_not_reuse_an_old_clarification():
    from src.agents.services.dvd.clarification import CLARIFICATION, selected_choice

    history = [
        {"role": "assistant", "content": CLARIFICATION + "\n\n- " + CHOICE},
        {"role": "assistant", "content": "Ответ уже дан."},
    ]
    assert selected_choice("первый вариант", history) is None


def test_explicit_clause_overrides_semantic_planner_and_preserves_document():
    plan = RetrievalPlanner._clamp(
        SemanticRetrievalPlan(
            retrieval_mode="semantic", search_query="пожар", types=["clause"]
        ),
        "о чем говорится в пункте 3.3 СП 2.13130.2020 2020?",
    )
    assert plan.retrieval_mode == "structure" and plan.pattern == "3.3"
    assert plan.types is None and plan.kind == "all"
    assert plan.document_names == ["СП 2.13130.2020"]


class Pages:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    async def search_fragments(self, request, *, mode):
        self.calls.append((mode, copy.deepcopy(request)))
        return self.pages.pop(0)


async def run(service, client, query="что в пункте 3.3 СП 2.13130.2020?"):
    return [
        event
        async for event in service.run_document_qa_pipeline(
            dvd_mcp_client=client,
            token="t",
            model="m",
            temperature=0,
            user_query=query,
            chat_id="chat-1",
        )
    ]


async def test_all_pages_and_descendants_reach_answer(service, fake_llm):
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Полный ответ [1] [2]."]
    client = Pages(
        [
            {
                "hits": [
                    {
                        "id": "a",
                        "name": "СП",
                        "numbering": "3.3",
                        "text": "Определение.",
                    }
                ],
                "total": 2,
                "complete": False,
                "next_cursor": "cursor",
            },
            {
                "hits": [
                    {
                        "id": "b",
                        "name": "СП",
                        "numbering": "3.3.1",
                        "text": "Обязательное исключение.",
                    }
                ],
                "total": 2,
                "complete": True,
            },
        ]
    )
    events = await run(service, client)
    assert answer_text(events) == "Полный ответ [1] [2]."
    assert len(client.calls) == 2
    assert client.calls[0][1]["context_height"] == 1
    assert client.calls[0][1] == {
        k: v for k, v in client.calls[1][1].items() if k != "cursor"
    }
    draft = next(c for c in fake_llm.chat_calls if c.stream)
    assert "Определение." in draft.messages[0]["content"]
    assert "Обязательное исключение." in draft.messages[0]["content"]


async def test_not_found_does_not_fall_back_to_semantic_or_remove_filters(
    service, fake_llm
):
    fake_llm.json_responses = [plan_json()]
    client = Pages([{"hits": [], "complete": True, "total": 0}])
    events = await run(service, client)
    assert "совпадений не найдено" in answer_text(events)
    assert len(client.calls) == 1
    assert not any(c.stream for c in fake_llm.chat_calls)


async def test_ambiguity_asks_instead_of_selecting_arbitrary_edition(service, fake_llm):
    fake_llm.json_responses = [plan_json()]
    client = Pages(
        [
            {
                "hits": [],
                "ambiguous": True,
                "candidates": [
                    {"name": "СП", "version": "2020", "numbering": "3.3"},
                    {"name": "СП", "version": "2024", "numbering": "3.3"},
                ],
            }
        ]
    )
    events = await run(service, client)
    assert "Уточните" in answer_text(events)
    assert "2020" in answer_text(events) and "2024" in answer_text(events)


async def test_name_parameters_are_passed_to_the_name_tool(service, fake_llm):
    fake_llm.json_responses = [
        json.dumps(
            {
                "retrieval_mode": "name",
                "name_query": "Защита",
                "name_scope": "path",
                "name_mode": "expanded",
            }
        )
    ]
    client = Pages([{"hits": [], "complete": True, "total": 0}])
    await run(service, client, "Найди внутри раздела Защита")
    mode, request = client.calls[0]
    assert (
        mode == "name"
        and request["name_scope"] == "path"
        and request["name_mode"] == "expanded"
    )


@pytest.mark.parametrize("address", ["п.3.3", "пункте 3.3", "section 3.3"])
def test_compact_address_and_lowercase_designation(address):
    plan = RetrievalPlanner._clamp(
        SemanticRetrievalPlan(retrieval_mode="semantic"),
        f"{address} сп2.13130.2020",
    )
    assert plan.pattern == "3.3" and plan.document_names == ["сп2.13130.2020"]


def test_explicit_clause_does_not_remove_ancestor_path():
    plan = RetrievalPlanner._clamp(
        StructureRetrievalPlan(retrieval_mode="structure", pattern="А / 3.3"),
        "пункт 3.3 приложения А",
    )
    assert plan.pattern == "А / 3.3"


async def test_large_retrieval_flows_through_parallel_reducer(service, fake_llm):
    from tests.unit.test_dvd_context_reducer import Summarizer

    llm = Summarizer()
    service.context_reducer.llm_client = llm
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Условие FACT1 и исключение FACT2 [1] [2]."]
    client = Pages(
        [
            {
                "hits": [
                    {"id": "a", "name": "СП", "text": "padding " * 800 + "FACT1"},
                    {"id": "b", "name": "СП", "text": "padding " * 800 + "FACT2"},
                ],
                "total": 2,
                "complete": True,
            }
        ]
    )
    events = await run(service, client)
    draft = next(c for c in fake_llm.chat_calls if c.stream)
    assert (
        "FACT1" in draft.messages[0]["content"]
        and "FACT2" in draft.messages[0]["content"]
    )
    assert llm.peak > 1
    assert any(
        e.get("content", {}).get("status") == "context_processing" for e in events
    )
    assert "FACT2" in answer_text(events)


async def test_partial_failure_prevents_drafting_and_persistence(service, fake_llm):
    from tests.unit.test_dvd_context_reducer import Summarizer

    service.context_reducer.llm_client = Summarizer(fail=True)
    service.context_reducer.retries = 0
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Доступное условие FACT2 [2]."]
    client = Pages(
        [
            {
                "hits": [
                    {"id": "a", "name": "СП", "text": "FAIL_SOURCE " * 800},
                    {"id": "b", "name": "СП", "text": "padding " * 800 + "FACT2"},
                ],
                "total": 2,
                "complete": True,
            }
        ]
    )
    events = await run(service, client)
    assert not answer_text(events)
    assert any(e["type"] == "error" for e in events)
    assert not any(c.stream for c in fake_llm.chat_calls)
    service._schedule_persist_answer.assert_not_called()


async def test_truncated_final_generation_cannot_be_accepted(service):
    async def stream():
        yield SimpleNamespace(
            message=SimpleNamespace(content="Незаконченная фраза"), done_reason="length"
        )

    async def chat(*args, **kwargs):
        return stream()

    service.llm_client = SimpleNamespace(chat=chat)
    with pytest.raises(ValueError, match="answer_generation_incomplete"):
        [
            e
            async for e in service._generate_answer(
                "m", "question", "[1] source", 0, [], 1
            )
        ]


async def test_same_address_different_content_is_not_merged_and_choice_round_trips(
    service, fake_llm
):
    base = {
        "name": CODE,
        "version": EDITION,
        "structure_path": ["52", "3.3"],
        "parent_id": "article52",
    }
    a = {
        **base,
        "id": "a",
        "excerpt": "Строительство линейного объекта",
        "content_digest": "digest-a",
    }
    b = {
        **base,
        "id": "b",
        "excerpt": "Другое условие строительства",
        "content_digest": "digest-b",
    }
    fake_llm.json_responses = [plan_json()]
    events = await run(
        service,
        Pages([{"ambiguous": True, "candidates": [a, b]}]),
        "Пункт 3.3 про линейный объект",
    )
    answer = answer_text(events)
    options = [s[2:] for s in answer.splitlines() if s.startswith("- ")]
    assert len(options) == 2 and "линейного" in options[0]
    service.get_chat_messages.return_value = SimpleNamespace(
        messages=[{"role": "assistant", "content": answer}]
    )
    fake_llm.json_responses = [verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Выбранный текст [1]."]
    client = Pages(
        [
            {
                "ambiguous": True,
                "candidates": [a, b],
                "hits": [
                    {**a, "text": "SELECTED CONTENT", "matched": True},
                    {**b, "text": "WRONG CONTENT", "matched": True},
                ],
                "total": 2,
                "complete": True,
            }
        ]
    )
    result = await run(service, client, "первый вариант")
    assert answer_text(result) == "Выбранный текст [1]."
    assert client.calls[0][1]["pattern"] == "52 / 3.3"
    context = next(c.messages[0]["content"] for c in fake_llm.chat_calls if c.stream)
    assert "SELECTED CONTENT" in context and "WRONG CONTENT" not in context


def test_legacy_identical_labels_are_not_proof_of_identity():
    from src.agents.services.dvd.clarification import matching_choices, ranked_choices

    a = {"id": "a", "name": CODE, "version": EDITION, "structure_path": ["3.3"]}
    b = {**a, "id": "b"}
    options = ranked_choices([a, b], "")
    assert len(options) == 2
    assert matching_choices([a, b], CHOICE) == []
    assert matching_choices([a, b], options[0]) == [a]


def test_question_mark_after_clause_is_not_a_wildcard():
    plan = RetrievalPlanner._clamp(
        SemanticRetrievalPlan(retrieval_mode="semantic"), "О чём пункт 3.3?"
    )
    assert plan.pattern == "3.3"
