import copy
import json
from types import SimpleNamespace

import pytest

from src.agents.services.dvd.dvd_reasoning import RetrievalPlanner
from src.agents.services.service_entities.dvd_plan import RetrievalPlan
from tests.helpers import answer_text, plan_json, verdict_json


def test_explicit_clause_overrides_semantic_planner_and_preserves_document():
    plan = RetrievalPlanner._clamp(
        RetrievalPlan(search_query="пожар", types=["clause"]),
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
    plan = RetrievalPlanner._clamp(RetrievalPlan(), f"{address} сп2.13130.2020")
    assert plan.pattern == "3.3" and plan.document_names == ["сп2.13130.2020"]


def test_explicit_clause_does_not_remove_ancestor_path():
    plan = RetrievalPlanner._clamp(
        RetrievalPlan(pattern="А / 3.3"), "пункт 3.3 приложения А"
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
    with pytest.raises(ValueError, match="before completion"):
        [
            e
            async for e in service._generate_answer(
                "m", "question", "[1] source", 0, [], 1
            )
        ]
