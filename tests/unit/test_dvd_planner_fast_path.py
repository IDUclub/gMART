"""Stage 2: an explicit address with its document is planned without the LLM."""

from __future__ import annotations

import pytest

from src.agents.services.dvd.dvd_reasoning import RetrievalPlanner
from tests.helpers import answer_text

planner = RetrievalPlanner(llm_client=None)


@pytest.mark.parametrize(
    ("query", "pattern", "documents"),
    [
        ("Что сказано в пункте 5.3 СП 55.13330?", "5.3", ["СП 55.13330"]),
        ("Покажи пункт 31.6.1 СП 42.13330.2016", "31.6.1", ["СП 42.13330.2016"]),
        ("Процитируй раздел 7 СП 42.13330", "раздел 7", ["СП 42.13330"]),
        (
            "Что говорится в статье 51 ГрК РФ?",
            "статья 51",
            ["Градостроительный кодекс Российской Федерации"],
        ),
    ],
)
def test_explicit_address_and_document(query, pattern, documents):
    plan = planner.fast_plan(query)
    assert plan is not None
    assert plan.retrieval_mode == "structure"
    assert plan.pattern == pattern
    assert plan.document_names == documents
    assert not plan.rank_by_relevance and plan.include_children


def test_address_continues_the_selected_document():
    scope = {"document_names": ["СП 55.13330.2016"], "version": "2016"}
    plan = planner.fast_plan("А пункт 3.4 в нём?", scope=scope)
    assert plan.pattern == "3.4"
    assert plan.document_names == ["СП 55.13330.2016"]
    assert plan.version == "2016"


def test_address_continues_the_document_named_in_history():
    history = [{"role": "user", "content": "Что в пункте 3.3 СП 55?"}]
    plan = planner.fast_plan("А пункт 3.4?", history)
    assert plan.document_names == ["СП 55"]


@pytest.mark.parametrize(
    "query",
    [
        # No address or no document: the planner must search.
        "Какие требования к озеленению жилых районов?",
        "Что сказано в пункте 5.3?",
        # A topic inside a section is a ranked search the planner phrases.
        "Что в разделе 7 СП 42.13330 об озеленении?",
        # Several documents, editions, amendments, tables, comparisons.
        "Сравни пункт 5.3 СП 55 и СП 42",
        "Что в пункте 5.3 СП 55 в редакции 2016 года?",
        "Пункт 5.3 СП 55 с изменением № 1",
        "Что в таблице 5.3 и пункте 5.3 СП 55?",
        "Пункт 5.3 или 5.4 СП 55?",
        # Which documents exist is a document list, not a lookup.
        "В каких документах есть пункт 5.3 СП 55?",
        # User documents need the project scope rules.
        "Что в пункте 2 моём документе СП 55?",
        # Orchestrator task wording.
        "Что сказано в пункте 5.3 СП 55?\n\nЗадача: найди требования к парковкам",
    ],
)
def test_requests_needing_the_planner(query):
    assert planner.fast_plan(query) is None


async def test_summary_scope_is_used_without_any_model_call(service, fake_llm):
    from tests.unit.test_dvd_structured_retrieval import Pages, run

    service.get_chat_messages.side_effect = RuntimeError("unavailable")
    service.chat_storage_client.get_context.return_value = {
        "content": {"summary": "Обсуждается СП 55.13330.2016, ред. 2016, пункт 3."},
        "updated_through_seq": 4,
    }
    client = Pages(
        [
            dict(
                hits=[dict(id="33", name="СП 55.13330.2016", text="Определение")],
                complete=True,
                total=1,
            )
        ]
    )
    events = await run(service, client, "А что написано в пункте 3.3?")
    assert client.calls[0][1]["document_names"] == ["СП 55.13330.2016"]
    assert client.calls[0][1]["version"] == "2016"
    assert "Полная цитата" in answer_text(events)
    assert fake_llm.chat_calls == []


async def test_fast_path_decision_is_recorded(service, fake_llm):
    from tests.unit.test_dvd_structured_retrieval import Pages

    client = Pages(
        [
            dict(
                hits=[dict(id="r", name="СП 55", text="3.3 Текст.")],
                complete=True,
                total=1,
            )
        ]
    )
    collected = {}
    original = service._run_qa_loop

    async def spy(*args, **kwargs):
        collected.update(metrics=args[5]["metrics"])
        async for event in original(*args, **kwargs):
            yield event

    service._run_qa_loop = spy
    async for _ in service.run_document_qa_pipeline(
        dvd_mcp_client=client,
        token="t",
        model="m",
        temperature=0,
        user_query="Процитируй пункт 3.3 СП 55",
        chat_id="chat-1",
    ):
        pass
    decisions = collected["metrics"].decisions
    assert decisions["planner_bypassed"] is True
    assert decisions["query_type"] == "exact_reference"
    assert "planner" not in collected["metrics"].stages_ms
