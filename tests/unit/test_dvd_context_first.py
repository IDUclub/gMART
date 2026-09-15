import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.schema.dvd_response import DvdStatusResponse
from src.agents.services.dvd.conversation_evidence import (
    ConversationEvidence,
    compact_hits,
    recover_quotation,
    source_context,
)
from src.agents.services.dvd.dialogue import pending_question, render_question
from src.agents.services.dvd.dvd_context import DvdContextBuilder
from src.agents.services.service_entities.dvd_plan import StructureRetrievalPlan
from tests.helpers import answer_text, plan_json, verdict_json
from tests.unit.test_dvd_filter_scope import sp_candidates
from tests.unit.test_dvd_structured_retrieval import Pages, run

NAME = "СП 55.13330.2016"
HITS = [
    dict(
        id="section",
        doc_id="sp55",
        name=NAME,
        version="2016",
        type="section",
        numbering="3",
        text="3 Термины и определения",
        matched=True,
        order=1,
        structure_path=["раздел 3"],
    ),
    dict(
        id="definition",
        doc_id="sp55",
        name=NAME,
        version="2016",
        type="clause",
        numbering="3.3",
        text="Блокированная застройка — дома с общей стеной без проемов.",
        matched=False,
        matched_ancestor_ids=["section"],
        parent_id="section",
        order=2,
        structure_path=["раздел 3", "3.3"],
    ),
]
PLAN = dict(
    retrieval_mode="structure",
    pattern="раздел 3",
    doc_id="sp55",
    document_names=[NAME],
    version="2016",
)


def decision(
    action="answer",
    answer="Раздел определяет термины, в том числе блокированную застройку [1] [2].",
    sources=None,
):
    return json.dumps(
        dict(
            action=action,
            answer=answer,
            source_numbers=[1, 2] if sources is None else sources,
        ),
        ensure_ascii=False,
    )


def snapshot(scenario_id=None):
    return dict(
        hits=compact_hits(HITS),
        plan=PLAN,
        question="Что написано в СП 55 пункт 3?",
        scenario_id=scenario_id,
        complete=True,
    )


async def seed(service):
    await service.state_store.set_document_evidence("chat-1", snapshot())
    await service.state_store.set_document_scope(
        "chat-1", dict(doc_id="sp55", document_names=[NAME], version="2016")
    )


async def test_live_dialogue_shape_answers_brief_followup_without_dvd(
    service, fake_llm
):
    fake_llm.json_responses = [plan_json()]
    first = await run(
        service,
        Pages([dict(ambiguous=True, candidates=sp_candidates())]),
        "Что написано в СП 55 пункт 3?",
    )
    history = [
        dict(role="user", content="Что написано в СП 55 пункт 3?"),
        dict(role="assistant", content=answer_text(first)),
    ]
    service.get_chat_messages.return_value = SimpleNamespace(messages=history)
    second = await run(
        service,
        Pages([dict(hits=HITS, total=2, complete=True)]),
        "Вариант 2: Раздел 3. Термины и определения",
    )
    history += [
        dict(role="user", content="Вариант 2: Раздел 3. Термины и определения"),
        dict(role="assistant", content=answer_text(second)),
    ]
    service.get_chat_messages.return_value = SimpleNamespace(messages=history)
    fake_llm.json_responses = [decision(), verdict_json(satisfied=True)]
    no_search = Pages([])
    third = await run(service, no_search, "Расскажи вкраце о чём этот пункт")
    assert "Раздел определяет" in answer_text(third)
    assert not no_search.calls and not any(e["type"] == "tool_call" for e in third)
    assert HITS[1]["text"] in fake_llm.chat_calls[-2].messages[0]["content"]
    assert any(e.get("content", {}).get("status") == "context_check" for e in third)
    # Rephrasing does not replace the source memory with the model's paraphrase.
    assert (await service.state_store.get_document_evidence("chat-1"))["hits"][1][
        "text"
    ] == HITS[1]["text"]


async def test_missing_context_runs_retrieval_after_assessment(service, fake_llm):
    await seed(service)
    fake_llm.json_responses = [
        json.dumps(
            dict(retrieval_mode="structure", pattern="10.6", document_names=[NAME])
        )
    ]
    client = Pages(
        [
            dict(
                hits=[
                    {**HITS[1], "numbering": "10.6", "text": "Меры экономии энергии."}
                ],
                complete=True,
                total=1,
            )
        ]
    )
    events = await run(service, client, "Что написано в пункте 10.6?")
    assert client.calls[0][1]["pattern"] == "10.6"
    assert "Меры экономии" in answer_text(events)


async def test_rejected_context_answer_is_not_emitted_and_fetches_selected_section(
    service, fake_llm
):
    await seed(service)
    fake_llm.json_responses = [
        decision(answer="Неверный черновик [1]."),
        verdict_json(satisfied=False),
        decision("search", "", []),
        json.dumps(PLAN),
        verdict_json(satisfied=True),
    ]
    fake_llm.answer_texts = ["Раздел определяет термины [1]."]
    client = Pages([dict(hits=HITS, complete=True, total=2)])
    events = await run(service, client, "Объясни этот пункт")
    assert client.calls[0][1]["pattern"] == "раздел 3"
    assert "Неверный" not in answer_text(events)
    assert "Раздел определяет" in answer_text(events)


async def test_unresolved_clarification_precedes_old_evidence(service, fake_llm):
    await seed(service)
    pending = pending_question(
        StructureRetrievalPlan(**PLAN), sp_candidates(), "Что написано в СП 55 пункт 3?"
    )
    await service.state_store.set_document_question("chat-1", pending)
    service.get_chat_messages.return_value = SimpleNamespace(
        messages=[dict(role="assistant", content=render_question(pending))]
    )
    no_search = Pages([])
    result = await run(service, no_search, "Расскажи вкратце об этом пункте")
    assert "Вариант 1" in answer_text(result) and "Вариант 2" in answer_text(result)
    assert not fake_llm.chat_calls and not no_search.calls
    assert await service.state_store.get_document_question("chat-1") == pending


async def test_context_ambiguity_uses_grouped_unique_choices(service, fake_llm):
    evidence = snapshot()
    evidence["hits"] = [
        {**HITS[1], "id": "a", "matched": True},
        {
            **HITS[1],
            "id": "b",
            "doc_id": "sp42",
            "name": "СП 42.13330.2016",
            "matched": True,
        },
    ]
    await service.state_store.set_document_evidence("chat-1", evidence)
    fake_llm.json_responses = [decision("clarify", "", [1, 2])]
    no_search = Pages([])
    events = await run(service, no_search, "Объясни этот пункт")
    assert NAME in answer_text(events) and "СП 42" in answer_text(events)
    assert (
        len((await service.state_store.get_document_question("chat-1"))["options"]) == 2
    )
    assert not no_search.calls


@pytest.mark.parametrize(
    "query,scenario",
    [
        ("Что в СП 42 пункт 3?", None),
        ("Обнови данные", None),
        ("Новый вопрос", None),
        ("Редакция 2020", None),
        ("Объясни", 772),
    ],
)
def test_source_memory_never_crosses_explicit_scope_or_freshness(query, scenario):
    assert not ConversationEvidence.applicable(query, snapshot(), scenario)


async def test_sources_are_not_used_without_chat_access(service, fake_llm):
    await seed(service)
    service.chat_storage_client.get_context.side_effect = RuntimeError("forbidden")
    service.get_chat_messages.side_effect = RuntimeError("forbidden")
    service.conversation_evidence.assess = AsyncMock()
    fake_llm.json_responses = [json.dumps(PLAN)]
    await run(
        service,
        Pages([dict(hits=HITS, total=2, complete=True)]),
        "Что написано в разделе 3 СП 55?",
    )
    service.conversation_evidence.assess.assert_not_called()


async def test_quotation_in_existing_chat_works_without_redis_evidence(
    service, fake_llm
):
    text = DvdContextBuilder().full_quote(HITS)
    service.get_chat_messages.return_value = SimpleNamespace(
        messages=[dict(role="assistant", content=text)], scenario_id=None
    )
    fake_llm.json_responses = [decision(), verdict_json(satisfied=True)]
    no_search = Pages([])
    events = await run(service, no_search, "Расскажи вкратце")
    assert not no_search.calls and "Раздел определяет" in answer_text(events)


def test_recovered_quote_preserves_bodies_and_not_assistant_interpretation():
    text = "Неподтвержденное пояснение.\n\n" + DvdContextBuilder().full_quote(HITS)
    recovered = recover_quotation([dict(role="assistant", content=text)], 772)
    assert [h["text"] for h in recovered["hits"]] == [h["text"] for h in HITS]
    assert recovered["plan"]["pattern"] == "раздел 3"
    assert "Неподтвержденное" not in source_context(recovered["hits"])
    assert (
        recover_quotation(
            [
                dict(role="assistant", content=text),
                dict(role="assistant", content="Другой вопрос не решён"),
            ],
            772,
        )
        is None
    )


async def test_invalid_context_decision_falls_back_to_targeted_dvd(service, fake_llm):
    await seed(service)
    fake_llm.json_responses = ["bad json", verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Раздел определяет термины [1]."]
    events = await run(
        service,
        Pages([dict(hits=HITS, total=2, complete=True)]),
        "Объясни этот пункт",
    )
    assert "Раздел определяет" in answer_text(events)


def test_context_stage_is_part_of_public_sse_contract():
    assert (
        DvdStatusResponse(status="context_check", text="Проверяю контекст").status
        == "context_check"
    )


async def test_context_model_failure_falls_back_without_losing_turn(service, fake_llm):
    from src.agents.model_clients.llm_base import LlmResponseError

    await seed(service)
    service.conversation_evidence.assess = AsyncMock(
        side_effect=LlmResponseError("Incomplete structured answer", 502)
    )
    client = Pages([dict(hits=HITS, total=2, complete=True)])
    fake_llm.json_responses = [verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Раздел определяет термины [1]."]
    events = await run(service, client, "Объясни этот пункт")
    assert client.calls[0][1]["pattern"] == "раздел 3"
    assert "Раздел определяет" in answer_text(events)
    assert not any(e["type"] == "error" for e in events)


async def test_missing_topic_assessment_searches_within_document(service, fake_llm):
    await seed(service)
    fake_llm.json_responses = [
        decision("search", "", []),
        json.dumps(
            dict(retrieval_mode="structure", pattern="10.6", document_names=[NAME])
        ),
    ]
    client = Pages(
        [
            dict(
                hits=[
                    {**HITS[1], "numbering": "10.6", "text": "Меры экономии энергии."}
                ],
                complete=True,
                total=1,
            )
        ]
    )
    events = await run(service, client, "Что написано о мерах экономии энергии?")
    assert client.calls[0][1]["pattern"] == "10.6"
    assert "Меры экономии" in answer_text(events)


async def test_context_draft_is_repaired_before_another_search(service, fake_llm):
    await seed(service)
    fake_llm.json_responses = [
        decision(answer="Неверный черновик [1]."),
        verdict_json(satisfied=False),
        decision(),
        verdict_json(satisfied=True),
    ]
    no_search = Pages([])
    events = await run(service, no_search, "Объясни этот пункт")
    assert "Неверный" not in answer_text(events)
    assert "Раздел определяет" in answer_text(events)
    assert not no_search.calls
    assert "Неверный черновик" in fake_llm.chat_calls[2].messages[-1]["content"]


async def test_cached_clause_quote_preserves_raw_text_and_active_target(
    service, fake_llm
):
    await seed(service)
    no_search = Pages([])
    events = await run(service, no_search, "Что написано в пункте 3.3?")
    assert HITS[1]["text"] in answer_text(events)
    assert "Полная цитата" in answer_text(events)
    assert not no_search.calls and not fake_llm.chat_calls
    stored = await service.state_store.get_document_evidence("chat-1")
    assert stored["plan"]["pattern"] == "3.3"
    # Keep the larger source set available for subsequent questions.
    assert len(stored["hits"]) == 2
