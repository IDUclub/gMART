from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.services.dvd.dialogue import (
    pending_question,
    render_question,
    resolve_reply,
)
from src.agents.services.service_entities.dvd_plan import StructureRetrievalPlan
from tests.helpers import answer_text, plan_json, verdict_json
from tests.unit.test_dvd_filter_scope import sp_candidates
from tests.unit.test_dvd_structured_retrieval import Pages, run


@pytest.mark.parametrize(
    "reply",
    [
        "2",
        "Вариант 2: Раздел 3. Термины и определения",
        "- **Вариант 2:** Раздел 3. Термины и определения",
    ],
)
def test_copied_choice_retains_identity(reply):
    pending = pending_question(
        StructureRetrievalPlan(retrieval_mode="structure", pattern="3"),
        sp_candidates(),
        "Что написано в СП 55 пункт 3?",
    )
    result = resolve_reply(reply, pending)
    assert result["selected_ids"] == ["2"]
    assert result["plan"]["doc_id"] == "sp55"
    assert result["plan"]["pattern"] == "раздел 3"


def test_document_root_is_not_rendered_as_empty_element():
    candidates = sp_candidates()
    for c in candidates:
        c["hierarchy"].insert(0, {"id": "doc", "type": "document"})
    pending = pending_question(
        StructureRetrievalPlan(retrieval_mode="structure", pattern="3"),
        candidates,
        "СП 55",
    )
    assert "Элемент" not in render_question(pending)


async def test_summary_is_read_again_each_turn_and_fresh_tail_is_retained(service):
    old = [
        {"message_id": str(i), "seq": i, "role": "user", "content": str(i)}
        for i in range(1, 16)
    ]
    service.get_chat_messages.return_value = SimpleNamespace(messages=old)
    service.chat_storage_client.get_context.side_effect = [
        {
            "content": {"summary": "Выбран СП 55.13330.2016", "structured": {}},
            "updated_through_seq": 12,
            "tail": [
                {"message_id": "16", "seq": 16, "role": "user", "content": "текущий"}
            ],
        },
        {
            "content": {"summary": "Теперь выбран СП 42", "structured": {}},
            "updated_through_seq": 12,
        },
    ]
    history = await service._load_dialogue_context("tok", "chat-1", "текущий")
    assert "СП 55" in history[0]["content"]
    assert [m["content"] for m in history[1:]] == [str(i) for i in range(6, 16)]
    again = await service._load_dialogue_context("tok", "chat-1", "следующий")
    assert "СП 42" in again[0]["content"]
    assert service.chat_storage_client.get_context.await_count == 2


async def test_uncovered_history_is_not_limited_to_ten_messages(service):
    service.get_chat_messages.return_value = SimpleNamespace(
        messages=[dict(seq=i, role="user", content=str(i)) for i in range(20)]
    )
    service.chat_storage_client.get_context.return_value = {
        "content": {"summary": "СП 55"},
        "updated_through_seq": 2,
        "tail_has_more": True,
    }
    history = await service._load_dialogue_context("tok", "chat-1", "q")
    assert len(history) == 18
    assert history[1]["content"] == "3"


async def test_summary_failure_falls_back_to_history(service):
    service.chat_storage_client.get_context.side_effect = RuntimeError("offline")
    service.get_chat_messages.return_value = SimpleNamespace(
        messages=[dict(role="user", content="СП 55")]
    )
    assert await service._load_dialogue_context("tok", "chat-1", "q") == [
        dict(role="user", content="СП 55")
    ]


async def test_refresh_is_enqueued_after_saved_answer(service):
    service.add_complex_message = AsyncMock(return_value=SimpleNamespace(seq=8))
    service.chat_storage_client.enqueue_context_refresh = AsyncMock()
    await service._persist_answer(
        "tok", "chat-1", {"model": "m", "final_answer": "Ответ"}, 772
    )
    service.chat_storage_client.enqueue_context_refresh.assert_awaited_once_with(
        "tok", "chat-1", target_seq=8, model="m", prompt_version="documents-v1"
    )


async def test_full_pasted_selection_keeps_original_question_and_quote(
    service, fake_llm
):
    question = "Что написано в СП 55 пункт 3?"
    fake_llm.json_responses = [plan_json()]
    first = await run(
        service, Pages([dict(ambiguous=True, candidates=sp_candidates())]), question
    )
    service.get_chat_messages.return_value = SimpleNamespace(
        messages=[
            dict(role="user", content=question),
            dict(role="assistant", content=answer_text(first)),
        ]
    )
    fake_llm.json_responses = [verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Раздел содержит определения [1]."]
    client = Pages(
        [
            dict(
                hits=[
                    dict(
                        id="2",
                        doc_id="sp55",
                        name="СП 55.13330.2016",
                        version="2016",
                        numbering="3",
                        type="section",
                        text="Термины и определения",
                        matched=True,
                    )
                ],
                total=1,
                complete=True,
            )
        ]
    )
    events = await run(service, client, "Вариант 2: Раздел 3. Термины и определения")
    assert client.calls[0][1]["pattern"] == "раздел 3"
    assert client.calls[0][1]["doc_id"] == "sp55"
    assert "Полная цитата" in answer_text(events)
    assert len(fake_llm.chat_calls) == 1  # The full source needs no LLM rewriting.


async def test_previous_quote_keeps_document_but_is_retrieved_again(service):
    service.get_chat_messages.return_value = SimpleNamespace(
        messages=[
            dict(
                role="assistant",
                content="Полная цитата:\n\n[1] СП 55, раздел 3\n\n> Длинный исходный текст",
            )
        ]
    )
    history = await service._load_dialogue_context("tok", "chat-1", "А пункт 3.3?")
    assert "СП 55, раздел 3" in history[0]["content"]
    assert "Длинный исходный текст" not in history[0]["content"]


async def test_summary_restores_filter_when_planner_omits_it(service, fake_llm):
    service.get_chat_messages.side_effect = RuntimeError("unavailable")
    service.chat_storage_client.get_context.return_value = {
        "content": {
            "summary": "Обсуждается СП 55.13330.2016, ред. 2016, пункт 3.",
            "structured": {"verified_facts": ["Также упоминается СП 54"]},
        },
        "updated_through_seq": 4,
    }
    fake_llm.json_responses = [plan_json(document_names=None)]
    client = Pages(
        [
            dict(
                hits=[
                    dict(
                        id="33",
                        doc_id="sp55",
                        name="СП 55.13330.2016",
                        numbering="3.3",
                        text="Определение",
                        matched=True,
                    )
                ],
                complete=True,
                total=1,
            )
        ]
    )
    events = await run(service, client, "А что написано в пункте 3.3?")
    assert client.calls[0][1]["document_names"] == ["СП 55.13330.2016"]
    assert client.calls[0][1]["version"] == "2016"
    assert "Полная цитата" in answer_text(events)
    history = fake_llm.chat_calls[0].messages
    assert sum(m["role"] == "system" for m in history) == 1
    assert any("Сводка предыдущего диалога" in m["content"] for m in history)


async def test_ambiguous_summary_does_not_guess_a_filter(service):
    service.chat_storage_client.get_context.return_value = {
        "content": {"summary": "Сравниваем СП 55 и СП 42"}
    }
    collected = {}
    await service._load_dialogue_context("tok", "chat", "q", collected)
    assert "summary_document_scope" not in collected
