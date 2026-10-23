"""Stage 3: chat creation and title generation leave the answer's critical path."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tests.helpers import events_of_type, plan_json, types_of, verdict_json


async def _run(service, mcp, **overrides):
    kwargs = dict(
        dvd_mcp_client=mcp,
        token="tok",
        model="m",
        temperature=0.0,
        user_query="Какие нормы озеленения жилых районов?",
        chat_id=None,
        scenario_id=772,
    )
    kwargs.update(overrides)
    return [event async for event in service.run_document_qa_pipeline(**kwargs)]


async def test_chat_is_created_beside_the_planner(service, fake_llm, fake_mcp):
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Не менее 6 м² на человека [1]."]
    planner_started = asyncio.Event()
    chat = fake_llm.chat

    async def llm(*args, **kwargs):
        planner_started.set()
        return await chat(*args, **kwargs)

    fake_llm.chat = llm

    async def create_chat(*args, **kwargs):
        # The chat is still being created when the planner already runs.
        await asyncio.wait_for(planner_started.wait(), 5)
        return "chat-xyz", kwargs["title"]

    service.create_chat = AsyncMock(side_effect=create_chat)
    service.rename_chat_with_generated_title = AsyncMock(return_value="Нормы")
    events = await _run(service, fake_mcp)

    kwargs = service.create_chat.await_args.kwargs
    assert kwargs["title"] == "Какие нормы озеленения жилых районов?"
    assert kwargs["project_id"] == 4242
    order = types_of(events)
    assert order.index("service_event") < order.index("chunk")
    created = events_of_type(events, "service_event")[0]["content"]["event"]
    assert created["chat_id"] == "chat-xyz"
    request_id = events[0]["content"]["request_id"]
    assert (await service.state_store.get_state(request_id))["chat_id"] == "chat-xyz"
    # The journal a reconnecting client replays has the same order.
    assert await service.state_store.get_buffered_events(request_id) == events
    service._schedule_persist_answer.assert_called_once()
    assert service._schedule_persist_answer.call_args.args[1] == "chat-xyz"
    await asyncio.sleep(0)
    service.rename_chat_with_generated_title.assert_awaited_once()
    assert service.rename_chat_with_generated_title.await_args.args[1] == "chat-xyz"


async def test_project_warning_precedes_chat_and_answer(
    service, fake_llm, fake_mcp, fake_urban
):
    fake_urban.raise_exc = RuntimeError("urban down")
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Ответ [1]."]
    service.rename_chat_with_generated_title = AsyncMock(return_value="Нормы")
    events = await _run(service, fake_mcp)
    order = types_of(events)
    assert order.index("warning") < order.index("service_event") < order.index("chunk")
    assert service.create_chat.await_args.kwargs["project_id"] is None


async def test_answer_does_not_wait_for_a_failing_chat_storage(
    service, fake_llm, fake_mcp
):
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Ответ [1]."]
    service.create_chat = AsyncMock(side_effect=RuntimeError("storage down"))
    service.rename_chat_with_generated_title = AsyncMock()
    events = await _run(service, fake_mcp)
    assert "Ответ [1]." in "".join(
        e["content"]["text"] for e in events_of_type(events, "chunk")
    )
    assert not events_of_type(events, "service_event")
    # Without a chat there is nothing to store the answer in (a no-op call).
    assert service._schedule_persist_answer.call_args.args[1] is None
    service.rename_chat_with_generated_title.assert_not_awaited()


async def test_switch_restores_title_before_answer(
    service, fake_llm, fake_mcp, monkeypatch
):
    monkeypatch.setenv("DVD_DEFERRED_CHAT_SETUP", "false")
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Ответ [1]."]
    service.rename_chat_with_generated_title = AsyncMock()
    await _run(service, fake_mcp)
    assert "title" not in service.create_chat.await_args.kwargs
    service.rename_chat_with_generated_title.assert_not_awaited()


def test_provisional_title_is_the_question_cut_at_a_word(service):
    title = service._provisional_title(
        "Какие требования предъявляются к размещению общеобразовательных школ "
        "в жилых районах?\n\nЗадача: найди нормы"
    )
    assert len(title) <= 61 and title.endswith("…")
    assert title.startswith("Какие требования предъявляются к размещению")
    assert "Задача" not in title
    assert service._provisional_title("  Нормы\nозеленения ") == "Нормы озеленения"


async def test_follow_up_question_is_stored_before_the_answer(service):
    stored = []
    release = asyncio.Event()

    async def add_question(*args, **kwargs):
        await release.wait()
        stored.append("question")

    async def add_answer(*args, **kwargs):
        stored.append("answer")
        return SimpleNamespace(seq=2)

    service.add_single_message = AsyncMock(side_effect=add_question)
    service.add_complex_message = AsyncMock(side_effect=add_answer)
    question = asyncio.create_task(service._persist_question("tok", "c", "q", None))
    answer = asyncio.create_task(
        service._persist_answer(
            "tok", "c", {"final_answer": "a", "question_persisted": question}, None
        )
    )
    await asyncio.sleep(0)
    release.set()
    await answer
    assert stored == ["question", "answer"]


async def test_chat_is_announced_before_a_terminal_error(service, fake_llm, fake_mcp):
    from src.agents.services.dvd.context_reducer import PreparedContext

    fake_llm.json_responses = [plan_json()]
    service.context_reducer.prepare = AsyncMock(
        return_value=PreparedContext("", failed_parts=["round-1/part-1: [1] (x)"])
    )
    service.rename_chat_with_generated_title = AsyncMock(return_value="Нормы")
    events = await _run(service, fake_mcp)
    order = types_of(events)
    assert order.index("service_event") < order.index("error")
    request_id = events[0]["content"]["request_id"]
    assert await service.state_store.get_buffered_events(request_id) == events
