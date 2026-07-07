"""Unit tests for DvdRagService — the iterative retrieve→draft→critique→refine loop,
project_id resolution + warning fallback, reconnect replay/resume, and persistence guard.

All external boundaries are faked (LLM, IDU_DVD MCP, Urban API); the PipelineStateStore runs
against fakeredis so the genuine buffering/checkpoint/replay code paths are exercised.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

from tests.helpers import (
    FakeDvdMcpClient,
    answer_text,
    events_of_type,
    final_chunk,
    plan_json,
    statuses,
    types_of,
    verdict_json,
)


async def _run(service, mcp, **overrides):
    kwargs = dict(
        dvd_mcp_client=mcp,
        token="tok",
        model="m",
        temperature=0.0,
        user_query="вопрос",
        chat_id="chat-1",
    )
    kwargs.update(overrides)
    return [event async for event in service.run_document_qa_pipeline(**kwargs)]


# ---------------------------------------------------------------------------
# Iterative loop behaviour
# ---------------------------------------------------------------------------
class TestLoop:
    async def test_accept_on_first_iteration(self, service, fake_llm, fake_mcp):
        fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
        fake_llm.answer_texts = ["Ответ со ссылкой [1]."]

        events = await _run(service, fake_mcp)

        assert types_of(events)[0] == "pipeline_started"
        assert "tool_call" in types_of(events)
        assert answer_text(events) == "Ответ со ссылкой [1]."
        fc = final_chunk(events)
        assert fc is not None and fc["done"] is True and fc["iteration"] == 1
        assert len(fake_mcp.search_calls) == 1
        service._schedule_persist_answer.assert_called_once()
        collected = service._schedule_persist_answer.call_args.args[2]
        assert collected["final_answer"] == "Ответ со ссылкой [1]."
        assert collected["newly_completed"] is True

    async def test_reject_then_requery_then_accept(self, service, fake_llm, fake_mcp):
        fake_llm.json_responses = [
            plan_json(search_query="первый"),
            verdict_json(
                satisfied=False, critique="нет пункта", refined_search_query="второй"
            ),
            plan_json(search_query="второй-план"),
            verdict_json(satisfied=True),
        ]
        fake_llm.answer_texts = ["Черновик 1", "Черновик 2 [1]"]

        events = await _run(service, fake_mcp)

        assert len(fake_mcp.search_calls) == 2
        assert fake_mcp.search_calls[1].query == "второй-план"
        # the rejection surfaced as a self_review status carrying the critique
        assert any("нет пункта" in t for t in statuses(events, "self_review"))
        # the refined query was fed to the second planning round
        second_plan_prompt = [c for c in fake_llm.chat_calls if not c.stream][
            2
        ].messages[0]["content"]
        assert "второй" in second_plan_prompt and "нет пункта" in second_plan_prompt
        # drafts are tagged with their iteration
        draft_iters = sorted(
            {
                e["content"]["iteration"]
                for e in events
                if e["type"] == "chunk" and e["content"]["text"]
            }
        )
        assert draft_iters == [1, 2]
        collected = service._schedule_persist_answer.call_args.args[2]
        assert collected["final_answer"] == "Черновик 2 [1]"

    async def test_max_iterations_accepts_last_without_critic(
        self, service, fake_llm, fake_mcp
    ):
        fake_llm.json_responses = [
            plan_json(),
            verdict_json(satisfied=False, critique="ещё", refined_search_query="r1"),
            plan_json(),
            verdict_json(satisfied=False, critique="ещё", refined_search_query="r2"),
            plan_json(),  # 3rd iteration: no critic call, accepted unconditionally
        ]
        fake_llm.answer_texts = ["d1", "d2", "d3"]

        events = await _run(service, fake_mcp)

        assert len(fake_mcp.search_calls) == 3
        assert final_chunk(events)["iteration"] == 3
        collected = service._schedule_persist_answer.call_args.args[2]
        assert collected["final_answer"] == "d3"
        # 3 plans + 2 verdicts = 5 non-stream LLM calls (critic skipped on the last round)
        non_stream = [c for c in fake_llm.chat_calls if not c.stream]
        assert len(non_stream) == 5

    async def test_no_hits_triggers_requery(self, service, fake_llm):
        mcp = FakeDvdMcpClient(hits_per_call=[[], [{"name": "A", "text": "норма"}]])
        fake_llm.json_responses = [
            plan_json(search_query="q1"),
            plan_json(search_query="q2"),
            verdict_json(satisfied=True),
        ]
        fake_llm.answer_texts = ["Ответ [1]"]

        events = await _run(service, mcp)

        assert len(mcp.search_calls) == 2
        assert any("не найдено" in t.lower() for t in statuses(events, "searching"))
        collected = service._schedule_persist_answer.call_args.args[2]
        assert collected["final_answer"] == "Ответ [1]"

    async def test_planner_receives_context_height_from_llm(
        self, service, fake_llm, fake_mcp
    ):
        fake_llm.json_responses = [
            plan_json(kind="table", limit=3, context_height=4),
            verdict_json(satisfied=True),
        ]
        fake_llm.answer_texts = ["Ответ"]

        await _run(service, fake_mcp)

        call = fake_mcp.search_calls[0]
        assert call.kind == "table"
        assert call.limit == 3
        assert call.context_height == 4


# ---------------------------------------------------------------------------
# project_id resolution + warning fallback
# ---------------------------------------------------------------------------
class TestProjectId:
    async def test_resolved_project_id_passed_to_create_chat(
        self, service, fake_llm, fake_mcp, fake_urban
    ):
        fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
        fake_llm.answer_texts = ["Ответ"]

        events = await _run(service, fake_mcp, chat_id=None, scenario_id=772)

        assert fake_urban.calls == [("tok", 772)]
        assert not events_of_type(events, "warning")
        kwargs = service.create_chat.await_args.kwargs
        assert kwargs["scenario_id"] == 772
        assert kwargs["project_id"] == 4242
        assert kwargs["resolve_project_id"] is False

    async def test_failed_lookup_warns_and_still_creates_chat(
        self, service, fake_llm, fake_mcp, fake_urban
    ):
        fake_urban.raise_exc = RuntimeError("urban api down")
        fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
        fake_llm.answer_texts = ["Ответ"]

        events = await _run(service, fake_mcp, chat_id=None, scenario_id=772)

        warnings = events_of_type(events, "warning")
        assert len(warnings) == 1
        assert warnings[0]["content"]["code"] == "project_id_unavailable"
        assert warnings[0]["content"]["scenario_id"] == 772
        # chat still created — with scenario_id but no project_id, and no re-resolution
        kwargs = service.create_chat.await_args.kwargs
        assert kwargs["scenario_id"] == 772
        assert kwargs["project_id"] is None
        assert kwargs["resolve_project_id"] is False
        # the request still completed
        assert answer_text(events) == "Ответ"
        # warning is emitted before the chat_created event
        order = types_of(events)
        assert order.index("warning") < order.index("service_event")

    async def test_no_scenario_id_skips_lookup(
        self, service, fake_llm, fake_mcp, fake_urban
    ):
        fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
        fake_llm.answer_texts = ["Ответ"]

        events = await _run(service, fake_mcp, chat_id=None, scenario_id=None)

        assert fake_urban.calls == []
        assert not events_of_type(events, "warning")
        assert service.create_chat.await_args.kwargs["project_id"] is None


# ---------------------------------------------------------------------------
# Reconnect (request_id): replay buffered events + resume from checkpoint
# ---------------------------------------------------------------------------
class TestReconnect:
    async def test_completed_run_only_replays_and_does_not_repersist(
        self, service, fake_llm, fake_mcp, state_store
    ):
        rid = "11111111-1111-1111-1111-111111111111"
        await state_store.create(
            rid,
            chat_id="chat-1",
            user_query="q",
            scenario_id=None,
            model="m",
            temperature=0.0,
        )
        buffered = [
            service._pipeline_started_event(rid),
            service._status("answer_drafting", "Формирую ответ…"),
            service._chunk("Готовый ответ", done=False, iteration=1),
            service._status("finalizing", "Ответ сформирован"),
            service._chunk("", done=True, iteration=1),
        ]
        for event in buffered:
            await state_store.buffer_event(rid, event)
        await state_store.save_checkpoint(
            rid,
            "qa_progress",
            {
                "completed_iterations": 1,
                "tool_calls": [],
                "accepted": True,
                "final_answer": "Готовый ответ",
                "final_iteration": 1,
                "prev_critique": None,
                "prev_query": None,
            },
        )

        events = await _run(service, fake_mcp, chat_id=None, request_id=rid)

        # replayed everything, did no new search / LLM work
        assert len(fake_mcp.search_calls) == 0
        assert fake_llm.chat_calls == []
        assert final_chunk(events) is not None
        # a completed reconnect must not write the answer to ChatStorage again
        service._schedule_persist_answer.assert_not_called()

    async def test_interrupted_run_resumes_from_checkpoint(
        self, service, fake_llm, fake_mcp, state_store
    ):
        rid = "22222222-2222-2222-2222-222222222222"
        await state_store.create(
            rid,
            chat_id="chat-1",
            user_query="q",
            scenario_id=None,
            model="m",
            temperature=0.0,
        )
        await state_store.buffer_event(rid, service._pipeline_started_event(rid))
        await state_store.buffer_event(
            rid, service._chunk("Черновик 1", done=False, iteration=1)
        )
        await state_store.save_checkpoint(
            rid,
            "qa_progress",
            {
                "completed_iterations": 1,
                "tool_calls": [{"function": {"name": "search_all", "arguments": {}}}],
                "accepted": False,
                "final_answer": None,
                "final_iteration": None,
                "prev_critique": "нужно точнее",
                "prev_query": "первый запрос",
            },
        )
        fake_llm.json_responses = [
            plan_json(search_query="второй"),
            verdict_json(satisfied=True),
        ]
        fake_llm.answer_texts = ["Черновик 2 финал"]

        events = await _run(service, fake_mcp, chat_id=None, request_id=rid)

        # iteration 1 is NOT redone — only iteration 2 runs a new search
        assert len(fake_mcp.search_calls) == 1
        assert fake_mcp.search_calls[0].query == "второй"
        # the buffered iteration-1 draft chunk is present from the replay
        assert any(
            e["type"] == "chunk" and e["content"]["iteration"] == 1 for e in events
        )
        service._schedule_persist_answer.assert_called_once()
        collected = service._schedule_persist_answer.call_args.args[2]
        assert collected["final_answer"] == "Черновик 2 финал"
        # tool_calls = 1 restored from checkpoint + 1 from the resumed search
        assert len(collected["tool_calls"]) == 2

    async def test_unknown_request_id_starts_fresh(self, service, fake_llm, fake_mcp):
        fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
        fake_llm.answer_texts = ["Ответ"]

        events = await _run(
            service, fake_mcp, request_id="33333333-3333-3333-3333-333333333333"
        )

        # no prior state for that id → a normal fresh run
        assert types_of(events)[0] == "pipeline_started"
        assert len(fake_mcp.search_calls) == 1


# ---------------------------------------------------------------------------
# User-question persistence (follow-up questions in an existing chat)
# ---------------------------------------------------------------------------
class TestUserQuestionPersistence:
    async def test_follow_up_question_saved_to_existing_chat(
        self, service, fake_llm, fake_mcp
    ):
        from src.agents.api_clients.chat_storage_client.entities import RoleEnum

        fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
        fake_llm.answer_texts = ["Ответ"]

        await _run(service, fake_mcp, chat_id="chat-1")

        service.add_single_message.assert_awaited_once()
        token, chat_id, role, text = service.add_single_message.await_args.args
        assert chat_id == "chat-1"
        assert role == RoleEnum.USER
        assert text == "вопрос"

    async def test_new_chat_does_not_double_save_first_question(
        self, service, fake_llm, fake_mcp
    ):
        # create_chat itself stores the first question — the pipeline must not add a copy
        fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
        fake_llm.answer_texts = ["Ответ"]

        await _run(service, fake_mcp, chat_id=None)

        service.create_chat.assert_awaited_once()
        service.add_single_message.assert_not_awaited()

    async def test_reconnect_does_not_resave_question(
        self, service, fake_llm, fake_mcp, state_store
    ):
        rid = "44444444-4444-4444-4444-444444444444"
        await state_store.create(
            rid,
            chat_id="chat-1",
            user_query="вопрос",
            scenario_id=None,
            model="m",
            temperature=0.0,
        )
        await state_store.save_checkpoint(
            rid,
            "qa_progress",
            {
                "completed_iterations": 1,
                "tool_calls": [],
                "accepted": True,
                "final_answer": "Готовый ответ",
                "final_iteration": 1,
                "prev_critique": None,
                "prev_query": None,
            },
        )

        await _run(service, fake_mcp, chat_id="chat-1", request_id=rid)

        service.add_single_message.assert_not_awaited()

    async def test_persist_failure_does_not_break_stream(
        self, service, fake_llm, fake_mcp
    ):
        service.add_single_message.side_effect = RuntimeError("chat storage down")
        fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
        fake_llm.answer_texts = ["Ответ"]

        events = await _run(service, fake_mcp, chat_id="chat-1")

        assert answer_text(events) == "Ответ"
        assert final_chunk(events)["done"] is True


# ---------------------------------------------------------------------------
# persist_history=False (A2A runs): no ChatStorage writes at all
# ---------------------------------------------------------------------------
class TestPersistHistoryDisabled:
    async def test_no_writes_with_existing_chat_but_history_still_read(
        self, service, fake_llm, fake_mcp
    ):
        fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
        fake_llm.answer_texts = ["Ответ"]

        events = await _run(service, fake_mcp, chat_id="chat-1", persist_history=False)

        service.create_chat.assert_not_awaited()
        service.add_single_message.assert_not_awaited()
        service._schedule_persist_answer.assert_not_called()
        # chat_id is still used for read-only LLM context
        service.get_chat_messages.assert_awaited_once()
        assert answer_text(events) == "Ответ"

    async def test_no_chat_created_without_chat_id(self, service, fake_llm, fake_mcp):
        fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
        fake_llm.answer_texts = ["Ответ"]

        events = await _run(service, fake_mcp, chat_id=None, persist_history=False)

        service.create_chat.assert_not_awaited()
        service.add_single_message.assert_not_awaited()
        service._schedule_persist_answer.assert_not_called()
        # no chat_created service_event is emitted
        assert not events_of_type(events, "service_event")
        assert answer_text(events) == "Ответ"


# ---------------------------------------------------------------------------
# Persistence (parts building)
# ---------------------------------------------------------------------------
async def test_persist_answer_builds_toolcall_and_text_parts(
    monkeypatch, fake_llm, fake_urban, state_store
):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.AsyncOllamaClient",
        lambda *a, **k: fake_llm,
    )
    from src.agents.services.dvd_rag_service import DvdRagService

    svc = DvdRagService("http://x", Mock(), fake_urban, state_store)
    svc.add_complex_message = AsyncMock()

    collected = {
        "final_answer": "Итоговый ответ [1]",
        "tool_calls": [
            {
                "function": {
                    "name": "search_all",
                    "arguments": {"query": "q", "limit": 5},
                }
            }
        ],
    }
    await svc._persist_answer("tok", "chat-1", collected, scenario_id=772)

    svc.add_complex_message.assert_awaited_once()
    call = svc.add_complex_message.await_args
    parts = call.args[3]
    assert [p.kind for p in parts] == ["tool_call", "text"]
    assert parts[1].payload.text == "Итоговый ответ [1]"
    assert call.kwargs.get("scenario_id") == 772
