"""Exercise retries at the real retrieval/draft/review and SSE producer seams."""

import json
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from src.agents.services.dvd.runs import stream_document_run
from tests.helpers import plan_json, verdict_json
from tests.unit.test_dvd_rag_service import _run


async def test_repeated_plan_reuses_sources_and_prepared_context(
    service, fake_llm, fake_mcp
):
    prepare = service.context_reducer.prepare
    service.context_reducer.prepare = AsyncMock(wraps=prepare)
    fake_llm.json_responses = [
        plan_json(search_query="школы"),
        verdict_json(satisfied=False, critique="PRIVATE_FIRST"),
        plan_json(search_query="  школы  "),
        verdict_json(satisfied=False, critique="PRIVATE_SECOND"),
        plan_json(search_query="школы"),
        verdict_json(satisfied=True),
    ]
    fake_llm.answer_texts = ["d1", "d2", "Исправленный ответ [1]"]

    events = await _run(service, fake_mcp)

    assert len(fake_mcp.search_calls) == 1
    assert len([e for e in events if e["type"] == "tool_call"]) == 1
    # Prepare sources once, then prepare each of the three distinct reviews.
    assert service.context_reducer.prepare.await_count == 4
    assert "PRIVATE_" not in json.dumps(events)
    drafts = [c for c in fake_llm.chat_calls if c.stream]
    assert "PRIVATE_FIRST" in drafts[2].messages[0]["content"]
    assert "PRIVATE_SECOND" in drafts[2].messages[0]["content"]
    assert events[-1]["content"]["done"]


async def test_repeated_plan_uses_critic_query_without_changing_scope(
    service, fake_llm, fake_mcp
):
    fake_llm.json_responses = [
        plan_json(search_query="школы", block="main"),
        verdict_json(
            satisfied=False,
            critique="Нужны источники",
            refined_search_query="школы расстояния",
        ),
        plan_json(search_query="школы", block="main"),
        verdict_json(satisfied=True),
    ]
    fake_llm.answer_texts = ["d1", "Ответ [1]"]
    await _run(service, fake_mcp)
    assert [c.query for c in fake_mcp.search_calls] == ["школы", "школы расстояния"]
    assert all(c.block == "main" for c in fake_mcp.search_calls)


async def test_changed_search_parameters_fetch_new_sources(service, fake_llm, fake_mcp):
    fake_llm.json_responses = [
        plan_json(limit=2),
        verdict_json(satisfied=False, critique="Нужен контекст"),
        plan_json(limit=5),
        verdict_json(satisfied=True),
    ]
    fake_llm.answer_texts = ["d1", "Ответ [1]"]
    await _run(service, fake_mcp)
    assert [c.limit for c in fake_mcp.search_calls] == [2, 5]


async def test_search_cache_is_not_shared_between_requests(service, fake_llm, fake_mcp):
    for _ in range(2):
        fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
        fake_llm.answer_texts = ["Ответ [1]"]
        await _run(service, fake_mcp, chat_id=None, persist_history=False)
    assert len(fake_mcp.search_calls) == 2


async def test_exact_retrieval_ignores_changed_ranking_query(service, fake_llm):
    from tests.unit.test_dvd_structured_retrieval import Pages

    scope = dict(
        retrieval_mode="structure", pattern="3.3", document_names=["СП 2.13130.2020"]
    )
    fake_llm.json_responses = [
        json.dumps({**scope, "search_query": "q1"}),
        verdict_json(
            satisfied=False, critique="Исправь формулировку", refined_search_query="q2"
        ),
        json.dumps({**scope, "search_query": "q2", "limit": 20}),
        verdict_json(satisfied=True),
    ]
    fake_llm.answer_texts = ["d1", "Ответ [1]"]
    client = Pages(
        [
            dict(
                hits=[
                    dict(
                        id="clause",
                        name="СП 2.13130.2020",
                        text="Текст пункта",
                        structure_path=["3.3"],
                    )
                ],
                total=1,
                complete=True,
            )
        ]
    )
    events = await _run(service, client)
    assert len(client.calls) == 1
    assert len([e for e in events if e["type"] == "tool_call"]) == 1


async def test_empty_repeated_search_is_not_reexecuted(service, fake_llm):
    from tests.helpers import FakeDvdMcpClient

    fake_llm.json_responses = [plan_json()] * 3
    client = FakeDvdMcpClient(default_hits=[])
    events = await _run(service, client)
    assert len(client.search_calls) == 1
    assert events[-1]["content"]["done"]
    assert "не найдены" in events[-1]["content"]["text"]


async def test_invalid_first_review_stops_without_new_retrieval(
    service, fake_llm, fake_mcp
):
    fake_llm.json_responses = [plan_json(), "bad json", "bad json", "bad json"]
    fake_llm.answer_texts = ["d1"]
    events = [
        e
        async for e in stream_document_run(
            service,
            model="m",
            dvd_mcp_client=fake_mcp,
            token=None,
            user_query="вопрос",
            temperature=0,
            persist_history=False,
        )
    ]
    assert len(fake_mcp.search_calls) == 1
    assert events[-1]["type"] == "error"
    assert not any(e["type"] == "chunk" for e in events)


@pytest.mark.parametrize("outcome", ["rejected", "invalid_json", "technical_error"])
async def test_terminal_reason_is_logged_but_not_exposed(
    service, fake_llm, fake_mcp, outcome
):
    fake_llm.json_responses = [
        plan_json(),
        verdict_json(satisfied=False, critique="PRIVATE_FIRST"),
        plan_json(),
        verdict_json(satisfied=False, critique="PRIVATE_SECOND"),
        plan_json(),
    ]
    fake_llm.json_responses += (
        [verdict_json(satisfied=False, critique="PRIVATE_LAST")]
        if outcome == "rejected"
        else ["not json"] * 3
    )
    fake_llm.answer_texts = ["d1", "d2", "d3"]
    original = fake_llm.chat

    async def chat(*args, **kwargs):
        if outcome == "technical_error" and len(fake_llm.chat_calls) == 8:
            raise RuntimeError("PRIVATE_TRANSPORT_ERROR")
        return await original(*args, **kwargs)

    fake_llm.chat = chat
    logs = []
    sink = logger.add(lambda message: logs.append(str(message)))
    try:
        events = [
            e
            async for e in stream_document_run(
                service,
                model="m",
                dvd_mcp_client=fake_mcp,
                token=None,
                user_query="вопрос",
                temperature=0,
                persist_history=False,
            )
        ]
    finally:
        logger.remove(sink)
    request_id = events[0]["content"]["request_id"]
    assert events[-1]["type"] == "error"
    assert events[-1]["content"] == {
        "traceback": "",
        "message": "Не удалось завершить запрос. Повторите попытку.",
    }
    assert not any(e["type"] == "chunk" for e in events)
    assert "PRIVATE_" not in json.dumps(events)
    assert (await service.state_store.get_state(request_id))["status"] == "failed"
    service._schedule_persist_answer.assert_not_called()
    text = "\n".join(logs)
    if outcome == "rejected":
        assert any(
            request_id in line and "PRIVATE_LAST" in line and "iteration=3" in line
            for line in logs
        )
        assert "reason=review_exhausted" in text
    elif outcome == "invalid_json":
        assert "reason=critic_invalid_response" in text
    else:
        assert "PRIVATE_TRANSPORT_ERROR" in text
        assert any(
            request_id in line and "stage=self_review" in line and "iteration=3" in line
            for line in logs
        )
