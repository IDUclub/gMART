"""Deterministic planner paths and speculative semantic retrieval."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.agents.services.dvd import flags
from src.agents.services.dvd.dvd_reasoning import RetrievalPlanner, is_simple_question
from src.agents.services.dvd.run_metrics import DvdRunMetrics
from src.agents.services.service_entities.dvd_plan import (
    SemanticRetrievalPlan,
    validate_retrieval_plan,
)
from tests.helpers import (
    FakeDvdMcpClient,
    answer_text,
    events_of_type,
    plan_json,
    verdict_json,
)

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
    assert decisions["planner_path"] == "exact"
    assert decisions["query_type"] == "exact_reference"
    assert "planner" not in collected["metrics"].stages_ms


@pytest.mark.parametrize(
    "query",
    [
        "Какое минимальное расстояние от школы до жилых зданий?",
        "Как рассчитать площадь озеленения жилого квартала?",
        "Расстояние до школы?",
        "Парковки?",
    ],
)
def test_simple_standalone_question(query):
    assert is_simple_question(query)
    plan = planner.simple_plan(query)
    assert plan.search_query == query
    assert plan.retrieval_mode == "semantic"
    defaults = SemanticRetrievalPlan(retrieval_mode="semantic", search_query=query)
    assert plan == defaults


@pytest.mark.parametrize(
    "query",
    [
        "Что требует СП 42.13330.2016?",
        "Что требует Градостроительный кодекс?",
        "Требования редакции 2016 года к школам?",
        "Какие изменения требований к школам?",
        "Какие действующие требования к размещению школ?",
        "Какой акт утратил силу?",
        "Какие размеры указаны в таблице?",
        "Какое расстояние показано на рисунке?",
        "Требования приложения к размещению школ?",
        "Найди раздел об озеленении",
        "Найди главу про школы",
        "Что в пункте 5.3?",
        "Сравни требования документов о школах",
        "В каких документах есть требования к школам?",
        "Приведи полную цитату о школах",
        "Как размещать школы?\n\nЗадача: найди требования",
        "Как размещать школы? Как размещать дома?",
        "Требования к размещению школ " * 12,
        "Школа или детский сад?",
        "Мой документ, редакция 2020: 5.3",
    ],
)
def test_simple_question_exclusions(query):
    assert not is_simple_question(query)
    assert planner.simple_plan(query) is None


@pytest.mark.parametrize(
    "reference",
    [
        "он",
        "она",
        "оно",
        "они",
        "его",
        "её",
        "их",
        "этот",
        "эта",
        "это",
        "этом",
        "там",
        "тут",
        "такой",
        "тот же",
        "в нём",
        "а для",
        "а если",
        "а в",
        "и ещё",
        "а требуется",
        "и требуется",
    ],
)
def test_simple_question_with_history_reference(reference):
    history = [{"role": "assistant", "content": "Обсуждалось размещение школы."}]
    query = f"{reference} минимальное расстояние от школы до жилых зданий?"
    assert planner.simple_plan(query, history) is None


def test_short_question_only_needs_planner_with_history():
    query = "Расстояние до школы?"
    assert planner.simple_plan(query) is not None
    assert planner.simple_plan(query, [{"role": "user", "content": "Школы"}]) is None


def test_history_only_blocks_dependent_questions():
    query = "Какие требования к школе, если она рядом с домом?"
    assert planner.simple_plan(query) is not None
    history = [{"role": "user", "content": "Требования к жилым зданиям"}]
    assert planner.simple_plan(query, history) is None
    independent = "Минимальная высота этажа школьного здания?"
    assert planner.simple_plan(independent, history) is not None


def test_simple_plan_clamps_and_applies_conversation_scope(monkeypatch):
    query = "Минимальное расстояние от школы до жилых зданий?"
    history = [{"role": "user", "content": "Обсуждаем СП 42.13330.2016"}]
    clamp = planner._clamp
    calls = []

    def spy(*args, **kwargs):
        calls.append(args)
        return clamp(*args, **kwargs)

    monkeypatch.setattr(planner, "_clamp", spy)
    plan = planner.simple_plan(query, history)
    assert calls and plan.document_names == ["СП 42.13330.2016"]
    scope = {"doc_id": "chosen", "version": "2020", "include_shared": False}
    plan = planner.simple_plan(query, history, scope)
    assert plan.doc_id == "chosen" and plan.version == "2020"
    assert plan.document_names is None and not plan.include_shared
    assert plan.tags is None and plan.types is None and plan.block is None


def test_pending_choice_needs_planner():
    from src.agents.services.dvd.clarification import CLARIFICATION

    history = [
        {
            "role": "assistant",
            "content": CLARIFICATION + "\n- СП 55, редакция 2016: 3.3",
        }
    ]
    assert planner.simple_plan("первый вариант", history) is None


@pytest.fixture
def planner_service(service, monkeypatch, fake_llm):
    monkeypatch.setenv(flags.PLANNER_SIMPLE_PATH, "true")
    monkeypatch.setenv(flags.PLANNER_PREFETCH, "true")
    fake_llm.json_responses = [verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Площадь озеленения — не менее 6 м² на человека [1]."]
    return service


async def _run_semantic(service, client, query="Площадь озеленения жилого квартала"):
    from tests.unit.test_dvd_structured_retrieval import run

    events = await run(service, client, query)
    assert not events_of_type(events, "error")
    assert answer_text(events)
    collected = service._schedule_persist_answer.call_args.args[2]
    return events, collected["metrics"]


async def test_simple_pipeline_bypasses_planner(planner_service, fake_mcp):
    planner_service.planner.build_plan = AsyncMock(
        side_effect=AssertionError("planner")
    )
    events, metrics = await _run_semantic(planner_service, fake_mcp)
    planner_service.planner.build_plan.assert_not_awaited()
    assert fake_mcp.search_calls[0].query == "Площадь озеленения жилого квартала"
    assert metrics.decisions["planner_path"] == "simple"
    assert metrics.decisions["planner_bypassed"] is True
    assert not metrics.counters.get("prefetch_started")
    assert events_of_type(events, "tool_call")


@pytest.mark.parametrize("ambiguous", [False, True])
async def test_simple_pipeline_falls_back_to_planner(planner_service, ambiguous):
    class InitialMiss(FakeDvdMcpClient):
        async def search(self, *args, **kwargs):
            result = await super().search(*args, **kwargs)
            if len(self.search_calls) == 1:
                return {**result, "ambiguous": True} if ambiguous else {"hits": []}
            return result

    client = InitialMiss()
    planner_service.planner.build_plan = AsyncMock(
        return_value=SemanticRetrievalPlan(
            retrieval_mode="semantic", search_query="озеленение"
        )
    )
    events, metrics = await _run_semantic(planner_service, client)
    planner_service.planner.build_plan.assert_awaited_once()
    assert len(client.search_calls) == 2
    assert client.search_calls[1].query == "озеленение"
    assert metrics.decisions["fast_path_fallback"] is True
    assert metrics.decisions["planner_path"] == "llm"
    assert not metrics.counters.get("prefetch_started")
    assert events_of_type(events, "chunk")[-1]["content"]["iteration"] == 2


async def test_simple_scope_survives_fallback(planner_service, fake_mcp):
    scope = {"doc_id": "selected", "version": "2016"}
    await planner_service.state_store.set_document_scope("chat-1", scope)
    requests = []
    search = fake_mcp.search_fragments

    async def first_miss(request, *, mode):
        requests.append(dict(request))
        if len(requests) == 1:
            return {"hits": [], "total": 0, "complete": True}
        return await search(request, mode=mode)

    fake_mcp.search_fragments = first_miss
    planner_service.planner.build_plan = AsyncMock(
        return_value=SemanticRetrievalPlan(
            retrieval_mode="semantic", search_query="озеленение"
        )
    )
    _, metrics = await _run_semantic(planner_service, fake_mcp)
    assert len(requests) == 2
    assert all(request["doc_id"] == "selected" for request in requests)
    assert all(request["version"] == "2016" for request in requests)
    assert metrics.decisions["fast_path_fallback"] is True
    assert metrics.decisions["planner_path"] == "llm"


@pytest.mark.parametrize(
    "changes,reused",
    [
        ({}, True),
        ({"search_query": "Площадь   озеленения жилого квартала"}, True),
        ({"search_query": "другая тема"}, False),
        ({"tags": ["school"]}, False),
        ({"block": "main"}, False),
        ({"types": ["clause"]}, False),
        ({"limit": 1}, False),
        ({"context_height": 0}, False),
        ({"kind": "text"}, False),
        ({"version": "2020"}, False),
        ({"document_names": ["СП 42.13330.2016"]}, False),
        ({"doc_id": "selected"}, False),
        ({"alternative_queries": ["озеленение"]}, False),
    ],
)
async def test_prefetch_effective_search_and_events(
    planner_service, fake_mcp, monkeypatch, changes, reused
):
    monkeypatch.setenv(flags.PLANNER_SIMPLE_PATH, "false")
    started = asyncio.Event()
    search = fake_mcp.search

    async def prefetched(*args, **kwargs):
        result = await search(*args, **kwargs)
        started.set()
        return result

    fake_mcp.search = prefetched

    async def build(*args, **kwargs):
        await asyncio.wait_for(started.wait(), 1)
        return validate_retrieval_plan(
            {
                "retrieval_mode": "semantic",
                "search_query": "Площадь озеленения жилого квартала",
                **changes,
            }
        )

    planner_service.planner.build_plan = AsyncMock(side_effect=build)
    events, metrics = await _run_semantic(planner_service, fake_mcp)
    assert metrics.counters["prefetch_started"] == 1
    assert metrics.counters.get("prefetch_reused", 0) == int(reused)
    normal_calls = 2 if changes.get("alternative_queries") else 1
    assert len(fake_mcp.search_calls) == (1 if reused else 1 + normal_calls)
    assert len(events_of_type(events, "tool_call")) == normal_calls
    assert metrics.decisions["planner_path"] == "llm"


async def test_prefetch_failure_does_not_affect_pipeline(
    planner_service, fake_mcp, monkeypatch
):
    monkeypatch.setenv(flags.PLANNER_SIMPLE_PATH, "false")
    failed = asyncio.Event()
    search = fake_mcp.search

    async def flaky(*args, **kwargs):
        if not failed.is_set():
            failed.set()
            raise RuntimeError("prefetch failed")
        return await search(*args, **kwargs)

    fake_mcp.search = flaky

    async def build(*args, **kwargs):
        await asyncio.wait_for(failed.wait(), 1)
        return SemanticRetrievalPlan(
            retrieval_mode="semantic", search_query="Площадь озеленения жилого квартала"
        )

    planner_service.planner.build_plan = AsyncMock(side_effect=build)
    events, metrics = await _run_semantic(planner_service, fake_mcp)
    assert metrics.counters["prefetch_started"] == 1
    assert not metrics.counters.get("prefetch_reused")
    assert len(fake_mcp.search_calls) == 1
    assert events_of_type(events, "tool_call")


@pytest.mark.parametrize("cancel_request", [False, True])
async def test_prefetch_task_is_cancelled_and_joined(
    planner_service, monkeypatch, cancel_request
):
    started, stopped = asyncio.Event(), asyncio.Event()

    async def search(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def build(*args, **kwargs):
        await asyncio.wait_for(started.wait(), 1)
        if cancel_request:
            await asyncio.Event().wait()
        return SemanticRetrievalPlan(
            retrieval_mode="semantic", search_query="другая тема"
        )

    monkeypatch.setattr(planner_service, "_semantic_search", search)
    planner_service.planner.build_plan = AsyncMock(side_effect=build)
    metrics = DvdRunMetrics("prefetch-test")
    task = asyncio.create_task(
        planner_service._plan_with_prefetch(
            FakeDvdMcpClient(),
            "m",
            "вопрос",
            [],
            None,
            None,
            {"metrics": metrics},
            None,
            True,
        )
    )
    if cancel_request:
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        _, prefetched = await asyncio.wait_for(task, 1)
        assert prefetched is None
    assert stopped.is_set()


async def test_new_flags_off_keep_planner_behaviour(
    planner_service, fake_mcp, monkeypatch
):
    monkeypatch.setenv(flags.PLANNER_SIMPLE_PATH, "false")
    monkeypatch.setenv(flags.PLANNER_PREFETCH, "false")
    planner_service.planner.build_plan = AsyncMock(
        return_value=SemanticRetrievalPlan(
            retrieval_mode="semantic", search_query="Площадь озеленения жилого квартала"
        )
    )
    _, metrics = await _run_semantic(planner_service, fake_mcp)
    planner_service.planner.build_plan.assert_awaited_once()
    assert len(fake_mcp.search_calls) == 1
    assert not metrics.counters.get("prefetch_started")
    assert metrics.decisions["planner_path"] == "llm"


def test_new_flags_default_on(monkeypatch):
    for name in (flags.PLANNER_SIMPLE_PATH, flags.PLANNER_PREFETCH):
        monkeypatch.delenv(name, raising=False)
        assert flags.enabled(name)


async def test_prefetch_reuses_real_planner_output(
    planner_service, fake_mcp, fake_llm, monkeypatch
):
    monkeypatch.setenv(flags.PLANNER_SIMPLE_PATH, "false")
    query = "Площадь озеленения жилого квартала"
    fake_llm.json_responses.insert(0, plan_json(search_query=query, limit=10))
    started = asyncio.Event()
    search, chat = fake_mcp.search, fake_llm.chat

    async def prefetched(*args, **kwargs):
        result = await search(*args, **kwargs)
        started.set()
        return result

    async def planner_chat(*args, **kwargs):
        if not fake_llm.chat_calls:
            await asyncio.wait_for(started.wait(), 1)
        return await chat(*args, **kwargs)

    fake_mcp.search = prefetched
    fake_llm.chat = planner_chat
    events, metrics = await _run_semantic(planner_service, fake_mcp, query)
    assert metrics.counters["planner_calls"] == 1
    assert metrics.counters["prefetch_reused"] == 1
    assert len(fake_mcp.search_calls) == 1
    assert len(events_of_type(events, "tool_call")) == 1
