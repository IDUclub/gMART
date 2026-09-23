"""Regression tests for document-QA failures observed on production (school regulations).

Covers topical search phrases, multi-query and document-list retrieval, broadening
instead of repeating an insufficient search, layout-free claim audits, source-label
hygiene, the metadata fallback, the grounded draft temperature and search logging.
"""

from __future__ import annotations

import json

from loguru import logger

from src.agents.services.dvd.dvd_context import DvdContextBuilder
from src.agents.services.dvd.dvd_reasoning import AnswerCritic, RetrievalPlanner
from src.agents.services.dvd.query_terms import (
    is_document_list_question,
    mentioned_documents,
    topical_query,
)
from src.agents.services.service_entities.dvd_plan import validate_retrieval_plan
from tests.helpers import FakeDvdMcpClient, answer_text, plan_json

PROD_TASK = (
    "Найти документы, содержащие требования к постройке школ, "
    "и предоставить выдержки с описанием требований"
)


async def _run(service, mcp, **overrides):
    kwargs = dict(
        dvd_mcp_client=mcp,
        token="tok",
        model="m",
        temperature=1.0,
        user_query="вопрос",
        chat_id="chat-1",
    )
    kwargs.update(overrides)
    return [event async for event in service.run_document_qa_pipeline(**kwargs)]


def _verdict(satisfied=False, critique="", missing=None, refined=None):
    return json.dumps(
        {
            "unsupported_claims": [],
            "missing_requirements": missing or [],
            "satisfied": satisfied,
            "critique": critique,
            "refined_search_query": refined,
            "claims": [],
        },
        ensure_ascii=False,
    )


def _stream_calls(fake_llm):
    return [call for call in fake_llm.chat_calls if call.stream]


# ---------------------------------------------------------------------------
# 1. Topical search phrases
# ---------------------------------------------------------------------------
def test_topical_query_strips_request_verbs_and_meta_words():
    assert topical_query(PROD_TASK) == "требования к постройке школ"
    assert (
        topical_query(
            "Скажи в каких документах содержится информация о пожарной безопасности."
        )
        == "пожарной безопасности"
    )
    assert topical_query("Какие требования к инсоляции жилых помещений?") == (
        "требования к инсоляции жилых помещений"
    )
    assert topical_query("Найти") == ""
    assert topical_query("Какие регламенты школ?\n\nЗадача: " + PROD_TASK) == (
        "регламенты школ; требования к постройке школ"
    )


def test_document_list_intent_detection():
    assert is_document_list_question(PROD_TASK)
    assert is_document_list_question("Какие есть регламенты застройки школ?")
    assert is_document_list_question("В каких документах требования к школам?")
    assert not is_document_list_question(
        "Какие требования к инсоляции жилых помещений?"
    )
    assert not is_document_list_question("что в пункте 3.3 СП 55")


def test_planner_replaces_instruction_with_topic_and_keeps_alternatives():
    plan = validate_retrieval_plan(
        {
            "retrieval_mode": "semantic",
            "search_query": PROD_TASK,
            "alternative_queries": [
                "Найти размещение общеобразовательных организаций",
                "требования к постройке школ",
                "земельный участок школы",
                "радиус доступности школ",
            ],
            "tags": ["образование", "выдуманный"],
        }
    )
    clamped = RetrievalPlanner._clamp(plan, PROD_TASK, ["образование"])
    assert clamped.search_query == "требования к постройке школ"
    assert clamped.alternative_queries == [
        "размещение общеобразовательных организаций",
        "земельный участок школы",
    ]
    assert clamped.intent == "document_list"
    assert clamped.tags == ["образование"]


def test_planner_echo_of_question_and_task_searches_the_task_topic():
    # Dev run: the planner echoed a two-part question plus the router task, and
    # the provision part («сколько жителей обеспечено») reached the vector index.
    question = (
        "Какие ограничения на строительство есть вокруг школ в проекте "
        "и сколько жителей обеспечено школами?"
    )
    task = "Получить выдержки из нормативных документов о строительстве вокруг школ"
    user_query = question + "\n\nЗадача: " + task
    plan = validate_retrieval_plan(
        {"retrieval_mode": "semantic", "search_query": user_query}
    )
    clamped = RetrievalPlanner._clamp(plan, user_query)
    assert clamped.search_query == topical_query(task)
    assert "жителей" not in clamped.search_query
    # «из нормативных документов» is the router's wording, not a document list.
    assert clamped.intent == "norm"


def test_planner_own_topic_is_kept_under_an_orchestrator_task():
    user_query = "Какие отступы от школ?\n\nЗадача: Найти требования к школам"
    plan = validate_retrieval_plan(
        {
            "retrieval_mode": "semantic",
            "search_query": "размещение зданий общеобразовательных организаций",
        }
    )
    clamped = RetrievalPlanner._clamp(plan, user_query)
    assert clamped.search_query == "размещение зданий общеобразовательных организаций"


def test_planner_keeps_exact_address_lookup_as_norm():
    plan = validate_retrieval_plan(
        {"retrieval_mode": "semantic", "search_query": "3.3", "intent": "document_list"}
    )
    clamped = RetrievalPlanner._clamp(plan, "какие документы: пункт 3.3 СП 55")
    assert clamped.retrieval_mode == "structure"
    assert clamped.intent == "norm"
    assert clamped.alternative_queries == []


def test_planner_prompt_lists_corpus_tags_and_forbids_instructions():
    prompt = RetrievalPlanner._prompt(None, None, ["образование", "транспорт"])
    assert "«найти документы" in prompt and "неверно" in prompt
    assert '"образование"' in prompt
    assert "tags=null" in RetrievalPlanner._prompt(None, None, None)


async def test_orchestrator_task_is_context_not_the_search_query(
    service, fake_llm, fake_mcp
):
    fake_llm.json_responses = [
        plan_json(search_query=PROD_TASK),
        _verdict(satisfied=True),
    ]
    fake_llm.answer_texts = ["Ответ"]
    await _run(
        service,
        fake_mcp,
        user_query="Какие есть регламенты застройки школ?",
        task=PROD_TASK,
        context_note="[Шаг 1, Граф] СП 42.13330.2016 п. 10.4",
        chat_id=None,
        persist_history=False,
    )
    planner_messages = fake_llm.chat_calls[0].messages
    assert "Задача: " + PROD_TASK in planner_messages[-1]["content"]
    assert any("СП 42.13330.2016" in m["content"] for m in planner_messages[1:-1])
    first = fake_mcp.search_calls[0]
    assert first.query == "требования к постройке школ"
    # A designation from an earlier step never becomes a literal filter.
    assert first.document_names is None


# ---------------------------------------------------------------------------
# 2. Multi-query, broadening and early stop
# ---------------------------------------------------------------------------
async def test_alternative_queries_are_searched_and_merged(service, fake_llm):
    mcp = FakeDvdMcpClient(
        hits_per_call=[
            [{"id": "a", "name": "СП 1", "numbering": "1", "text": "Первый [1]"}],
            [
                {"id": "a", "name": "СП 1", "numbering": "1", "text": "Первый [1]"},
                {"id": "b", "name": "СП 2", "numbering": "2", "text": "Второй"},
            ],
        ]
    )
    plan = json.loads(plan_json())
    plan["alternative_queries"] = ["земельный участок школы"]
    fake_llm.json_responses = [json.dumps(plan, ensure_ascii=False), _verdict(True)]
    fake_llm.answer_texts = ["Ответ"]
    events = await _run(service, mcp)
    assert [c.query for c in mcp.search_calls] == [
        "нормы озеленения",
        "земельный участок школы",
    ]
    assert len([e for e in events if e["type"] == "tool_call"]) == 2
    system = _stream_calls(fake_llm)[0].messages[0]["content"]
    assert "Первый" in system and "Второй" in system
    assert system.count("Первый") == 1


async def test_insufficient_repeat_is_broadened_then_stops_without_new_draft(
    service, fake_llm, fake_mcp
):
    fake_llm.json_responses = [
        plan_json(),
        _verdict(critique="нет требований", missing=["размер участка"]),
        _verdict(critique="нет требований", missing=["размер участка"]),
    ]
    fake_llm.answer_texts = ["d1", "d2", "d3"]
    events = await _run(service, fake_mcp)
    assert [c.limit for c in fake_mcp.search_calls] == [5, 20]
    assert fake_mcp.search_calls[1].context_height == 2
    # The third round has nothing new to read: no third draft is written.
    assert len(_stream_calls(fake_llm)) == 2
    text = answer_text(events)
    assert "Не удалось подтвердить" in text
    assert "СП 42.13330.2016" in text and "фрагменты: 7.5" in text
    assert "d1" not in text and "d2" not in text


async def test_draft_defect_is_rewritten_over_the_same_fragments(
    service, fake_llm, fake_mcp
):
    fake_llm.json_responses = [
        plan_json(),
        _verdict(critique="неверная ссылка"),
        _verdict(satisfied=True),
    ]
    fake_llm.answer_texts = ["d1", "Ответ [1]"]
    events = await _run(service, fake_mcp)
    assert len(fake_mcp.search_calls) == 1
    assert "Ответ [1]" in answer_text(events)


# ---------------------------------------------------------------------------
# 3-4. Critic: layout lines are not claims, evidence cites [N] labels only
# ---------------------------------------------------------------------------
def test_claim_texts_skip_table_layout_and_headings():
    answer = "\n".join(
        [
            "**Документы, содержащие требования к постройке школ**",
            "",
            "| № | Документ | Источник |",
            "|---|----------|----------|",
            "| 1 | СП 251.1325800.2016 | [2] |",
            "## Итог",
            "Перечень ниже:",
            "- Участок не менее 2 га [1]",
        ]
    )
    assert AnswerCritic._claim_texts(answer) == [
        "| 1 | СП 251.1325800.2016 | [2] |",
        "Участок не менее 2 га [1]",
    ]


async def test_critic_constrains_evidence_labels_and_flags_missing_evidence(fake_llm):
    captured = {}
    original = fake_llm.chat

    async def chat(*args, **kwargs):
        captured.setdefault("format", kwargs.get("format"))
        captured.setdefault("system", kwargs["messages"][0]["content"])
        return await original(*args, **kwargs)

    fake_llm.chat = chat
    fake_llm.json_responses = [_verdict(critique="нет", missing=["участок"])]
    context = "[1] СП 1\nТекст первый\x1e[2] СП 2\nТекст второй\x1e"
    verdict = await AnswerCritic(fake_llm).review(
        "m", "вопрос", context, "Ответ [1]", intent="document_list"
    )
    source_id = captured["format"]["$defs"]["ClaimEvidence"]["properties"]["source_id"]
    assert source_id["enum"] == ["[1]", "[2]"]
    assert "WHICH documents" in captured["system"]
    assert not verdict.satisfied and verdict.needs_evidence


def test_tree_context_has_no_document_labels():
    hits = [
        {
            "doc_id": "1",
            "name": "СП 1",
            "version": "2020",
            "numbering": "1.1",
            "text": "a",
        },
        {
            "doc_id": "2",
            "name": "СП 1",
            "version": "2020",
            "numbering": "2.1",
            "text": "b",
        },
        {
            "doc_id": "3",
            "name": "СП 3",
            "version": "2021",
            "numbering": "3.1",
            "text": "c",
        },
    ]
    context = DvdContextBuilder()._tree_context(hits)
    assert "Документ D" not in context
    assert "Документ: СП 1, ред. 2020" in context
    assert "другой документ с тем же названием, 2" in context


# ---------------------------------------------------------------------------
# 5, 7. Document-list retrieval and the metadata fallback
# ---------------------------------------------------------------------------
def test_mentioned_documents_prefers_resolved_references():
    hits = [
        {
            "text": "См. СП 251.1325800.2016 и ГОСТ Р 12.3.047-2012.",
            "references": [
                {"target_name": "СП 252.1325800.2016", "scope": "external"},
                {"target_name": "п. 3", "scope": "internal"},
            ],
        }
    ]
    assert mentioned_documents(hits) == [
        "СП 252.1325800.2016",
        "СП 251.1325800.2016",
        "ГОСТ Р 12.3.047-2012",
    ]


async def test_document_list_fetches_text_of_mentioned_documents(service, fake_llm):
    listing = {
        "id": "list",
        "name": "Приказ N 1-3.39-626/23",
        "numbering": "1",
        "text": "Перечень: СП 251.1325800.2016 «Здания общеобразовательных организаций».",
    }
    own = {
        "id": "own",
        "name": "СП 251.1325800.2016",
        "numbering": "5.1",
        "text": "Размер земельного участка школы определяют по заданию.",
    }
    mcp = FakeDvdMcpClient(hits_per_call=[[listing], [own], [listing]])
    fake_llm.json_responses = [plan_json(search_query=PROD_TASK), _verdict(True)]
    fake_llm.answer_texts = ["- СП 251.1325800.2016: размер участка по заданию [1]"]
    events = await _run(service, mcp, user_query=PROD_TASK)
    assert mcp.search_calls[1].document_names == ["СП 251.1325800.2016"]
    assert mcp.search_calls[2].document_names == ["Приказ N 1-3.39-626/23"]
    draft_system = _stream_calls(fake_llm)[0].messages[0]["content"]
    assert "Размер земельного участка школы" in draft_system
    assert "КАКИЕ документы" in draft_system
    assert draft_system.index("Размер земельного") < draft_system.index("Перечень:")
    assert "СП 251.1325800.2016: размер участка" in answer_text(events)


async def test_document_list_fallback_names_found_and_mentioned_documents(
    service, fake_llm
):
    listing = {
        "id": "list",
        "name": "Приказ N 1-3.39-626/23",
        "version": "2023",
        "numbering": "1",
        "text": "Перечень: СП 251.1325800.2016.",
    }
    mcp = FakeDvdMcpClient(default_hits=[listing])
    fake_llm.json_responses = [
        plan_json(search_query=PROD_TASK),
        _verdict(critique="нет выдержек", missing=["требования"]),
        _verdict(critique="нет выдержек", missing=["требования"]),
    ]
    fake_llm.answer_texts = ["d1", "d2"]
    events = await _run(service, mcp, user_query=PROD_TASK)
    text = answer_text(events)
    assert "По теме найдены фрагменты в следующих документах" in text
    assert "- Приказ N 1-3.39-626/23, ред. 2023 — фрагменты: 1" in text
    mentioned = text.split("упоминаются документы", 1)[1]
    assert "- СП 251.1325800.2016" in mentioned
    assert "не проверенное изложение" in text


# ---------------------------------------------------------------------------
# 6. Grounded draft temperature
# ---------------------------------------------------------------------------
async def test_draft_temperature_is_capped(service, fake_llm, fake_mcp, monkeypatch):
    monkeypatch.setenv("DVD_ANSWER_TEMPERATURE", "0.25")
    fake_llm.json_responses = [plan_json(), _verdict(True)]
    fake_llm.answer_texts = ["Ответ"]
    await _run(service, fake_mcp, temperature=1.0)
    assert _stream_calls(fake_llm)[0].options["temperature"] == 0.25
    fake_llm.json_responses = [plan_json(), _verdict(True)]
    fake_llm.answer_texts = ["Ответ"]
    fake_llm.chat_calls.clear()
    await _run(service, fake_mcp, temperature=0.1, chat_id=None, persist_history=False)
    assert _stream_calls(fake_llm)[0].options["temperature"] == 0.1


# ---------------------------------------------------------------------------
# 9. Search logging
# ---------------------------------------------------------------------------
async def test_mcp_client_logs_returned_documents_and_reads_tags():
    from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient

    client = DvdMcpClient.__new__(DvdMcpClient)

    async def execute_tool(name, arguments):
        if name == "get_tags":
            return {"count": 2, "tags": ["образование", "транспорт"]}
        return {
            "count": 1,
            "hits": [
                {
                    "name": "СП 251.1325800.2016",
                    "version": "2016",
                    "numbering": "5.1",
                    "score": 0.81234,
                    "text": "секретный текст",
                }
            ],
        }

    client.execute_tool = execute_tool
    logs = []
    sink = logger.add(lambda message: logs.append(str(message)))
    try:
        await client.search("школы")
        tags = await client.get_tags()
    finally:
        logger.remove(sink)
    line = next(line for line in logs if "DVD search" in line)
    assert "СП 251.1325800.2016" in line and "'numbering': '5.1'" in line
    assert "0.8123" in line and "секретный текст" not in line
    assert tags == ["образование", "транспорт"]


# ---------------------------------------------------------------------------
# Latency: concurrent searches and the rewrite budget
# ---------------------------------------------------------------------------
class _ConcurrentDvd(FakeDvdMcpClient):
    """Counts searches in flight at once."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_flight = self.peak = 0

    async def search(self, query, kind="all", limit=10, context_height=0, **kwargs):
        import asyncio

        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        return await super().search(query, kind, limit, context_height, **kwargs)


async def test_alternative_and_document_searches_run_concurrently(service, fake_llm):
    listing = {
        "id": "list",
        "name": "Приказ",
        "numbering": "1",
        "text": "Перечень: СП 251.1325800.2016, СП 252.1325800.2016.",
    }
    mcp = _ConcurrentDvd(default_hits=[listing])
    plan = json.loads(plan_json(search_query=PROD_TASK))
    plan["alternative_queries"] = ["земельный участок школы", "радиус доступности школ"]
    fake_llm.json_responses = [json.dumps(plan, ensure_ascii=False), _verdict(True)]
    fake_llm.answer_texts = ["Ответ [1]"]
    await _run(service, mcp, user_query=PROD_TASK)
    # 3 topical queries, then 3 documents (two named + the one found).
    assert len(mcp.search_calls) == 6
    assert mcp.peak == 3


async def test_rewrite_budget_is_configurable(service, fake_llm, fake_mcp, monkeypatch):
    monkeypatch.setenv("DVD_MAX_DRAFTS_PER_RETRIEVAL", "3")
    fake_llm.json_responses = [
        plan_json(),
        _verdict(critique="арифметика"),
        _verdict(critique="арифметика"),
        _verdict(satisfied=True),
    ]
    fake_llm.answer_texts = ["d1", "d2", "Ответ [1]"]
    events = await _run(service, fake_mcp)
    assert len(_stream_calls(fake_llm)) == 3
    assert len(fake_mcp.search_calls) == 1
    assert "Ответ [1]" in answer_text(events)


def test_empty_settings_from_ci_fall_back_to_defaults(monkeypatch):
    from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter
    from src.agents.services.dvd import dvd_rag_service
    from src.agents.services.dvd.dvd_reasoning import critic_reasoning_effort

    for name in (
        "DVD_ANSWER_TEMPERATURE",
        "DVD_MAX_DRAFTS_PER_RETRIEVAL",
        "DVD_CRITIC_REASONING_EFFORT",
    ):
        monkeypatch.setenv(name, "")
    assert dvd_rag_service._answer_temperature() == 0.2
    assert dvd_rag_service._max_drafts_per_retrieval() == 2
    adapter = OpenAiCompatAdapter.__new__(OpenAiCompatAdapter)
    assert critic_reasoning_effort(adapter, "gpt-oss-20b") == "medium"


def test_draft_table_becomes_a_bullet_list_with_labels():
    from src.agents.services.dvd.answer_generation import tables_to_lists

    draft = (
        "**Документы**\n\n"
        "| № | Документ | Требование |\n"
        "|---|----------|------------|\n"
        "| 1 | СП 2.4.3648‑20 | • до 1 км [1] |\n"
        "| 2 | Постановление № 525 | для садов – не более 1200 м [3] |\n"
        "\nВывод: прямого требования нет."
    )
    assert tables_to_lists(draft) == (
        "**Документы**\n\n"
        "- Документ: СП 2.4.3648‑20; Требование: до 1 км [1]\n"
        "- Документ: Постановление № 525; Требование: для садов – не более 1200 м [3]\n"
        "\nВывод: прямого требования нет."
    )
    # Plain text and a lone pipe line are left alone.
    assert tables_to_lists("a | b\n| c |") == "a | b\n| c |"
