"""Hermetic comparison contracts: identity, editions, paired evidence and pipeline."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastmcp.exceptions import ToolError

from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient
from src.agents.model_clients.context_budget import estimated_input_tokens
from src.agents.services.dvd.answer_generation import AnswerGenerationError
from src.agents.services.dvd.dialogue import resolve_reply
from src.agents.services.dvd.document_comparison import (
    ComparisonClarification,
    build_comparison_context,
    comparison_notice,
    comparison_numbers_supported,
    comparison_topic,
    has_paired_citations,
    literal_comparison,
    prepare_comparison_response,
    resolve_comparison,
)
from src.agents.services.dvd.dvd_context import source_records
from src.agents.services.dvd.dvd_reasoning import RetrievalPlanner
from src.agents.services.dvd.query_terms import is_document_comparison_question
from src.agents.services.dvd.retry_policy import CriticBudgetError, CriticResponseError
from src.agents.services.dvd.run_metrics import DvdRunMetrics
from src.agents.services.dvd.source_citations import readable
from src.agents.services.orchestrator.orchestrator_catalog import AGENT_CATALOG
from src.agents.services.orchestrator.orchestrator_plan_builder import (
    OrchestratorPlanBuilder,
)
from src.agents.services.service_entities.dvd_plan import validate_retrieval_plan
from src.agents.services.service_entities.orchestrator_plan import OrchestratorAgent
from tests.helpers import answer_text, events_of_type, final_chunk, statuses


def plan(**updates):
    return validate_retrieval_plan(
        {
            "retrieval_mode": "semantic",
            "intent": "document_comparison",
            "comparison_left": {"name": "СП 42"},
            **updates,
        }
    )


def client(versions=("2016", "2020", "2024")):
    return SimpleNamespace(
        find_document=AsyncMock(
            return_value=[{"doc_id": "d", "name": "СП 42", "version": versions[-1]}]
        ),
        document_versions=AsyncMock(return_value=list(versions)),
        list_user_documents=AsyncMock(return_value=[]),
        list_documents=AsyncMock(return_value=[]),
    )


def response():
    return {
        "mode": "editions",
        "left": {
            "doc_id": "d",
            "name": "СП 42",
            "version": "2016",
            "owner": "shared",
            "effective_date": "2016-01-01",
        },
        "right": {
            "doc_id": "d",
            "name": "СП 42",
            "version": "2024",
            "owner": "shared",
            "effective_date": "2024-01-01",
        },
        "stats": {
            "counts": {"modified": 1},
            "left_compared": 10,
            "right_compared": 10,
            "pairs_compared": 10,
            "unchanged": 9,
            "total_changes": 1,
        },
        "sections": [{"section": "Парковки", "counts": {"modified": 1}}],
        "changes": [
            {
                "kind": "modified",
                "left": {
                    "id": "old",
                    "numbering": "1",
                    "section": "Парковки",
                    "text": "Не менее 10 мест.",
                },
                "right": {
                    "id": "new",
                    "numbering": "2",
                    "section": "Парковки",
                    "text": "Не менее 20 мест.",
                    "text_truncated": True,
                },
                "changed_numbers": {"left": "10", "right": "20"},
                "word_diff": [{"left": "10", "right": "20"}],
            }
        ],
        "truncated": True,
        "topic_top_k": 60,
    }


@pytest.mark.parametrize(
    "query",
    [
        "Расскажи, в чём различия документа А от документа B.",
        "Чем отличаются версии документа А?",
        "Что изменилось в СП 42 в редакции 2024 по сравнению с 2016?",
        "Чем отличаются требования к парковкам в СП 42 и СП 55?",
        "Чем отличаются требования к парковкам в А и B?",
        "Сравни загруженный документ и документ из библиотеки",
        "В чём разница редакций?",
        "Чем новая редакция СП 42 отличается от прежней?",
    ],
)
def test_detection(query):
    assert is_document_comparison_question(query)


@pytest.mark.parametrize(
    "query",
    [
        "Какие требования к парковкам в СП 42?",
        "Какие документы регулируют парковки?",
        "Какая разница между стоянкой и гаражом?",
        "Что сказано в пункте 3.3 СП 42?",
        "Как изменилось население?",
        "Чем отличается парковка от гаража в СП 42?",
        "Сравни обеспеченность школами и садами",
    ],
)
def test_detection_negative(query):
    assert not is_document_comparison_question(query)


def test_schema_backwards_compatible_and_clamp():
    old = validate_retrieval_plan({"retrieval_mode": "semantic"})
    assert old.intent == "norm" and old.comparison_left is None
    comparison = plan(comparison_topic="парковки")
    result = RetrievalPlanner._clamp(comparison, "Что изменилось в пункте 3.3 СП 42?")
    assert result.intent == "document_comparison" and result.pattern is None
    assert result.comparison_topic == "парковки"
    assert "comparison_left" in RetrievalPlanner._prompt(None, None)


async def test_newest_previous_and_other_versions():
    result = await resolve_comparison(client(), plan(), "Чем отличаются версии СП 42?")
    assert result.arguments["left_version"] == "2020"
    assert result.arguments["right_version"] == "2024"
    assert "2016" in result.note


@pytest.mark.parametrize(
    "named,query,expected",
    [
        ("2016", "сравни редакции", "2024"),
        ("2024", "сравни редакции", "2020"),
        ("2020", "с предыдущей редакцией", "2016"),
        ("2020", "со следующей редакцией", "2024"),
    ],
)
async def test_one_named_edition(named, query, expected):
    result = await resolve_comparison(
        client(), plan(comparison_left={"name": "СП 42", "version": named}), query
    )
    assert (
        result.arguments["left_version"],
        result.arguments["right_version"],
    ) == tuple(sorted((named, expected)))


@pytest.mark.parametrize(
    "question,expected_query",
    [
        (
            "сравни требования к парковкам в редакциях 2024 и 2016",
            "требования к парковкам",
        ),
        ("сравни", None),
    ],
)
async def test_explicit_editions_and_topic(question, expected_query):
    result = await resolve_comparison(
        client(),
        plan(
            comparison_left={"name": "СП 42", "version": "2024"},
            comparison_right={"name": "СП 42", "version": "2016"},
            comparison_topic="требования к парковкам",
        ),
        question,
    )
    assert result.arguments == {
        "left_doc_id": "d",
        "right_doc_id": "d",
        "left_version": "2016",
        "right_version": "2024",
        "query": expected_query,
    }


async def test_incomplete_planner_preserves_literal_edition_years():
    result = await resolve_comparison(
        client(), plan(), "Что изменилось в СП 42 в редакции 2024 по сравнению с 2016?"
    )
    assert (result.arguments["left_version"], result.arguments["right_version"]) == (
        "2016",
        "2024",
    )


async def test_single_edition():
    with pytest.raises(ComparisonClarification, match="Сравнивать нечего.*2016"):
        await resolve_comparison(client(("2016",)), plan(), "сравни")


async def test_named_date_and_chronological_order():
    mcp = client(("01.02.2020", "31.12.2016", "01.01.2024"))
    result = await resolve_comparison(
        mcp, plan(comparison_left={"name": "СП 42", "version": "2020"}), "сравни"
    )
    assert result.arguments["left_version"] == "01.02.2020"
    assert result.arguments["right_version"] == "01.01.2024"


async def test_reversed_dates_are_sorted_chronologically():
    mcp = client(("31.12.2016", "01.01.2024"))
    result = await resolve_comparison(
        mcp,
        plan(
            comparison_left={"name": "СП 42", "version": "01.01.2024"},
            comparison_right={"name": "СП 42", "version": "31.12.2016"},
        ),
        "сравни",
    )
    assert result.arguments["left_version"] == "31.12.2016"
    assert result.arguments["right_version"] == "01.01.2024"


async def test_document_id_and_partial_designation_catalog_fallback():
    mcp = client()
    mcp.find_document.return_value = []
    mcp.list_documents.return_value = [
        {"doc_id": "d", "name": "СП 42.13330.2016", "version": "2024"}
    ]
    result = await resolve_comparison(
        mcp, plan(comparison_left={"doc_id": "d"}), "сравни"
    )
    assert result.arguments["left_doc_id"] == "d"
    result = await resolve_comparison(mcp, plan(), "сравни СП 42")
    assert result.arguments["left_doc_id"] == "d"


async def test_different_documents_default_to_current_editions():
    mcp = client()
    mcp.find_document.side_effect = lambda key: [
        {"doc_id": key, "name": key, "version": "2024"}
    ]
    original = plan(comparison_right={"name": "СП 55"})
    resolved = await resolve_comparison(mcp, original, "сравни СП 42 и СП 55")
    assert resolved.arguments["left_version"] is None
    assert resolved.arguments["right_version"] is None
    mcp.compare_documents = AsyncMock(return_value={"mode": "documents"})
    assert await resolved.compare(mcp) == {"mode": "documents"}
    mcp.compare_documents.assert_awaited_once_with(**resolved.arguments)


@pytest.mark.parametrize("side", ["left", "right"])
async def test_document_editions_clarification_roundtrip(side):
    mcp = client()
    mcp.find_document.side_effect = lambda key: [
        {"doc_id": key, "name": key, "version": "2024"}
    ]
    original = plan(comparison_right={"name": "СП 55"}, comparison_topic="парковки")
    resolved = await resolve_comparison(mcp, original, "сравни СП 42 и СП 55")
    mcp.compare_documents = AsyncMock(
        side_effect=ToolError(f"multiple editions found; specify {side}_version")
    )
    with pytest.raises(ComparisonClarification) as caught:
        await resolved.compare(mcp)
    pending = caught.value.pending
    assert pending["comparison_version"] is True
    assert pending["comparison_side"] == f"comparison_{side}"
    assert [option["members"][0]["version"] for option in pending["options"]] == [
        "2016",
        "2020",
        "2024",
    ]
    reply = resolve_reply("вариант 1", pending)
    assert reply["plan"][f"comparison_{side}"]["version"] == "2016"
    other = "right" if side == "left" else "left"
    assert reply["plan"][f"comparison_{other}"]["version"] is None
    assert reply["plan"]["comparison_topic"] == "парковки"


async def test_multiple_editions_message_with_both_fields_targets_unspecified_side():
    mcp = client()
    mcp.find_document.side_effect = lambda key: [
        {"doc_id": key, "name": key, "version": "2024"}
    ]
    resolved = await resolve_comparison(
        mcp,
        plan(
            comparison_left={"name": "СП 42", "version": "2016"},
            comparison_right={"name": "СП 55"},
        ),
        "сравни",
    )
    assert resolved.arguments["left_version"] == "2016"
    assert resolved.arguments["right_version"] is None
    mcp.compare_documents = AsyncMock(
        side_effect=ToolError(
            "multiple editions found; specify left_version/right_version"
        )
    )
    with pytest.raises(ComparisonClarification) as caught:
        await resolved.compare(mcp)
    assert caught.value.pending["comparison_side"] == "comparison_right"


async def test_unrelated_dvd_error_is_not_edition_clarification():
    mcp = client()
    resolved = await resolve_comparison(mcp, plan(), "сравни")
    mcp.compare_documents = AsyncMock(side_effect=ToolError("document not found"))
    with pytest.raises(ToolError, match="document not found"):
        await resolved.compare(mcp)


def test_editions_notice_names_before_and_after():
    notice = comparison_notice(response())
    assert "Сравнение: СП 42, ред. 2016 → ред. 2024." in notice


@pytest.mark.parametrize("multiple_editions", [False, True])
async def test_pipeline_current_editions_or_dvd_clarification(
    service, fake_llm, fake_mcp, multiple_editions
):
    fake_llm.json_responses = [
        plan(comparison_right={"name": "СП 55"}).model_dump_json()
    ]
    fake_mcp.find_document = AsyncMock(
        side_effect=lambda key: [{"doc_id": key, "name": key, "version": "2024"}]
    )
    fake_mcp.document_versions = AsyncMock(return_value=["2016", "2020", "2024"])
    data = response()
    data.update(mode="documents", changes=[])
    fake_mcp.compare_documents = AsyncMock(return_value=data)
    if multiple_editions:
        fake_mcp.compare_documents.side_effect = ToolError(
            "multiple editions found; specify right_version"
        )
    events = [
        e
        async for e in service.run_document_qa_pipeline(
            dvd_mcp_client=fake_mcp,
            token=None,
            model="m",
            temperature=0,
            user_query="Сравни СП 42 и СП 55",
            chat_id="chat-1",
            persist_history=False,
        )
    ]
    assert final_chunk(events)["done"]
    assert fake_mcp.compare_documents.call_args.kwargs["left_version"] is None
    assert fake_mcp.compare_documents.call_args.kwargs["right_version"] is None
    pending = await service.state_store.get_document_question("chat-1")
    if multiple_editions:
        assert pending["comparison_side"] == "comparison_right"
        assert "Укажите номер варианта" in answer_text(events)
    else:
        assert pending is None
        assert "различий не найдено" in answer_text(events)


async def test_user_document_and_shared_document():
    mcp = client(("2024",))
    mcp.list_user_documents.return_value = [
        {"doc_id": "private", "name": "Мой файл", "version": "2025"}
    ]
    result = await resolve_comparison(
        mcp,
        plan(
            comparison_left={"name": "Мой файл", "owner": "user"},
            comparison_right={"name": "СП 42", "owner": "shared"},
        ),
        "сравни",
        17,
    )
    assert result.arguments["left_doc_id"] == "private"
    assert result.arguments["right_doc_id"] == "d"
    mcp.list_user_documents.assert_awaited_once_with(17)


async def test_ambiguous_roundtrip_preserves_other_side_and_topic():
    mcp = client()
    mcp.find_document.return_value.append(
        {"doc_id": "other", "name": "СП 55", "version": "2024"}
    )
    original = plan(comparison_right={"name": "СП 55"}, comparison_topic="парковки")
    with pytest.raises(ComparisonClarification) as caught:
        await resolve_comparison(mcp, original, "сравни документы")
    pending = caught.value.pending
    assert pending and "Вариант" in str(caught.value)
    reply = resolve_reply("вариант 2", pending)
    assert reply["plan"]["comparison_left"]["doc_id"] == "other"
    assert reply["plan"]["comparison_right"]["name"] == "СП 55"
    assert reply["plan"]["comparison_topic"] == "парковки"


async def test_unresolved_never_uses_first_catalog_item():
    mcp = client()
    mcp.find_document.return_value = []
    mcp.list_documents.return_value = [
        {"doc_id": "unrelated", "name": "Другой документ"}
    ]
    with pytest.raises(ComparisonClarification, match="Не удалось найти"):
        await resolve_comparison(mcp, plan(), "сравни")


def test_paired_context_labels_survive_document_reordering():
    data = response()
    data["changes"].append(
        {
            "kind": "added",
            "right": {
                "id": "newer",
                "numbering": "3",
                "text": "Добавлено 30 мест.",
                "section": "Другой раздел",
            },
        }
    )
    before = deepcopy(data)
    summary, context, refs = build_comparison_context(data)
    assert data == before
    records = source_records(context)
    assert records["[1]"][1].startswith("Не менее 10")
    assert "20" in records["[2]"][1]
    assert "30" in records["[3]"][1]
    assert refs["[1]"] == ["СП 42, ред. 2016", "п. 1"]
    assert refs["[2]"] == ["СП 42, ред. 2024", "п. 2"]
    assert '"right": "[3]"' in summary
    assert '"changed_numbers"' in summary
    assert "**10 → 20**" in literal_comparison(data)
    assert not has_paired_citations(data, "Изменено [1]")
    assert has_paired_citations(data, "Изменено [1] [2]")


def test_user_and_shared_same_names_remain_distinct_sources():
    data = response()
    data["right"].update(doc_id="private", version="2016", owner="user")
    _, _, refs = build_comparison_context(data)
    assert "общая библиотека" in refs["[1]"][0]
    assert "пользовательский" in refs["[2]"][0]


async def test_mcp_flat_arguments_and_transport_identity():
    mcp = DvdMcpClient(Mock(), user_id="u")
    mcp.execute_tool = AsyncMock(return_value=response())
    result = await mcp.compare_documents("d", "d", "2016", "2024", "парковки")
    assert result["mode"] == "editions"
    tool, arguments = mcp.execute_tool.call_args.args
    assert tool == "compare_documents" and arguments["query"] == "парковки"
    assert "request" not in arguments and "user_id" not in arguments
    assert arguments["limit"] == 60


async def test_lookup_wrappers_unwrap_results_without_identity_arguments():
    mcp = DvdMcpClient(Mock(), user_id="u")
    mcp.execute_tool = AsyncMock(
        side_effect=[
            ["2016", "2024"],
            {"documents": [SimpleNamespace(doc_id="d", name="СП 42")]},
            {"documents": [{"doc_id": "private", "name": "Файл"}]},
        ]
    )
    assert await mcp.document_versions("СП 42") == ["2016", "2024"]
    assert (await mcp.find_document("СП 42"))[0]["doc_id"] == "d"
    assert (await mcp.list_user_documents(17))[0]["doc_id"] == "private"
    assert mcp.execute_tool.call_args.args == (
        "list_user_documents",
        {"scenario_id": "17"},
    )


async def test_orchestrator_routes_comparison_to_dvd(fake_llm):
    fake_llm.json_responses = [
        json.dumps({"mode": "execute", "steps": [{"agent": "norms", "task": "сравни"}]})
    ]
    builder = OrchestratorPlanBuilder(fake_llm)
    result = await builder.build_plan(
        "m", "Сравни версии СП 42", list(AGENT_CATALOG.values())
    )
    assert [step.agent for step in result.steps] == [OrchestratorAgent.DOCUMENTS]


async def test_orchestrator_preserves_independent_task(fake_llm):
    fake_llm.json_responses = [
        json.dumps(
            {
                "mode": "execute",
                "steps": [
                    {"agent": "provision", "task": "рассчитай обеспеченность"},
                    {"agent": "norms", "task": "сравни документы"},
                ],
            }
        )
    ]
    builder = OrchestratorPlanBuilder(fake_llm)
    result = await builder.build_plan(
        "m",
        "Сравни версии СП 42 и рассчитай обеспеченность школами",
        list(AGENT_CATALOG.values()),
    )
    assert [step.agent for step in result.steps] == [
        OrchestratorAgent.PROVISION,
        OrchestratorAgent.DOCUMENTS,
    ]


async def test_failed_interpretation_uses_literal_diff(service, fake_llm, fake_mcp):
    fake_llm.json_responses = [plan().model_dump_json()]
    fake_llm.answer_texts = ["Неподтверждённое толкование [1]."]
    service.critic.review = AsyncMock(return_value=SimpleNamespace(satisfied=False))
    lookup = client()
    fake_mcp.find_document = lookup.find_document
    fake_mcp.document_versions = lookup.document_versions
    fake_mcp.compare_documents = AsyncMock(return_value=response())
    events = [
        e
        async for e in service.run_document_qa_pipeline(
            dvd_mcp_client=fake_mcp,
            token=None,
            model="m",
            temperature=0,
            user_query="Сравни версии СП 42",
            persist_history=False,
        )
    ]
    answer = answer_text(events)
    assert "Неподтверждённое" not in answer
    assert "> Не менее 10 мест." in answer
    assert "> Не менее 20 мест." in answer
    assert "**10 → 20** [1] [2]" in answer


async def test_pipeline_comparison_and_reconnect(service, fake_llm, fake_mcp):
    fake_llm.json_responses = [plan().model_dump_json()]
    fake_llm.answer_texts = ["Число мест выросло с 10 до 20 [1] [2]."]
    service.critic.review = AsyncMock(return_value=SimpleNamespace(satisfied=True))
    lookup = client()
    fake_mcp.find_document = lookup.find_document
    fake_mcp.document_versions = lookup.document_versions
    fake_mcp.compare_documents = AsyncMock(return_value=response())
    kwargs = dict(
        dvd_mcp_client=fake_mcp,
        token="tok",
        model="m",
        temperature=0,
        user_query="Чем отличаются версии СП 42?",
        chat_id="chat-1",
    )
    events = [e async for e in service.run_document_qa_pipeline(**kwargs)]
    assert final_chunk(events)["done"]
    assert "10 до 20 [1] [2]" in answer_text(events)
    assert answer_text(events).count("Показаны наиболее существенные изменения") == 1
    assert answer_text(events).count("Сравнение:") == 1
    assert "По разделам:" not in answer_text(events)
    assert answer_text(events).count("Изменившиеся числа:") == 1
    assert statuses(events, "searching") and statuses(events, "self_review")
    assert not fake_mcp.search_calls
    service.critic.review.assert_awaited_once()
    assert set(source_records(service.critic.review.call_args.args[2])) == {
        "[1]",
        "[2]",
    }
    assert "юридическую эквивалентность" in next(
        c.messages[0]["content"] for c in fake_llm.chat_calls if c.stream
    )
    rid = events_of_type(events, "pipeline_started")[0]["content"]["request_id"]
    await_run = [
        e async for e in service.run_document_qa_pipeline(**kwargs, request_id=rid)
    ]
    assert await_run == events
    fake_mcp.compare_documents.assert_awaited_once()


@pytest.mark.parametrize(
    "critic_mode", ["oversize", "unavailable", "satisfied", "rejected"]
)
@pytest.mark.parametrize(
    "draft, accepted",
    [
        ("- Обзор раздела: изменено требование, теперь 20 мест [1] [2].", True),
        ("- Обзор раздела: теперь 999 мест [1] [2].", False),
        ("- Обзор раздела: теперь 20 мест [1] [2] [999].", False),
    ],
)
async def test_comparison_deterministic_check(
    service, fake_llm, fake_mcp, monkeypatch, critic_mode, draft, accepted
):
    from src.agents.services.dvd import dvd_rag_service

    fake_llm.json_responses = [plan().model_dump_json()]
    fake_llm.answer_texts = [draft, draft]
    service.critic.review = AsyncMock(
        return_value=SimpleNamespace(satisfied=critic_mode != "rejected")
    )
    if critic_mode == "unavailable":
        service.critic.review.side_effect = CriticResponseError("invalid verdict")
    # Only the critic's window is reduced; drafting and evidence retain their budget.
    if critic_mode == "oversize":
        monkeypatch.setattr(dvd_rag_service, "current_context_window", lambda: 512)
    metrics = DvdRunMetrics(request_id="comparison-test")
    monkeypatch.setattr(dvd_rag_service, "run_metrics", lambda collected: metrics)
    lookup = client()
    fake_mcp.find_document = lookup.find_document
    fake_mcp.document_versions = lookup.document_versions
    fake_mcp.compare_documents = AsyncMock(return_value=response())
    events = [
        e
        async for e in service.run_document_qa_pipeline(
            dvd_mcp_client=fake_mcp,
            token=None,
            model="m",
            temperature=0,
            user_query="Сравни версии СП 42",
            persist_history=False,
        )
    ]
    answer = answer_text(events)
    assert final_chunk(events)["done"]
    accepted = accepted and critic_mode != "rejected"
    assert (draft in answer) is accepted
    assert "**10 → 20** [1] [2]" in answer
    if not accepted:
        assert "> Не менее 10 мест." in answer
        assert "> Не менее 20 мест." in answer
    if critic_mode == "oversize":
        service.critic.review.assert_not_awaited()
        assert metrics.counters["critic_skipped"] == 1
        assert metrics.counters.get("critic_calls", 0) == 0
    else:
        service.critic.review.assert_awaited_once()


@pytest.mark.parametrize(
    "draft, supported",
    [
        ("- Было 20 m, стало **25 m** [1] [2].", True),
        ("- Доля 12,5 % [1] [2].", True),
        ("- Стало 250 m [1] [2].", False),
        ("- Доля 50 % [1] [2].", False),
    ],
)
def test_comparison_quantities(draft, supported):
    data = response()
    data["changes"][0]["changed_numbers"] = {"left": "20 m", "right": "25 m"}
    assert comparison_numbers_supported(data, "Доля 12.5%", draft) is supported


@pytest.mark.parametrize("failure", ["single", "tool"])
async def test_pipeline_resolution_failure_never_drafts(
    service, fake_llm, fake_mcp, failure
):
    fake_llm.json_responses = [plan().model_dump_json()]
    lookup = client(("2024",))
    fake_mcp.find_document = lookup.find_document
    fake_mcp.document_versions = lookup.document_versions
    if failure == "tool":
        fake_mcp.find_document.side_effect = ToolError("document not found")
    events = [
        e
        async for e in service.run_document_qa_pipeline(
            dvd_mcp_client=fake_mcp,
            token=None,
            model="m",
            temperature=0,
            user_query="Сравни версии СП 42",
            persist_history=False,
        )
    ]
    assert final_chunk(events)["done"]
    assert not statuses(events, "answer_drafting")
    assert (
        "Сравнивать нечего" in answer_text(events)
        if failure == "single"
        else "Не удалось сопоставить" in answer_text(events)
    )


@pytest.mark.parametrize("separate_ids", [False, True])
@pytest.mark.parametrize(
    "named,query,expected",
    [
        (None, "Чем отличаются версии СП 55.13330.2016?", ("2020", "2024")),
        ("2016", "сравни редакции", ("2016", "2024")),
        ("2024", "сравни редакции", ("2020", "2024")),
        ("2020", "до 2020", ("2016", "2020")),
        ("2020", "после 2020", ("2020", "2024")),
    ],
)
async def test_editions_resolve_storage_id(separate_ids, named, query, expected):
    mcp = client()
    mcp.find_document.return_value = [
        {
            "doc_id": f"d-{v}" if separate_ids else "d",
            "name": "СП 55.13330.2016",
            "version": v,
        }
        for v in ("2016", "2020", "2024")
    ]
    resolved = await resolve_comparison(
        mcp, plan(comparison_left={"name": "СП 55.13330.2016", "version": named}), query
    )
    for side, version in zip(("left", "right"), expected):
        assert resolved.arguments[f"{side}_version"] == version
        assert resolved.arguments[f"{side}_doc_id"] == (
            f"d-{version}" if separate_ids else "d"
        )


async def test_editions_use_other_versions_mapping():
    mcp = client()
    mcp.find_document.return_value = [
        {
            "doc_id": "old",
            "name": "СП 42",
            "version": "2020",
            "other_versions": ["2016"],
        },
        {"doc_id": "new", "name": "СП 42", "version": "2024"},
    ]
    resolved = await resolve_comparison(
        mcp, plan(comparison_left={"name": "СП 42", "version": "2016"}), "сравни"
    )
    assert resolved.arguments["left_doc_id"] == "old"
    assert resolved.arguments["right_doc_id"] == "new"


def test_comparison_context_fits_32k_window():
    data = response()
    data["changes"] = []
    for i in range(60):
        change = deepcopy(response()["changes"][0])
        for side in ("left", "right"):
            change[side].update(
                id=f"{side}-{i}", text=("Длинный текст нормы. " * 100)[:2000]
            )
            assert len(change[side]["text"]) == 2000
        data["changes"].append(change)
    before = deepcopy(data)
    budget = int((32000 - 8000 - 2300) * 0.8)
    packed = prepare_comparison_response(data, budget)
    summary, context, refs = build_comparison_context(packed, budget=budget)
    assert estimated_input_tokens([{"content": summary + context}]) <= budget
    assert packed["omitted_changes"] > 0
    assert packed["omitted_changes"] + len(packed["changes"]) == 60
    assert "Показаны наиболее существенные изменения" in comparison_notice(packed)
    assert "пропущено изменений" not in comparison_notice(packed)
    assert '"word_diff"' in summary and '"changed_numbers"' in summary
    assert len(refs) == len(packed["changes"]) * 2
    assert data == before


@pytest.mark.parametrize(
    "failure",
    [
        CriticResponseError("Critic produced invalid structured response"),
        CriticBudgetError("budget exhausted"),
        ValueError("structured request exceeds configured context window"),
        AnswerGenerationError("draft failed"),
    ],
)
async def test_comparison_model_failure_finishes(service, fake_llm, fake_mcp, failure):
    fake_llm.json_responses = [plan().model_dump_json()]
    fake_llm.answer_texts = ["Неподтверждённое толкование [1]."]
    service.critic.review = AsyncMock(side_effect=failure)
    if isinstance(failure, AnswerGenerationError):

        async def fail_draft(*args, **kwargs):
            raise failure
            yield

        service._generate_answer = fail_draft
        service.critic.review = AsyncMock(return_value=SimpleNamespace(satisfied=True))
    lookup = client()
    fake_mcp.find_document = lookup.find_document
    fake_mcp.document_versions = lookup.document_versions
    fake_mcp.compare_documents = AsyncMock(return_value=response())
    events = [
        e
        async for e in service.run_document_qa_pipeline(
            dvd_mcp_client=fake_mcp,
            token=None,
            model="m",
            temperature=0,
            user_query="Сравни версии СП 42",
            persist_history=False,
        )
    ]
    assert final_chunk(events)["done"]
    answer = answer_text(events)
    assert "Неподтверждённое" not in answer
    assert "> Не менее 10 мест." in answer
    assert "> Не менее 20 мест." in answer
    assert "**10 → 20** [1] [2]" in answer
    arguments = fake_mcp.compare_documents.call_args.kwargs
    assert arguments["max_chars"] == 600
    assert 1 <= arguments["limit"] <= 60


async def test_explicit_storage_uuid_resolves_edition_family():
    mcp = client(("2016", "2024"))
    entries = [
        {"doc_id": "old", "name": "СП 42", "version": "2016"},
        {"doc_id": "new", "name": "СП 42", "version": "2024"},
    ]
    mcp.find_document.side_effect = lambda key: (
        entries if key == "СП 42" else [c for c in entries if c["doc_id"] == key]
    )
    resolved = await resolve_comparison(
        mcp, plan(comparison_left={"doc_id": "new"}), "сравни"
    )
    assert resolved.arguments["left_doc_id"] == "old"
    assert resolved.arguments["right_doc_id"] == "new"


async def test_explicit_separate_editions_are_chronological():
    mcp = client(("2016", "2024"))
    mcp.find_document.return_value = [
        {"doc_id": "old", "name": "СП 42", "version": "2016"},
        {"doc_id": "new", "name": "сп  42", "version": "2024"},
    ]
    resolved = await resolve_comparison(
        mcp,
        plan(
            comparison_left={"name": "СП 42", "version": "2024"},
            comparison_right={"name": "СП 42", "version": "2016"},
        ),
        "сравни",
    )
    assert resolved.arguments["left_doc_id"] == "old"
    assert resolved.arguments["right_doc_id"] == "new"


@pytest.mark.parametrize(
    "question,names,expected",
    [
        (
            "Чем отличаются требования к естественному освещению учебных кабинетов в СП 2.4.3648-20 и СП 2.4.2.4283-26?",
            (),
            "требования к естественному освещению учебных кабинетов",
        ),
        (
            "Какие изменения в разделе Пожарная безопасность между редакцией А и редакцией B 2020 и 2024 годов?",
            ("редакцией А", "редакцией B"),
            "пожарная безопасность",
        ),
        ("Расскажи, в чём различия А от B", ("А", "B"), None),
        ("Чем отличаются версии СП 55.13330.2016?", (), None),
        ("Чем отличается СП 42 в редакциях 2020 и 2024?", (), None),
    ],
)
def test_comparison_topic_from_question(question, names, expected):
    assert comparison_topic(question, names) == expected


async def test_derived_topic_is_forwarded_to_dvd():
    question = "Чем отличаются требования к естественному освещению учебных кабинетов в СП 2.4.3648-20 и СП 2.4.2.4283-26?"
    resolved = await resolve_comparison(client(), plan(), question)
    assert (
        resolved.arguments["query"]
        == "требования к естественному освещению учебных кабинетов"
    )


@pytest.mark.parametrize(
    "topic,expected",
    [
        ("различия", None),
        ("сравнение документов СП 2.4.3648-20 и СП 2.4.2.4283-26 2020 2026", None),
        ("изменения парковок", None),
        (
            "различия в требованиях к освещению СП 2.4.3648-20 2020",
            None,
        ),
    ],
)
async def test_planner_topic_is_sanitized(topic, expected):
    resolved = await resolve_comparison(
        client(),
        plan(comparison_topic=topic),
        "Расскажи, в чём различия СП 2.4.3648-20 от СП 2.4.2.4283-26.",
    )
    assert resolved.arguments["query"] == expected


@pytest.mark.parametrize(
    "question,topic,expected",
    [
        (
            "Чем отличаются версии СП 55.13330.2016?",
            "различия в требованиях к проектированию и размещению зданий общеобразовательных организаций",
            None,
        ),
        (
            "Чем отличаются требования к естественному освещению учебных кабинетов в СП 42?",
            "требования к естественному освещению учебных кабинетов",
            "требования к естественному освещению учебных кабинетов",
        ),
        (
            "Чем отличаются требования к естественному освещению учебных кабинетов в СП 42?",
            "требования к проектированию и размещению зданий",
            "требования к естественному освещению учебных кабинетов",
        ),
        (
            "Чем отличаются требования к естественному освещению учебных кабинетов в СП 42?",
            "требованиями к естественному освещению учебного кабинета",
            "требованиями к естественному освещению учебного кабинета",
        ),
        (
            "Чем отличаются требования к учёту освещения в СП 42?",
            "требования к учету освещения",
            "требования к учету освещения",
        ),
        (
            "Чем отличаются требования к освещению кабинетов в СП 42?",
            "освещение кабинетов зданий организаций",
            "требования к освещению кабинетов",
        ),
    ],
)
async def test_planner_topic_must_be_grounded_in_question(question, topic, expected):
    resolved = await resolve_comparison(
        client(), plan(comparison_topic=topic), question
    )
    assert resolved.arguments["query"] == expected


@pytest.mark.parametrize(
    "version,suffix",
    [
        ("СП 2.4.2.4283-26", ""),
        (
            "СП 2.4.2.4283-26 (от 2 июня 2026 г. N 19)",
            " (от 2 июня 2026 г. N 19)",
        ),
        (
            "СП 2.4.2.4283-26 от 2 июня 2026 г. N 19",
            " (от 2 июня 2026 г. N 19)",
        ),
    ],
)
def test_redundant_edition_is_removed_from_all_comparison_labels(
    version, suffix, monkeypatch
):
    data = response()
    data["mode"] = "documents"
    data["left"].update(name="СП 2.4.3648-20", version="СП 2.4.3648-20")
    data["right"].update(name="СП 2.4.2.4283-26", version=version)
    before = deepcopy(data)
    notice = comparison_notice(data)
    assert notice.splitlines()[0] == (
        f"Сравнение: СП 2.4.3648-20 → СП 2.4.2.4283-26{suffix}."
    )
    _, context, refs = build_comparison_context(data, budget=17000)
    assert refs["[1]"][0] == "СП 2.4.3648-20"
    assert refs["[2]"][0] == f"СП 2.4.2.4283-26{suffix}"
    monkeypatch.setattr(
        "src.agents.services.dvd.source_citations.enabled", lambda: True
    )
    rendered = [
        notice,
        context,
        readable(literal_comparison(data), refs),
        readable(literal_comparison(data, numbers_only=True), refs),
    ]
    for text in rendered:
        assert "ред." not in text
        if suffix:
            assert suffix in text
        assert "СП 2.4.2.4283-26, ред." not in text
        assert "СП 2.4.2.4283-26 (от 2 июня 2026 г. N 19) (от" not in text
    assert data == before


def test_independent_editions_keep_their_labels():
    data = response()
    data["right"]["version"] = "2021"
    assert comparison_notice(data).startswith(
        "Сравнение: СП 42, ред. 2016 → ред. 2021.\n"
    )
    _, context, refs = build_comparison_context(data, budget=17000)
    assert refs["[1]"][0] == "СП 42, ред. 2016"
    assert refs["[2]"][0] == "СП 42, ред. 2021"
    assert "ред. 2016" in context and "ред. 2021" in context


def test_notice_is_compact_and_numbers_appendix_contains_only_numbers():
    data = many_sections_response()
    data.update(omitted_changes=29, omitted_sections=295)
    data["left"].update(title="", status="active")
    notice = comparison_notice(data)
    assert len(notice.splitlines()) <= 5
    assert notice.count("Показаны наиболее существенные изменения") == 1
    for technical in (
        ", ;",
        "дата",
        "active",
        "бюджет",
        "29",
        "295",
        "modified",
        "По разделам",
    ):
        assert technical not in notice
    appendix = literal_comparison(data, numbers_only=True)
    assert appendix == "Изменившиеся числа: **10 → 20** [1] [2]"


def test_documents_notice_names_both_sides_and_uses_public_counts():
    data = response()
    data["mode"] = "documents"
    data["right"].update(name="СП 55", version=None, title="")
    data["stats"]["counts"] = dict(only_left=2, only_right=3, different=4, similar=5)
    notice = comparison_notice(data)
    assert notice.startswith("Сравнение: СП 42, ред. 2016 → СП 55.\n")
    assert "Только в A: 2; Только в B: 3; Различаются: 4; Сходны: 5." in notice
    assert notice.count("юридическую эквивалентность") == 1
    assert notice.count("тематическая выборка") == 1
    assert "не указана" not in notice


@pytest.mark.parametrize("labels", ["", "[999]", "[N]"])
@pytest.mark.parametrize(
    "repair",
    ["accepted", "missing", "invented", "unpaired", "critic_rejected", "failure"],
)
async def test_comparison_citation_revision(
    service, fake_llm, fake_mcp, labels, repair
):
    fake_llm.json_responses = [plan().model_dump_json()]
    draft = f"- Число мест выросло с 10 до 20. {labels}".strip()
    revised = {
        "accepted": "- Число мест выросло с 10 до 20. [1] → [2]",
        "missing": "- Число мест выросло с 10 до 20.",
        "invented": "- Число мест выросло до 999 мест. [1] [2]",
        "unpaired": "- Число мест выросло с 10 до 20. [1]",
        "critic_rejected": "- Число мест выросло с 10 до 20. [1] → [2]",
        "failure": "",
    }[repair]
    fake_llm.answer_texts = [draft, revised]
    if repair == "failure":
        generate = service._generate_answer

        async def fail_revision(*args, **kwargs):
            if kwargs.get("revision_note"):
                raise AnswerGenerationError("revision failed")
            async for event in generate(*args, **kwargs):
                yield event

        service._generate_answer = fail_revision
    service.critic.review = AsyncMock(
        return_value=SimpleNamespace(satisfied=repair != "critic_rejected")
    )
    lookup = client()
    fake_mcp.find_document = lookup.find_document
    fake_mcp.document_versions = lookup.document_versions
    fake_mcp.compare_documents = AsyncMock(return_value=response())
    events = [
        e
        async for e in service.run_document_qa_pipeline(
            dvd_mcp_client=fake_mcp,
            token=None,
            model="m",
            temperature=0,
            user_query="Сравни версии СП 42",
            persist_history=False,
        )
    ]
    answer = answer_text(events)
    assert final_chunk(events)["done"]
    assert answer.count("Сравнение:") == 1
    assert answer.count("Изменившиеся числа:") == 1
    if repair == "accepted":
        assert revised in answer
        assert "> Не менее" not in answer
        assert "По разделам:" not in answer
    else:
        assert "> Не менее 10 мест." in answer
        assert "> Не менее 20 мест." in answer
    calls = [c for c in fake_llm.chat_calls if c.stream]
    assert len(calls) == (1 if repair == "failure" else 2)
    if repair != "failure":
        assert "добавь метки источников [N] к каждому пункту" in str(calls[-1].messages)
        assert "[999]" not in service.critic.review.call_args.args[3]
    fake_mcp.compare_documents.assert_awaited_once()


def many_sections_response():
    data = response()
    data["sections"] = [
        (
            {
                "section": f"Раздел {i}: " + "Описание " * 25,
                "counts": {"modified": i + 1, "unchanged": 999},
            }
            if i < 15
            else {"section": f"Неизменённый раздел {i}", "counts": {"unchanged": 999}}
        )
        for i in range(100)
    ]
    data["stats"]["counts"]["unchanged"] = 9
    return data


def test_summary_caps_changed_sections_and_compact_fallback():
    data = many_sections_response()
    packed = prepare_comparison_response(data, 17000)
    assert len(packed["sections"]) == 10
    assert packed["sections"][0]["section"].startswith("Раздел 14:")
    assert all(
        len(s["section"]) <= 80 and "unchanged" not in s["counts"]
        for s in packed["sections"]
    )
    summary, _, _ = build_comparison_context(packed, budget=17000)
    notice = comparison_notice(data)
    for text in (summary, notice):
        assert "unchanged" not in text
        assert "Неизменённый раздел" not in text
    assert summary.count("- Раздел ") == 10
    assert "Пропущено разделов сводки: 5." in summary
    assert "По разделам:" not in notice
    assert "Пропущено" not in notice
    fallback = comparison_notice(packed) + "\n" + literal_comparison(packed)
    assert len(fallback) < 3000
    assert fallback.count("- Раздел ") == 10
    assert "Пропущено" not in fallback
    assert "без изменений" in fallback


async def test_32k_budget_keeps_eight_changes_in_draft_prompt(
    service, fake_llm, fake_mcp
):
    data = many_sections_response()
    data["changes"] = []
    for i in range(8):
        change = deepcopy(response()["changes"][0])
        for side in ("left", "right"):
            change[side].update(
                id=f"{side}-{i}",
                text=("Требования к естественному освещению. " * 30)[:600],
            )
        data["changes"].append(change)
    data["stats"]["total_changes"] = 8
    fake_llm.json_responses = [plan().model_dump_json()]
    fake_llm.answer_texts = ["Изменение [1] [2]."]
    service.context_reducer.configured_window = 32000
    service.critic.review = AsyncMock(return_value=SimpleNamespace(satisfied=False))
    lookup = client()
    fake_mcp.find_document = lookup.find_document
    fake_mcp.document_versions = lookup.document_versions
    fake_mcp.compare_documents = AsyncMock(return_value=data)
    events = [
        e
        async for e in service.run_document_qa_pipeline(
            dvd_mcp_client=fake_mcp,
            token=None,
            model="m",
            temperature=0,
            user_query="Чем отличаются версии СП 42?",
            persist_history=False,
        )
    ]
    assert final_chunk(events)["done"]
    args = fake_mcp.compare_documents.call_args.kwargs
    assert 30 <= args["limit"] <= 60
    assert args["max_chars"] == 600
    assert args["query"] is None
    prompt = next(c.messages[0]["content"] for c in fake_llm.chat_calls if c.stream)
    # Parse evidence separately so the builder's tree prefix remains at the start.
    evidence = prompt.split("\nФрагменты нормативных документов:\n", 1)[1]
    assert set(source_records(evidence)) == {f"[{i}]" for i in range(1, 17)}
    assert "Детали изменений не поместились" not in answer_text(events)


def test_summary_is_trimmed_before_priority_changes():
    data = many_sections_response()
    data["changes"] = [deepcopy(response()["changes"][0]) for _ in range(10)]
    for i, change in enumerate(data["changes"]):
        change["changed_numbers"] = None if i < 2 else {"left": "10", "right": "20"}
        for side in ("left", "right"):
            change[side].update(id=f"{side}-{i}", text="Текст нормы. " * 50)
    packed = prepare_comparison_response(data, 14000)
    assert len(packed["changes"]) >= 8
    assert all(c["changed_numbers"] for c in packed["changes"][:8])
    assert len(packed["sections"]) < 10
    summary, context, refs = build_comparison_context(packed, budget=14000)
    assert estimated_input_tokens([{"content": summary + context}]) <= 14000
    assert len(refs) >= 16
