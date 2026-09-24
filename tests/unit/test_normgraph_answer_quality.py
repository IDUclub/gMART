"""Norms-QA answers: grounded values, labelled context, no tables, relevant norms only."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient
from src.agents.services.dvd.answer_generation import (
    StreamingTableRewriter,
    tables_to_lists,
)
from src.agents.services.normgraph.normgraph_context import (
    NormGraphContextBuilder,
    display_version,
)
from src.agents.services.normgraph.normgraph_rag_service import NormGraphRagService
from src.agents.services.normgraph.normgraph_reasoning import (
    NormGraphAnswerCritic,
    NormGraphRetrievalPlanner,
    is_placement_question,
    ungrounded_quantities,
)
from src.agents.services.service_entities.normgraph_plan import NormGraphPlan

HIT = {
    "id": "r1",
    "subject": "КСК",
    "object": "селитебная зона",
    "kind": "минимальное_расстояние",
    "value": {
        "operator": ">=",
        "number": 300.0,
        "unit": "м",
        "condition": "до 20 голов",
    },
    "extraction_text": "КСК должны быть отделены от селитебной зоны санитарно-защитной зоной.",
    "provenance": {"name": "СП 2.4.3648-20", "version": "3648", "numbering": "5.4"},
}


# ── context ────────────────────────────────────────────────────────────


def test_context_separates_the_triple_from_the_quotable_clause_text():
    context = NormGraphContextBuilder().build_context([HIT])

    assert "Структура (служебная, не цитата): КСК → селитебная зона" in context
    assert (
        "Текст пункта: «КСК должны быть отделены от селитебной зоны "
        "санитарно-защитной зоной.»" in context
    )


@pytest.mark.parametrize(
    ("name", "version", "shown"),
    [
        ("СП 2.4.3648-20", "3648", None),
        ("СП 2.4.2.4283-26", "4283", None),
        ("Постановление от 4 декабря 2017 г. N 525", "2017", None),
        ("Тестовые Нормы", "unknown", None),
        ("СП 42.13330", "2016", "2016"),
        ("СП 42.13330.2016", "ред. 2", "ред. 2"),
    ],
)
def test_version_taken_from_the_document_code_is_not_a_redaction(name, version, shown):
    assert display_version(name, version) == shown


def test_header_omits_a_bogus_redaction():
    context = NormGraphContextBuilder().build_context([HIT])
    assert context.startswith("[1] СП 2.4.3648-20, п. 5.4 (id: r1)")


# ── grounding of values ─────────────────────────────────────────────────


def test_quantities_absent_from_the_context_are_reported():
    context = (
        "[1] … | >= 300.0 м (условие: до 20 голов)\nТекст пункта: «не более 1 000 м»"
    )
    answer = (
        "Не менее 300 м [1]; не более 1000 метров; для сёл 1,5 км; "
        "п. 2.6.10 и [2] не величины; 40 м от окон."
    )
    assert ungrounded_quantities(answer, context) == ["1,5 км", "40 м"]


@pytest.mark.asyncio
async def test_critic_rejects_an_invented_value_without_asking_the_model():
    class NoLlm:
        async def chat(self, **kwargs):
            raise AssertionError("the model must not be asked")

    verdict = await NormGraphAnswerCritic(NoLlm()).review(
        "model", "вопрос", "Структура: a → b | k | >= 300.0 м", "Не менее 500 м [1]."
    )

    assert not verdict.satisfied
    assert "500 м" in verdict.critique


def test_critic_prompt_rejects_off_topic_norms_interpretations_and_fake_quotes():
    prompt = NormGraphAnswerCritic._prompt()
    assert "не по предмету вопроса" in prompt
    assert "расшифровки сокращений" in prompt
    assert "«Текст пункта»" in prompt


# ── retrieval planning ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("query", "placement"),
    [
        ("Какие ограничения действуют на размещение детских садов?", True),
        ("Какие ограничения на строительство вокруг школ?", True),
        ("Минимальное расстояние от окон до автостоянок?", True),
        ("Что такое красная линия?", False),
        ("Какие помещения должны быть в детском саду?", False),
    ],
)
def test_placement_questions_are_recognised(query, placement):
    assert is_placement_question(query) is placement


def test_planner_never_passes_model_chosen_kinds():
    plan = NormGraphRetrievalPlanner._clamp(
        NormGraphPlan(search_query="q", kinds=["выдуманный_вид"]), "q"
    )
    assert plan.kinds is None


def test_planner_prompt_asks_for_placement_vocabulary():
    assert "лексикой размещения" in NormGraphRetrievalPlanner._prompt(None, None, None)


# ── answer drafting ─────────────────────────────────────────────────────


class StreamingLlm:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    async def chat(self, model, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        text = self.text

        async def parts():
            for index in range(0, len(text), 5):
                yield SimpleNamespace(
                    message=SimpleNamespace(content=text[index : index + 5])
                )

        return parts()


def _service(llm) -> NormGraphRagService:
    service = NormGraphRagService.__new__(NormGraphRagService)
    service.llm_client = llm
    return service


async def _draft(service, context: str) -> str:
    parts = []
    async for event in service._generate_answer(
        "model", "Какие ограничения?", context, 0.2, [], 1
    ):
        parts.append(event["content"]["text"])
    return "".join(parts)


@pytest.mark.asyncio
async def test_streamed_table_reaches_the_client_as_a_list():
    table = (
        "Нормы:\n| № | Норма | Источник |\n|---|---|---|\n| 1 | 500 м | [1] |\nИтог."
    )
    draft = await _draft(_service(StreamingLlm(table)), "контекст")

    assert draft == tables_to_lists(table)
    assert "|" not in draft


@pytest.mark.asyncio
async def test_conflict_rule_only_when_conflicts_were_found():
    llm = StreamingLlm("ответ")
    service = _service(llm)

    await _draft(service, "[1] СП")
    await _draft(service, "[1] СП\n\nОбнаруженные противоречия:\n- a vs b")

    without, with_conflicts = (call["messages"][0]["content"] for call in llm.calls)
    assert "противоречи" not in without.split("Контекст (найденные ограничения)")[0]
    assert "Обнаруженные противоречия» — обязательно" in with_conflicts
    assert "Не используй таблицы" in without
    assert "Используй только ограничения, которые прямо отвечают" in without


def test_streaming_table_rewriter_matches_the_batch_rewrite_for_any_chunking():
    text = (
        "Итог:\n\n| № | Норма | Источник |\n|---|---|---|\n| 1 | не более 500 м | [1] |\n"
        "| 2 | 1 км<br>для сёл | [2] |\n\nГотово [1].\nКонец | не таблица"
    )
    expected = tables_to_lists(text)
    rng = random.Random(0)
    for _ in range(200):
        rewriter, out, index = StreamingTableRewriter(), "", 0
        while index < len(text):
            size = rng.randint(1, 9)
            out += rewriter.feed(text[index : index + size])
            index += size
        assert out + rewriter.flush() == expected


# ── MCP client ─────────────────────────────────────────────────────────


class RecordingMcp:
    def __init__(self, data):
        self.data = data
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def call_tool(self, name, arguments, *, meta, timeout):
        self.calls.append((name, arguments, timeout))
        return SimpleNamespace(data=self.data, meta=None)


@pytest.mark.asyncio
async def test_list_restrictions_pages_with_a_bounded_wait():
    mcp = RecordingMcp({"count": 1, "hits": [{"id": "r1"}], "next_after_id": "r1"})
    client = NormGraphMcpClient(mcp)

    page = await client.list_restrictions(limit=200, after_id="r0")

    assert page == {"count": 1, "hits": [{"id": "r1"}], "next_after_id": "r1"}
    assert mcp.calls == [("list_restrictions", {"after_id": "r0", "limit": 200}, 120.0)]
