import pytest

from src.agents.services.dvd.dvd_context import DvdContextBuilder
from src.agents.services.dvd.dvd_reasoning import RetrievalPlanner
from src.agents.services.service_entities.dvd_plan import SemanticRetrievalPlan


@pytest.mark.parametrize(
    "query,pattern,document",
    [
        ("Что в пункте 3.3 раздела 3 СП 55?", "раздел 3 / 3.3", "СП 55"),
        ("В разделе 3 СП 55 что в пункте 3.3?", "раздел 3 / 3.3", "СП 55"),
        ("Что в п. 3.3 раздела II СП 55?", "раздел II / 3.3", "СП 55"),
        (
            "Что в статье 19 конституции?",
            "статья 19",
            "Конституция Российской Федерации",
        ),
        (
            "Что в части 1 статьи 19 Конституции РФ?",
            "статья 19 / 1",
            "Конституция Российской Федерации",
        ),
        (
            "ГрК РФ, ст. 52, ч. 3.3",
            "статья 52 / 3.3",
            "Градостроительный кодекс Российской Федерации",
        ),
        ("п. 3.3 град кодекса", "3.3", "Градостроительный кодекс Российской Федерации"),
    ],
)
def test_explicit_reference_survives_minimal_model_plan(query, pattern, document):
    plan = RetrievalPlanner._clamp(
        SemanticRetrievalPlan(retrieval_mode="semantic"), query
    )
    assert plan.pattern == pattern
    assert plan.document_names == [document]
    assert plan.retrieval_mode == "structure"


def test_target_context_cannot_attribute_neighbour_to_clause():
    context = DvdContextBuilder().build_context(
        [
            dict(
                id="one",
                numbering="3.3",
                text="Блокированная застройка.",
                context="Зелёный дом. Блокированная застройка.",
            )
        ]
    )
    assert "Блокированная застройка." in context
    assert "Зелёный дом" not in context


async def test_full_quote_retains_paginated_condition_even_if_explanation_omits_it(
    service, fake_llm
):
    from tests.helpers import answer_text, plan_json, verdict_json
    from tests.unit.test_dvd_structured_retrieval import Pages, run

    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Пункт перечисляет меры экономии энергии [1]."]
    root = dict(
        id="root",
        name="СП 55",
        numbering="10.6",
        text="10.6 Предусматривают меры:",
        context="Соседний пункт.",
    )
    child = dict(
        id="child",
        name="СП 55",
        text="Если меры обеспечивают соблюдение условий, допускается снижение.",
        matched=False,
        matched_ancestor_ids=["root"],
    )
    client = Pages(
        [
            dict(hits=[root], total=2, complete=False, next_cursor="second"),
            dict(hits=[child], total=2, complete=True),
        ]
    )
    events = await run(service, client, "Что написано в пункте 10.6 СП 55? Объясни.")
    answer = answer_text(events)
    assert "Пункт перечисляет меры экономии энергии [1]." in answer
    assert "> " + root["text"] in answer
    assert "> " + child["text"] in answer
    assert "Соседний пункт" not in answer
    assert len(client.calls) == 2
    assert answer.count("Полная цитата:") == 1


async def test_explicit_quote_does_not_allow_model_rewriting(service, fake_llm):
    from tests.helpers import answer_text, plan_json
    from tests.unit.test_dvd_structured_retrieval import Pages, run

    fake_llm.json_responses = [plan_json()]
    text = "3.3 Требование действует только при условии А."
    events = await run(
        service,
        Pages([dict(hits=[dict(id="r", text=text)], total=1, complete=True)]),
        "Процитируй пункт 3.3 СП 55",
    )
    assert "> " + text in answer_text(events)
    assert len(fake_llm.chat_calls) == 1


def test_quote_preserves_original_text_and_uses_article_label():
    quote = DvdContextBuilder().full_quote(
        [
            dict(
                id="internal-id",
                type="article",
                numbering="19",
                text="Нормализованный текст",
                source_text="Статья 19\n\nИсходный текст.",
                source_file_url="https://dvd.example/source.docx",
            )
        ]
    )
    assert "статья 19" in quote
    assert "> Статья 19\n> \n> Исходный текст." in quote
    assert "Нормализованный текст" not in quote
    assert "internal-id" not in quote
    assert "[исходный документ](https://dvd.example/source.docx)" in quote


def test_unrelated_foreign_constitution_is_not_replaced_with_russian():
    from src.agents.services.dvd.document_reference import parse_reference

    assert not parse_reference("Что в статье 19 Конституции Казахстана?").document_names


def test_reference_is_invariant_under_all_component_permutations():
    from itertools import permutations

    from src.agents.services.dvd.document_reference import parse_reference

    for parts in permutations(["СП 55", "раздел 3", "пункт 3.3"]):
        ref = parse_reference(", ".join(parts))
        assert ref.pattern == "раздел 3 / 3.3"
        assert ref.document_names == ["СП 55"]


@pytest.mark.parametrize(
    "suffix", ["вместе с подпунктами.", "и его подпункты.", "с примечаниями."]
)
def test_inflected_word_is_not_a_letter_address(suffix):
    from src.agents.services.dvd.document_reference import parse_reference

    ref = parse_reference("Процитируй пункт 10.6 СП 55 " + suffix)
    assert ref.pattern == "10.6"


def test_quote_and_context_share_source_order_when_payload_order_is_missing():
    hits = [
        dict(doc_id="d", id="child", text="Дочерний текст", char_start=20),
        dict(doc_id="d", id="parent", text="Статья 19", char_start=0),
    ]
    builder = DvdContextBuilder()
    for text in (builder.build_context(hits), builder.full_quote(hits)):
        assert text.index("Статья 19") < text.index("Дочерний текст")


@pytest.mark.parametrize("label", ["[N]", "[99]"])
async def test_unknown_source_label_cannot_be_approved_by_model(fake_llm, label):
    from src.agents.services.dvd.dvd_reasoning import AnswerCritic

    verdict = await AnswerCritic(fake_llm).review(
        "m", "q", "[1] СП 55\nТекст", "Объяснение " + label
    )
    assert not verdict.satisfied
    assert not fake_llm.chat_calls


async def test_explanation_cannot_deny_the_text_quoted_in_same_answer(fake_llm):
    from src.agents.services.dvd.dvd_reasoning import AnswerCritic

    verdict = await AnswerCritic(fake_llm).review(
        "m",
        "q",
        "[1] Источник\nКонтрольный пункт раздела II.",
        "Сам текст пункта не приведён.\n\nПолная цитата:\n> Контрольный пункт раздела II.",
    )
    assert not verdict.satisfied
    assert not fake_llm.chat_calls


async def test_absence_check_handles_reversed_words_and_decimal_address(fake_llm):
    from src.agents.services.dvd.dvd_reasoning import AnswerCritic

    verdict = await AnswerCritic(fake_llm).review(
        "m",
        "q",
        "[1] Источник\nКонтрольный пункт раздела II.",
        "Текст самого пункта 3.3 не приведён.\n\nПолная цитата:\n> Контрольный пункт раздела II.",
    )
    assert not verdict.satisfied
    assert not fake_llm.chat_calls


async def test_absence_check_handles_negation_before_text(fake_llm):
    from src.agents.services.dvd.dvd_reasoning import AnswerCritic

    verdict = await AnswerCritic(fake_llm).review(
        "m",
        "q",
        "[1] Источник\nКонтрольный пункт раздела II.",
        "В источнике нет текста пункта 3.3.\n\nПолная цитата:\n> Контрольный пункт раздела II.",
    )
    assert not verdict.satisfied
    assert not fake_llm.chat_calls
