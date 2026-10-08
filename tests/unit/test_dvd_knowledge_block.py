"""Documents first, then the model's own knowledge under a marked heading."""

from types import SimpleNamespace

import pytest

from src.agents.services.dvd import knowledge_block
from src.agents.services.dvd.knowledge_block import (
    KNOWLEDGE_TITLE,
    NO_GROUNDED_ANSWER,
    render,
    sort_lines,
    split,
    strip_external_referrals,
)
from tests.helpers import FakeDvdMcpClient, answer_text, plan_json
from tests.unit.test_dvd_answer_revision import ACCESS, ROOF, SOURCE, audit
from tests.unit.test_dvd_rag_service import _run

GENERAL = "Кровли гостиниц обычно проектируют с учётом требований к эксплуатации [1]."


def claim(text, status):
    return SimpleNamespace(text=text, status=status)


@pytest.mark.parametrize(
    "heading",
    [
        "Без опоры на загруженные документы:",
        "**Без опоры на загруженные документы:**",
        "### Без опоры на загруженные документы",
    ],
)
def test_split_separates_the_marked_block(heading):
    grounded, knowledge = split(f"Ответ [1].\n\n{heading}\n- Общее пояснение.")
    assert grounded == "Ответ [1]."
    assert knowledge == ["- Общее пояснение."]


def test_split_keeps_text_on_the_heading_line():
    grounded, knowledge = split(
        "Ответ [1].\n**Без опоры на загруженные документы:** Обычно так."
    )
    assert (grounded, knowledge) == ("Ответ [1].", ["Обычно так."])


def test_split_without_block_returns_the_draft():
    assert split("Ответ [1].") == ("Ответ [1].", [])


@pytest.mark.parametrize(
    "advice",
    [
        "Рекомендуется обратиться к официальному тексту СП.",
        "Уточните актуальную редакцию в КонсультантПлюс.",
        "Подробнее: https://docs.cntd.ru/document/1.",
        "Для точного ответа проконсультируйтесь со специалистом.",
        "Проверьте требования на официальном сайте Минстроя.",
        "Можно найти полный текст документа в справочной системе.",
    ],
)
def test_external_referrals_are_removed(advice):
    text = f"Ширина проезда — 6 м [1].\nОбщее пояснение. {advice}"
    assert (
        strip_external_referrals(text) == "Ширина проезда — 6 м [1].\nОбщее пояснение."
    )


@pytest.mark.parametrize(
    "line",
    [
        # A retrieved norm may itself name a body or an official publication.
        "Проект согласуют с органом местного самоуправления [1].",
        "Уточните вопрос или укажите конкретный документ.",
        "Проверьте, относится ли объект к жилым зданиям.",
    ],
)
def test_internal_wording_is_kept(line):
    assert strip_external_referrals(line) == line


def test_sorting_keeps_confirmed_lines_and_moves_the_rest():
    grounded = "\n".join(
        ["Требования:", f"- {ACCESS}", f"- {GENERAL}", f"- {ROOF}", "- Высота 3 м [1]."]
    )
    kept, moved, counts = sort_lines(
        grounded,
        [
            claim(ACCESS, "supported"),
            claim(GENERAL, "insufficient"),
            claim(ROOF, "contradicted"),
            claim("Высота 3 м [1].", "insufficient"),
        ],
    )
    assert kept == f"Требования:\n- {ACCESS}"
    # Labels go: the line is no longer attributed to a fragment.
    assert moved == [
        "- Кровли гостиниц обычно проектируют с учётом требований к эксплуатации."
    ]
    assert counts == {"supported": 1, "moved": 1, "dropped": 2}


def test_nothing_confirmed_leaves_no_grounded_text():
    kept, moved, _ = sort_lines(
        f"Согласно документам:\n- {GENERAL}", [claim(GENERAL, "insufficient")]
    )
    assert kept == ""
    assert render(kept, moved) == (
        f"{NO_GROUNDED_ANSWER}\n\n{KNOWLEDGE_TITLE}\n"
        "- Кровли гостиниц обычно проектируют с учётом требований к эксплуатации."
    )


def test_knowledge_block_has_no_figures_or_clause_numbers():
    answer = render(
        f"{ACCESS}",
        [
            "- Тему обычно регулирует СП 42.13330.",
            "- Ширина проезда не менее 6 м.",
            "- Требование есть в п. 7.1.",
            "- Обратитесь к официальному тексту документа.",
        ],
    )
    assert (
        answer
        == f"{ACCESS}\n\n{KNOWLEDGE_TITLE}\n- Тему обычно регулирует СП 42.13330."
    )


def test_quotation_stays_with_the_grounded_part():
    answer = render("", ["- Общее пояснение."], "Полная цитата:\n> текст")
    assert answer == f"Полная цитата:\n> текст\n\n{KNOWLEDGE_TITLE}\n- Общее пояснение."


async def test_one_audit_sorts_the_answer_without_a_rewrite(service, fake_llm):
    client = FakeDvdMcpClient(hits_per_call=[[{"name": "СП 257", "text": SOURCE}]])
    fake_llm.json_responses = [
        plan_json(),
        audit(
            [
                (ACCESS, "supported", SOURCE[-50:]),
                (GENERAL, "insufficient", ""),
                (ROOF, "contradicted", ""),
            ]
        ),
    ]
    fake_llm.answer_texts = [
        f"- {ACCESS}\n- {GENERAL}\n- {ROOF}\n\n"
        "Без опоры на загруженные документы:\n"
        "- Доступность обычно обеспечивают пандусами.\n"
        "- Подробности уточните на официальном сайте."
    ]
    events = await _run(service, client)

    assert answer_text(events) == (
        f"- {ACCESS}\n\n{KNOWLEDGE_TITLE}\n"
        "- Кровли гостиниц обычно проектируют с учётом требований к эксплуатации.\n"
        "- Доступность обычно обеспечивают пандусами."
    )
    audits = [c for c in fake_llm.chat_calls if not c.stream][1:]
    assert len(audits) == 1
    # The marked block is not audited against the fragments.
    assert "пандусами" not in audits[0].messages[-1]["content"]
    assert len([c for c in fake_llm.chat_calls if c.stream]) == 1
    assert len(client.search_calls) == 1


async def test_nothing_confirmed_searches_once_more(service, fake_llm):
    client = FakeDvdMcpClient(
        hits_per_call=[
            [{"name": "СП 257", "text": "Посторонний текст."}],
            [{"name": "СП 257", "text": SOURCE}],
        ]
    )
    fake_llm.json_responses = [
        plan_json(),
        audit([(GENERAL, "insufficient", "")]),
        plan_json(),
        audit([(ACCESS, "supported", SOURCE[-50:])], satisfied=True),
    ]
    fake_llm.answer_texts = [GENERAL, ACCESS]
    events = await _run(service, client)
    assert answer_text(events) == ACCESS
    assert len(client.search_calls) >= 2


def test_strict_mode_keeps_the_old_prompt(monkeypatch):
    monkeypatch.setenv("DVD_KNOWLEDGE_FALLBACK", "false")
    assert not knowledge_block.enabled()
    monkeypatch.delenv("DVD_KNOWLEDGE_FALLBACK")
    assert knowledge_block.enabled()


async def test_answer_prompt_forbids_external_sources(service, fake_llm):
    client = FakeDvdMcpClient(hits_per_call=[[{"name": "СП 257", "text": SOURCE}]])
    fake_llm.json_responses = [
        plan_json(),
        audit([(ACCESS, "supported", SOURCE[-50:])], satisfied=True),
    ]
    fake_llm.answer_texts = [ACCESS]
    await _run(service, client)
    system = [c for c in fake_llm.chat_calls if c.stream][0].messages[0]["content"]
    assert "Без опоры на загруженные документы" in system
    assert "Никогда не советуй обращаться к внешним источникам" in system
    assert "СТРОГО" not in system
