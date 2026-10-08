"""Stage 1: the critic's reasoning effort follows the risk of the audited answer."""

from __future__ import annotations

import pytest

from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter
from src.agents.services.dvd.answer_risk import assess_risk
from src.agents.services.dvd.dvd_reasoning import critic_reasoning_effort

ONE_DOCUMENT = {
    "[1]": ("doc-1", "СП 42.13330.2016", "2016"),
    "[2]": ("doc-1", "СП 42.13330.2016", "2016"),
}


@pytest.mark.parametrize(
    "answer",
    [
        "Красная линия — граница территорий общего пользования [1].",
        # Clause, document and edition numbers identify the source, not a quantity.
        "Пункт 7.5 СП 42.13330, ред. 2016, посвящён озеленению жилых районов [1].",
        # The verbatim quotation is not a model claim.
        "Пункт описывает озеленение микрорайона [1].\n\nПолная цитата:\n\n"
        "[1] СП 42.13330, п. 7.5\n\n> не менее 6 м² на человека",
    ],
)
def test_short_single_source_statement_is_low_risk(answer):
    assert assess_risk(answer, ONE_DOCUMENT).low


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ("Площадь озеленения — 6 м² на человека [1].", "numbers"),
        ("Расстояние составляет от 10 до 15 метров [1].", "numbers"),
        ("Проезд должен быть обеспечен с двух сторон [1].", "obligation"),
        ("Размещение не допускается на участке школы [1].", "obligation"),
        ("Правило действует, за исключением реконструкции [1].", "conditions"),
        (
            "В найденных фрагментах нет сведений о вертолётных площадках.",
            "absence_claim",
        ),
        ("Озеленение описано в своде правил.", "no_citations"),
        ("Термин определён в своде правил [1], [2].", "multi_source_claim"),
        (
            "Термин определён [1].\nЕго уточняет примечание [2].\nСм. также [3].",
            "many_sources",
        ),
    ],
)
def test_risky_answer_keeps_full_audit(answer, reason):
    risk = assess_risk(answer, {**ONE_DOCUMENT, "[3]": ONE_DOCUMENT["[1]"]})
    assert not risk.low
    assert reason in risk.reasons


def test_two_documents_or_editions_are_high_risk():
    answer = "Термин определён в СП [1].\nТермин также определён в кодексе [2]."
    documents = {"[1]": ("a", "СП 42", "2016"), "[2]": ("b", "ГрК РФ", "2024")}
    assert "multiple_documents" in assess_risk(answer, documents).reasons
    editions = {"[1]": ("a", "СП 42", "2016"), "[2]": ("b", "СП 42", "2011")}
    assert "multiple_editions" in assess_risk(answer, editions).reasons


def test_pipeline_state_makes_any_answer_high_risk():
    answer = "Красная линия — граница территорий общего пользования [1]."
    assert "previous_rejection" in assess_risk(answer, rejected_before=True).reasons
    assert "partial_context" in assess_risk(answer, context_incomplete=True).reasons
    assert "document_list" in assess_risk(answer, intent="document_list").reasons


@pytest.fixture
def gpt_oss():
    return OpenAiCompatAdapter("http://llm:8000/v1")


def test_low_risk_audit_uses_low_effort(gpt_oss, monkeypatch):
    monkeypatch.delenv("DVD_ADAPTIVE_CRITIC", raising=False)
    monkeypatch.delenv("DVD_CRITIC_REASONING_EFFORT", raising=False)
    low = assess_risk("Термин определён в своде правил [1].", ONE_DOCUMENT)
    high = assess_risk("Не менее 6 м² [1].", ONE_DOCUMENT)
    assert critic_reasoning_effort(gpt_oss, "openai/gpt-oss-20b", low) == "low"
    assert critic_reasoning_effort(gpt_oss, "openai/gpt-oss-20b", high) == "medium"
    # An unclassified audit (e.g. partial-answer selection) keeps the full effort.
    assert critic_reasoning_effort(gpt_oss, "openai/gpt-oss-20b") == "medium"
    # Other models keep their own default.
    assert critic_reasoning_effort(gpt_oss, "qwen3", low) is None


def test_adaptive_critic_switch_restores_fixed_effort(gpt_oss, monkeypatch):
    monkeypatch.setenv("DVD_ADAPTIVE_CRITIC", "false")
    low = assess_risk("Термин определён в своде правил [1].", ONE_DOCUMENT)
    assert critic_reasoning_effort(gpt_oss, "openai/gpt-oss-20b", low) == "medium"


async def test_pipeline_passes_risk_to_critic(service, fake_llm, fake_mcp):
    from tests.helpers import plan_json, verdict_json

    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Не менее 6 м² на человека [1]."]
    seen = []
    review = service.critic.review

    async def spy(*args, **kwargs):
        seen.append(kwargs.get("risk"))
        return await review(*args, **kwargs)

    service.critic.review = spy
    async for _ in service.run_document_qa_pipeline(
        dvd_mcp_client=fake_mcp,
        token="tok",
        model="m",
        temperature=0.0,
        user_query="нормы озеленения",
        chat_id="chat-1",
    ):
        pass
    (risk,) = seen
    assert not risk.low
    assert {"numbers", "obligation"} <= set(risk.reasons)
