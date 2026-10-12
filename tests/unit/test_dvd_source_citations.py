"""The user reads document and clause references instead of source labels."""

import pytest

from src.agents.services.dvd.source_citations import citations, readable
from tests.helpers import FakeDvdMcpClient, answer_text, plan_json
from tests.unit.test_dvd_answer_revision import ACCESS, SOURCE, audit
from tests.unit.test_dvd_rag_service import _run

HITS = [
    {
        "name": "СП 55.13330.2016",
        "version": "2016",
        "numbering": "9.18",
        "text": "a",
        "char_start": 10,
    },
    {
        "name": "СП 55.13330.2016",
        "version": "2016",
        "numbering": "9.17",
        "text": "b",
        "char_start": 5,
    },
    {
        "name": "СП 42.13330.2016",
        "version": "ред. от 31.05.2022",
        "type": "table",
        "numbering": "7.1",
        "text": "c",
    },
    {
        "name": "СП 42.13330.2016",
        "version": "ред. от 31.05.2022",
        "structure_path": ["Общие положения"],
        "text": "d",
    },
]


@pytest.fixture(autouse=True)
def _readable(monkeypatch):
    monkeypatch.setenv("DVD_READABLE_CITATIONS", "true")


def test_labels_follow_the_context_order():
    # Within one document sources are numbered in reading order.
    assert citations(HITS) == {
        "[1]": ["СП 55.13330.2016", "п. 9.17"],
        "[2]": ["СП 55.13330.2016", "п. 9.18"],
        "[3]": ["СП 42.13330.2016, ред. от 31.05.2022", "таблица 7.1"],
        "[4]": ["СП 42.13330.2016, ред. от 31.05.2022", "раздел «Общие положения»"],
    }


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Ширина 6 м [2].", "Ширина 6 м (СП 55.13330.2016, п. 9.18)."),
        (
            "- Требование [1], [2].",
            "- Требование (СП 55.13330.2016, п. 9.17, п. 9.18).",
        ),
        (
            "Значения [2][3].",
            "Значения (СП 55.13330.2016, п. 9.18; "
            "СП 42.13330.2016, ред. от 31.05.2022, таблица 7.1).",
        ),
        (
            "Общее правило [4]",
            "Общее правило (СП 42.13330.2016, ред. от 31.05.2022, раздел «Общие положения»)",
        ),
        # A label without a source points at nothing the user can find.
        ("Утверждение [9].", "Утверждение."),
    ],
)
def test_label_runs_become_one_reference(text, expected):
    assert readable(text, citations(HITS)) == expected


def test_quoted_source_text_keeps_its_own_references():
    text = (
        "Смысл пункта [2].\n\nПолная цитата:\n\n"
        "[2] СП 55.13330.2016, ред. 2016, п. 9.18, полный исходный текст фрагмента\n\n"
        "> 9.18 Следует предусматривать инсоляцию [6]."
    )
    assert readable(text, citations(HITS)) == (
        "Смысл пункта (СП 55.13330.2016, п. 9.18).\n\nПолная цитата:\n\n"
        "СП 55.13330.2016, ред. 2016, п. 9.18, полный исходный текст фрагмента\n\n"
        "> 9.18 Следует предусматривать инсоляцию [6]."
    )


def test_switch_keeps_labels(monkeypatch):
    monkeypatch.setenv("DVD_READABLE_CITATIONS", "false")
    assert readable("Ширина [2].", citations(HITS)) == "Ширина [2]."


async def test_final_answer_names_document_and_clause(service, fake_llm):
    client = FakeDvdMcpClient(
        hits_per_call=[[{"name": "СП 257", "numbering": "6.1.11", "text": SOURCE}]]
    )
    fake_llm.json_responses = [
        plan_json(),
        audit([(ACCESS, "supported", SOURCE[-50:])], satisfied=True),
    ]
    fake_llm.answer_texts = [ACCESS]
    events = await _run(service, client)
    assert answer_text(events) == (
        "В гостиницах обеспечивается доступ для МГН (СП 257, п. 6.1.11)."
    )
    system = [c for c in fake_llm.chat_calls if c.stream][0].messages[0]["content"]
    assert "приложение само заменит метку названием документа" in system
