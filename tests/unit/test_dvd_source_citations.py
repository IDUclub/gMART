"""The user reads document and clause references instead of source labels."""

import pytest

from src.agents.services.dvd.source_citations import citations, readable
from tests.helpers import FakeDvdMcpClient, answer_text, plan_json
from tests.unit.test_dvd_answer_revision import ACCESS, SOURCE, audit
from tests.unit.test_dvd_rag_service import _run

pytestmark = pytest.mark.usefixtures("legacy_critic_quotes")

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
    "name, version, expected",
    [
        ("Приказ N 111", "ред. от 06.12.2024", "Приказ N 111, ред. от 06.12.2024"),
        ("Приказ N 111", "2016", "Приказ N 111, ред. 2016"),
        (
            "Приказ N 111",
            "с изм. от 06.12.2024",
            "Приказ N 111, ред. с изм. от 06.12.2024",
        ),
        (
            "Приказ N 111",
            "Приказ от 25.04.2016 N 111 (ред. от 06.12.2024)",
            "Приказ от 25.04.2016 N 111 (ред. от 06.12.2024)",
        ),
        (
            "ПРИКАЗ жилищно‑коммунального хозяйства N 111",
            "Приказ  жилищно-коммунального хозяйства от 25.04.2016 N 111",
            "Приказ жилищно-коммунального хозяйства от 25.04.2016 N 111",
        ),
        (
            "Министерства строительства N 111",
            "Приказ Министерства строительства от 25.04.2016 N 111",
            "Приказ Министерства строительства от 25.04.2016 N 111",
        ),
        (
            "СП 42.13330.2016 Планировка и застройка",
            "Планировка и застройка (ред. от 06.12.2024)",
            "СП 42.13330.2016 Планировка и застройка, ред. "
            "Планировка и застройка (ред. от 06.12.2024)",
        ),
        (
            "Приказ N 111 СП 42.13330.2016",
            "Приказ N 111 (ред. от 06.12.2024)",
            "Приказ N 111 СП 42.13330.2016, ред. Приказ N 111 (ред. от 06.12.2024)",
        ),
    ],
)
def test_document_title_preserves_name_information(name, version, expected):
    assert citations([{"name": name, "version": version}]) == {"[1]": [expected, ""]}


def test_regional_act_titles_from_production_share_one_reference():
    issuer = (
        "Приказ Министерства строительства, жилищно‑коммунального хозяйства "
        "и энергетики Республики Карелия"
    )
    version_a = f"{issuer} от 25.04.2016 N 111 (ред. от 06.12.2024)"
    version_b = f"{issuer} от 25 апреля 2016 г. N 111 (ред. от 06.12.2024)"
    hits = [
        {"name": f"{issuer} N 111", "version": version_a, "numbering": "1"},
        {"name": issuer, "version": version_b, "numbering": "2"},
    ]
    refs = citations(hits)
    assert refs == {"[1]": [version_a, "п. 1"], "[2]": [version_b, "п. 2"]}
    assert readable("Требование [1], [2].", refs) == (
        f"Требование ({version_b}, п. 1, п. 2)."
    )


@pytest.mark.parametrize(
    "other",
    [
        "Приказ Министерства от 25.04.2016 N 112 (ред. от 06.12.2024)",
        "Приказ Ведомства от 25.04.2016 N 111 (ред. от 06.12.2024)",
        "Приказ Министерства от 26.04.2016 N 111 (ред. от 06.12.2024)",
        "Приказ Министерства от 25.04.2016 N 111 (ред. от 07.12.2024)",
        "Приказ Министерства от 25.04.2016 N 111",
    ],
)
def test_different_acts_and_editions_stay_separate(other):
    document = "Приказ Министерства от 25.04.2016 N 111 (ред. от 06.12.2024)"
    refs = citations(
        [{"name": document, "numbering": "1"}, {"name": other, "numbering": "2"}]
    )
    assert readable("Требование [1][2].", refs) == (
        f"Требование ({document}, п. 1; {other}, п. 2)."
    )


@pytest.mark.parametrize("hyphen", ["‐", "‑", "–", "-"])
@pytest.mark.parametrize("month", ["января", "апреля", "декабря"])
def test_document_identity_normalises_typography_and_dates(hyphen, month):
    number = {"января": "01", "апреля": "04", "декабря": "12"}[month]
    document = f"Приказ «Жилищно‑коммунальное хозяйство» от 05.{number}.2016 N 111"
    other = (
        f'ПРИКАЗ  "Жилищно{hyphen}коммунальное хозяйство" '
        f"от 5 {month} 2016 г. N 111"
    )
    refs = citations(
        [{"name": document, "numbering": "1"}, {"name": other, "numbering": "2"}]
    )
    expected = max((" ".join(value.split()) for value in [document, other]), key=len)
    assert readable("Требование [1][2][1].", refs) == (
        f"Требование ({expected}, п. 1, п. 2)."
    )


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


@pytest.mark.parametrize(
    "path, address",
    [
        (["2.3.35 Для парковки легковых автомобилей посетителей"], "п. 2.3.35"),
        (["3.4.15"], "п. 3.4.15"),
        (["35 Статья 35. Виды и состав территориальных зон"], "статья 35"),
        (
            ["Приложение Р Правила подсчета общей площади, строительного"],
            "приложение Р",
        ),
        (["Б.1 Приложение Б Вместимость зрительных залов"], "приложение Б"),
        (["Окончание таблицы 12.11"], "таблица 12.11"),
        (["Спортивные манежи"], "раздел «Спортивные манежи»"),
        # A table row or a sentence is the fragment's text, not its section.
        (
            [
                "Таблица 6.1",
                "Машино-места в гаражах Тип 1-4 (бытовые, медленные, стандартные, "
                "быстрые) 5% (в 2025 году), 10% (в 2026 году)",
            ],
            "таблица 6.1",
        ),
        (
            ["Возможно совмещение в одном помещении туалета и умывальной комнаты"],
            "",
        ),
        (["30.12.2020 N 494-ФЗ)"], ""),
        (["(часть"], ""),
        (["Содержание"], ""),
    ],
)
def test_section_title_becomes_a_short_address(path, address):
    hit = {"name": "СП 1", "structure_path": path, "text": "x"}
    assert citations([hit])["[1]"] == ["СП 1", address]


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
